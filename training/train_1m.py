import sys
import time
import json
import csv
import pickle
import argparse
import logging
from pathlib import Path

import numpy as np
import torch
from stable_baselines3 import DQN, PPO, A2C
from stable_baselines3.common.callbacks import BaseCallback

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from environment.custom_env import PowerGridEnv

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CHECKPOINT_DIR = PROJECT_ROOT / "models" / "checkpoints" / "1m"
FINAL_MODEL_DIR = PROJECT_ROOT / "models"
LOG_DIR = PROJECT_ROOT / "logs"
RESULTS_LOG_DIR = PROJECT_ROOT / "results" / "logs"
TOTAL_TIMESTEPS = 1_000_000
CHECKPOINT_STEP_INTERVAL = 25_000
CHECKPOINT_TIME_INTERVAL = 300

BEST_CONFIGS = {
    "dqn": {
        "learning_rate": 5e-4,
        "gamma": 0.95,
        "buffer_size": 50000,
        "batch_size": 64,
        "exploration_fraction": 0.4,
        "exploration_final_eps": 0.02,
        "target_update_interval": 1000,
        "net_arch": [200, 200],
    },
    "ppo": {
        "learning_rate": 5e-4,
        "gamma": 0.95,
        "clip_range": 0.3,
        "n_steps": 1024,
        "batch_size": 128,
        "n_epochs": 20,
        "ent_coef": 0.01,
        "gae_lambda": 0.9,
        "net_arch": [128, 128, 128],
    },
    "a2c": {
        "learning_rate": 7e-4,
        "gamma": 0.99,
        "n_steps": 32,
        "ent_coef": 0.01,
        "vf_coef": 0.5,
        "max_grad_norm": 1.0,
        "net_arch": [200, 200, 200],
    },
    "reinforce": {
        "learning_rate": 1e-4,
        "gamma": 0.9,
        "net_arch": [256, 256],
        "max_grad_norm": 1.0,
    },
}

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def setup_logger(algo):
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger(f"1m_{algo}")
    logger.setLevel(logging.DEBUG)
    logger.handlers.clear()
    fh = logging.FileHandler(LOG_DIR / f"1m_{algo}.log")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    logger.addHandler(fh)
    sh = logging.StreamHandler()
    sh.setLevel(logging.INFO)
    sh.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(sh)
    return logger


class LongRunCallback(BaseCallback):
    def __init__(self, algo, logger, checkpoint_dir, log_path, verbose=0):
        super().__init__(verbose)
        self.algo = algo
        self.ext_logger = logger
        self.ckpt_dir = checkpoint_dir
        self.log_path = log_path
        self.episode_rewards = []
        self.episode_lengths = []
        self.losses = []
        self.current_episode_reward = 0.0
        self.current_episode_length = 0
        self.last_checkpoint_step = 0
        self.last_checkpoint_time = time.time()
        self._last_print_step = 0

        RESULTS_LOG_DIR.mkdir(parents=True, exist_ok=True)
        if not self.log_path.exists():
            with open(self.log_path, "w", newline="") as f:
                csv.writer(f).writerow(["timestep", "episode", "episode_reward", "episode_length", "mean_reward_100", "loss"])

    def _on_step(self):
        self.current_episode_reward += self.locals.get("rewards", [0.0])[0]
        self.current_episode_length += 1

        dones = self.locals.get("dones", [False])
        if len(dones) > 0 and dones[0]:
            self.episode_rewards.append(float(self.current_episode_reward))
            self.episode_lengths.append(int(self.current_episode_length))
            self._write_csv()
            self.current_episode_reward = 0.0
            self.current_episode_length = 0

        if hasattr(self.model, "logger") and self.model.logger is not None:
            try:
                loss_val = self.model.logger.name_to_value.get("train/loss")
                if loss_val is not None:
                    self.losses.append(float(loss_val))
            except Exception:
                pass

        steps_since = self.num_timesteps - self.last_checkpoint_step
        time_since = time.time() - self.last_checkpoint_time
        if steps_since >= CHECKPOINT_STEP_INTERVAL or time_since >= CHECKPOINT_TIME_INTERVAL:
            self._save_checkpoint()

        return True

    def _write_csv(self):
        ep_num = len(self.episode_rewards)
        ep_reward = self.episode_rewards[-1]
        ep_length = self.episode_lengths[-1]
        mean_100 = float(np.mean(self.episode_rewards[-100:]))
        loss = self.losses[-1] if self.losses else 0.0

        with open(self.log_path, "a", newline="") as f:
            csv.writer(f).writerow([
                self.num_timesteps, ep_num, f"{ep_reward:.4f}", ep_length, f"{mean_100:.4f}", f"{loss:.6f}",
            ])

        steps_since_print = self.num_timesteps - self._last_print_step
        if steps_since_print >= 50000:
            self._last_print_step = self.num_timesteps
            self.ext_logger.info(
                f"  step {self.num_timesteps:>7} | ep {ep_num:>5} | "
                f"reward {ep_reward:>7.2f} | mean100 {mean_100:>7.2f}"
            )

    def _save_checkpoint(self):
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)
        self.model.save(str(self.ckpt_dir / "model"))

        if hasattr(self.model, "save_replay_buffer"):
            try:
                self.model.save_replay_buffer(str(self.ckpt_dir / "replay_buffer"))
            except Exception:
                pass

        meta = {
            "timestep": int(self.num_timesteps),
            "episode_count": len(self.episode_rewards),
            "episode_rewards": self.episode_rewards[-200:],
            "episode_lengths": self.episode_lengths[-200:],
            "losses": [float(l) for l in self.losses[-500:]],
            "mean_reward_100": float(np.mean(self.episode_rewards[-100:])) if self.episode_rewards else 0.0,
        }
        with open(self.ckpt_dir / "meta.json", "w") as f:
            json.dump(meta, f)

        rng_state = {"numpy": np.random.get_state(), "torch": torch.random.get_rng_state()}
        if torch.cuda.is_available():
            rng_state["torch_cuda"] = torch.cuda.get_rng_state_all()
        with open(self.ckpt_dir / "rng_state.pkl", "wb") as f:
            pickle.dump(rng_state, f)

        self.last_checkpoint_step = self.num_timesteps
        self.last_checkpoint_time = time.time()
        self.ext_logger.debug(f"checkpoint at step {self.num_timesteps}")


def train_sb3(algo_name, algo_class, resume=False):
    logger = setup_logger(algo_name)
    params = BEST_CONFIGS[algo_name]
    ckpt_dir = CHECKPOINT_DIR / algo_name
    completed_marker = ckpt_dir / "COMPLETED"
    log_path = RESULTS_LOG_DIR / f"1m_{algo_name}.csv"

    if completed_marker.exists():
        logger.info(f"{algo_name} 1m run already completed, skipping")
        return

    logger.info(f"{algo_name} 1m training starting")
    logger.info(f"  params: {params}")
    logger.info(f"  device: {DEVICE}")

    env = PowerGridEnv()
    net_arch = params.pop("net_arch", [128, 128])
    remaining_steps = TOTAL_TIMESTEPS

    if resume and (ckpt_dir / "model.zip").exists():
        logger.info(f"  resuming from checkpoint")
        model = algo_class.load(str(ckpt_dir / "model"), env=env, device=DEVICE)

        if hasattr(model, "load_replay_buffer"):
            try:
                model.load_replay_buffer(str(ckpt_dir / "replay_buffer"))
            except Exception:
                pass

        if (ckpt_dir / "meta.json").exists():
            with open(ckpt_dir / "meta.json") as f:
                meta = json.load(f)
            completed_steps = meta.get("timestep", 0)
            remaining_steps = TOTAL_TIMESTEPS - completed_steps
            logger.info(f"  resumed at step {completed_steps}, {remaining_steps} remaining")

        if (ckpt_dir / "rng_state.pkl").exists():
            with open(ckpt_dir / "rng_state.pkl", "rb") as f:
                rng_state = pickle.load(f)
            np.random.set_state(rng_state["numpy"])
            torch.random.set_rng_state(rng_state["torch"])
            if torch.cuda.is_available() and "torch_cuda" in rng_state:
                torch.cuda.set_rng_state_all(rng_state["torch_cuda"])
    else:
        model = algo_class(
            "MlpPolicy", env,
            **params,
            policy_kwargs={"net_arch": net_arch},
            verbose=0,
            device=DEVICE,
        )

    params["net_arch"] = net_arch

    callback = LongRunCallback(algo_name, logger, ckpt_dir, log_path)

    if remaining_steps <= 0:
        logger.info(f"  already at {TOTAL_TIMESTEPS} steps")
    else:
        start = time.time()
        model.learn(total_timesteps=remaining_steps, callback=callback, reset_num_timesteps=False)
        elapsed = time.time() - start
        logger.info(f"  finished in {elapsed:.1f}s")

    final_dir = FINAL_MODEL_DIR / ("dqn" if algo_name == "dqn" else "pg")
    final_dir.mkdir(parents=True, exist_ok=True)
    model.save(str(final_dir / f"1m_{algo_name}_best"))
    logger.info(f"  saved to {final_dir / f'1m_{algo_name}_best'}")

    if callback.episode_rewards:
        summary = {
            "mean_reward": float(np.mean(callback.episode_rewards)),
            "std_reward": float(np.std(callback.episode_rewards)),
            "final_100_mean": float(np.mean(callback.episode_rewards[-100:])),
            "episodes": len(callback.episode_rewards),
        }
        with open(final_dir / f"1m_{algo_name}_best.meta.json", "w") as f:
            json.dump(summary, f, indent=2)

    completed_marker.mkdir(parents=True, exist_ok=True) if not ckpt_dir.exists() else None
    (ckpt_dir / "COMPLETED").touch()
    env.close()


def train_reinforce(resume=False):
    from training.pg_training import ReinforceAgent

    algo_name = "reinforce"
    logger = setup_logger(algo_name)
    params = BEST_CONFIGS[algo_name]
    ckpt_dir = CHECKPOINT_DIR / algo_name
    completed_marker = ckpt_dir / "COMPLETED"
    log_path = RESULTS_LOG_DIR / f"1m_{algo_name}.csv"

    if completed_marker.exists():
        logger.info(f"{algo_name} 1m run already completed, skipping")
        return

    logger.info(f"{algo_name} 1m training starting")
    logger.info(f"  params: {params}")
    logger.info(f"  device: {DEVICE}")

    env = PowerGridEnv()
    agent = ReinforceAgent(
        env.observation_space.shape[0],
        env.action_space.n,
        lr=params["learning_rate"],
        gamma=params["gamma"],
        net_arch=params["net_arch"],
        max_grad_norm=params["max_grad_norm"],
    )

    total_steps = 0
    last_save_time = time.time()
    last_print_step = 0

    ckpt_dir.mkdir(parents=True, exist_ok=True)
    if not log_path.exists():
        with open(log_path, "w", newline="") as f:
            csv.writer(f).writerow(["timestep", "episode", "episode_reward", "episode_length", "mean_reward_100", "loss"])

    if resume and (ckpt_dir / "reinforce.pt").exists():
        agent.load(str(ckpt_dir / "reinforce.pt"))
        if (ckpt_dir / "meta.json").exists():
            with open(ckpt_dir / "meta.json") as f:
                meta = json.load(f)
            total_steps = meta.get("timestep", 0)
            agent.episode_count = meta.get("episode_count", 0)
            agent.episode_rewards_history = meta.get("episode_rewards", [])
        logger.info(f"  resumed at step {total_steps}")

    start = time.time()

    while total_steps < TOTAL_TIMESTEPS:
        obs, _ = env.reset()
        episode_reward = 0.0
        episode_steps = 0

        while True:
            action = agent.select_action(obs)
            obs, reward, terminated, truncated, info = env.step(action)
            agent.rewards.append(reward)
            episode_reward += reward
            episode_steps += 1
            total_steps += 1

            if terminated or truncated:
                agent.update()
                agent.episode_count += 1
                agent.episode_rewards_history.append(float(episode_reward))

                mean_100 = float(np.mean(agent.episode_rewards_history[-100:]))
                with open(log_path, "a", newline="") as f:
                    csv.writer(f).writerow([
                        total_steps, agent.episode_count, f"{episode_reward:.4f}",
                        episode_steps, f"{mean_100:.4f}", "0.0",
                    ])

                if total_steps - last_print_step >= 50000:
                    last_print_step = total_steps
                    logger.info(
                        f"  step {total_steps:>7} | ep {agent.episode_count:>5} | "
                        f"reward {episode_reward:>7.2f} | mean100 {mean_100:>7.2f}"
                    )

                now = time.time()
                if total_steps % CHECKPOINT_STEP_INTERVAL < 1000 or (now - last_save_time) >= CHECKPOINT_TIME_INTERVAL:
                    agent.save(str(ckpt_dir / "reinforce.pt"))
                    meta = {
                        "timestep": total_steps,
                        "episode_count": agent.episode_count,
                        "episode_rewards": [float(r) for r in agent.episode_rewards_history[-200:]],
                    }
                    with open(ckpt_dir / "meta.json", "w") as f:
                        json.dump(meta, f)
                    last_save_time = now

                break

    elapsed = time.time() - start
    logger.info(f"  finished in {elapsed:.1f}s")

    final_dir = FINAL_MODEL_DIR / "pg"
    final_dir.mkdir(parents=True, exist_ok=True)
    agent.save(str(final_dir / "1m_reinforce_best.pt"))
    logger.info(f"  saved to {final_dir / '1m_reinforce_best.pt'}")

    if agent.episode_rewards_history:
        summary = {
            "mean_reward": float(np.mean(agent.episode_rewards_history)),
            "final_100_mean": float(np.mean(agent.episode_rewards_history[-100:])),
            "episodes": len(agent.episode_rewards_history),
        }
        with open(final_dir / "1m_reinforce_best.meta.json", "w") as f:
            json.dump(summary, f, indent=2)

    (ckpt_dir / "COMPLETED").touch()
    env.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--algorithm", type=str, default="all", choices=["dqn", "ppo", "a2c", "reinforce", "all"])
    args = parser.parse_args()

    algos = ["dqn", "ppo", "a2c", "reinforce"] if args.algorithm == "all" else [args.algorithm]

    print(f"1m step training: {', '.join(algos)}")
    print(f"device: {DEVICE}")

    for algo in algos:
        if algo == "reinforce":
            train_reinforce(resume=args.resume)
        else:
            cls = {"dqn": DQN, "ppo": PPO, "a2c": A2C}[algo]
            train_sb3(algo, cls, resume=args.resume)

    print("all 1m runs complete")


if __name__ == "__main__":
    main()
