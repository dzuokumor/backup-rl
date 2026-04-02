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
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from environment.custom_env import PowerGridEnv

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CHECKPOINT_DIR = PROJECT_ROOT / "models" / "checkpoints" / "ppo_optimized"
FINAL_MODEL_DIR = PROJECT_ROOT / "models" / "pg"
LOG_DIR = PROJECT_ROOT / "logs"
RESULTS_LOG_DIR = PROJECT_ROOT / "results" / "logs"
TOTAL_TIMESTEPS = 1_000_000
CHECKPOINT_STEP_INTERVAL = 25_000
CHECKPOINT_TIME_INTERVAL = 300
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

PPO_PARAMS = {
    "learning_rate": 5e-4,
    "gamma": 0.95,
    "clip_range": 0.3,
    "n_steps": 1024,
    "batch_size": 128,
    "n_epochs": 20,
    "ent_coef": 0.01,
    "gae_lambda": 0.9,
}
NET_ARCH = [128, 128, 128]

ENV_PARAMS = {
    "fault_probability": 0.001,
    "max_disconnected_lines": 10,
    "fault_recovery_steps": 20,
    "survival_bonus": 0.2,
}


def setup_logger():
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("ppo_opt")
    logger.setLevel(logging.DEBUG)
    logger.handlers.clear()
    fh = logging.FileHandler(LOG_DIR / "ppo_optimized.log")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    logger.addHandler(fh)
    sh = logging.StreamHandler()
    sh.setLevel(logging.INFO)
    sh.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(sh)
    return logger


class OptCallback(BaseCallback):
    def __init__(self, logger, verbose=0):
        super().__init__(verbose)
        self.ext_logger = logger
        self.episode_rewards = []
        self.episode_lengths = []
        self.current_episode_reward = 0.0
        self.current_episode_length = 0
        self.last_checkpoint_step = 0
        self.last_checkpoint_time = time.time()
        self._last_print_step = 0
        self.log_path = RESULTS_LOG_DIR / "ppo_optimized.csv"
        RESULTS_LOG_DIR.mkdir(parents=True, exist_ok=True)
        if not self.log_path.exists():
            with open(self.log_path, "w", newline="") as f:
                csv.writer(f).writerow(["timestep", "episode", "episode_reward", "episode_length", "mean_reward_100"])

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

        with open(self.log_path, "a", newline="") as f:
            csv.writer(f).writerow([self.num_timesteps, ep_num, f"{ep_reward:.4f}", ep_length, f"{mean_100:.4f}"])

        if self.num_timesteps - self._last_print_step >= 50000:
            self._last_print_step = self.num_timesteps
            survived = ep_length >= 1000
            self.ext_logger.info(
                f"  step {self.num_timesteps:>7} | ep {ep_num:>5} | "
                f"reward {ep_reward:>8.2f} | len {ep_length:>4} | "
                f"mean100 {mean_100:>8.2f} | {'survived' if survived else ''}"
            )

    def _save_checkpoint(self):
        CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
        self.model.save(str(CHECKPOINT_DIR / "model"))
        meta = {
            "timestep": int(self.num_timesteps),
            "episode_count": len(self.episode_rewards),
            "episode_rewards": self.episode_rewards[-200:],
            "episode_lengths": self.episode_lengths[-200:],
            "mean_reward_100": float(np.mean(self.episode_rewards[-100:])) if self.episode_rewards else 0.0,
        }
        with open(CHECKPOINT_DIR / "meta.json", "w") as f:
            json.dump(meta, f)
        rng_state = {"numpy": np.random.get_state(), "torch": torch.random.get_rng_state()}
        if torch.cuda.is_available():
            rng_state["torch_cuda"] = torch.cuda.get_rng_state_all()
        with open(CHECKPOINT_DIR / "rng_state.pkl", "wb") as f:
            pickle.dump(rng_state, f)
        self.last_checkpoint_step = self.num_timesteps
        self.last_checkpoint_time = time.time()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    logger = setup_logger()
    logger.info("ppo optimized training (reduced faults, auto-recovery, survival bonus)")
    logger.info(f"  ppo params: {PPO_PARAMS}")
    logger.info(f"  env params: {ENV_PARAMS}")
    logger.info(f"  device: {DEVICE}")

    completed = CHECKPOINT_DIR / "COMPLETED"
    if completed.exists():
        logger.info("already completed, skipping")
        return

    env = PowerGridEnv(**ENV_PARAMS)
    remaining_steps = TOTAL_TIMESTEPS

    if args.resume and (CHECKPOINT_DIR / "model.zip").exists():
        logger.info("resuming from checkpoint")
        model = PPO.load(str(CHECKPOINT_DIR / "model"), env=env, device=DEVICE)
        if (CHECKPOINT_DIR / "meta.json").exists():
            with open(CHECKPOINT_DIR / "meta.json") as f:
                meta = json.load(f)
            completed_steps = meta.get("timestep", 0)
            remaining_steps = TOTAL_TIMESTEPS - completed_steps
            logger.info(f"  resumed at step {completed_steps}, {remaining_steps} remaining")
        if (CHECKPOINT_DIR / "rng_state.pkl").exists():
            with open(CHECKPOINT_DIR / "rng_state.pkl", "rb") as f:
                rng_state = pickle.load(f)
            np.random.set_state(rng_state["numpy"])
            torch.random.set_rng_state(rng_state["torch"])
            if torch.cuda.is_available() and "torch_cuda" in rng_state:
                torch.cuda.set_rng_state_all(rng_state["torch_cuda"])
    else:
        model = PPO(
            "MlpPolicy", env,
            **PPO_PARAMS,
            policy_kwargs={"net_arch": NET_ARCH},
            verbose=0,
            device=DEVICE,
        )

    callback = OptCallback(logger)
    start = time.time()
    model.learn(total_timesteps=remaining_steps, callback=callback, reset_num_timesteps=False)
    elapsed = time.time() - start
    logger.info(f"  finished in {elapsed:.1f}s")

    FINAL_MODEL_DIR.mkdir(parents=True, exist_ok=True)
    model.save(str(FINAL_MODEL_DIR / "ppo_optimized_best"))

    if callback.episode_rewards:
        rewards = callback.episode_rewards
        lengths = callback.episode_lengths
        survival_rate = sum(1 for l in lengths if l >= 1000) / len(lengths)
        summary = {
            "mean_reward": float(np.mean(rewards)),
            "final_100_mean": float(np.mean(rewards[-100:])),
            "episodes": len(rewards),
            "survival_rate": float(survival_rate),
            "mean_steps": float(np.mean(lengths)),
        }
        with open(FINAL_MODEL_DIR / "ppo_optimized_best.meta.json", "w") as f:
            json.dump(summary, f, indent=2)
        logger.info(f"  survival rate: {survival_rate:.1%}")
        logger.info(f"  final 100 mean: {np.mean(rewards[-100:]):.2f}")
        logger.info(f"  mean steps: {np.mean(lengths):.1f}")

    (CHECKPOINT_DIR / "COMPLETED").touch()
    logger.info("saved to models/pg/ppo_optimized_best")
    env.close()


if __name__ == "__main__":
    main()
