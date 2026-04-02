import sys
import time
import json
import pickle
import argparse
import logging
import csv
from pathlib import Path

import numpy as np
import torch
from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import BaseCallback

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from environment.custom_env import PowerGridEnv

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CHECKPOINT_DIR = PROJECT_ROOT / "models" / "checkpoints" / "dqn"
FINAL_MODEL_DIR = PROJECT_ROOT / "models" / "dqn"
LOG_DIR = PROJECT_ROOT / "logs"
RESULTS_LOG_DIR = PROJECT_ROOT / "results" / "logs"
RESULTS_TABLE_DIR = PROJECT_ROOT / "results" / "tables"
TOTAL_TIMESTEPS = 100_000
CHECKPOINT_STEP_INTERVAL = 10_000
CHECKPOINT_TIME_INTERVAL = 300

HYPERPARAMS = [
    {
        "run_id": 0,
        "learning_rate": 3e-4,
        "gamma": 0.9,
        "buffer_size": 100000,
        "batch_size": 128,
        "exploration_fraction": 0.5,
        "exploration_final_eps": 0.05,
        "target_update_interval": 1000,
        "net_arch": [200, 200, 200],
    },
    {
        "run_id": 1,
        "learning_rate": 1e-4,
        "gamma": 0.95,
        "buffer_size": 100000,
        "batch_size": 128,
        "exploration_fraction": 0.4,
        "exploration_final_eps": 0.02,
        "target_update_interval": 500,
        "net_arch": [256, 256],
    },
    {
        "run_id": 2,
        "learning_rate": 5e-4,
        "gamma": 0.9,
        "buffer_size": 50000,
        "batch_size": 64,
        "exploration_fraction": 0.5,
        "exploration_final_eps": 0.05,
        "target_update_interval": 1000,
        "net_arch": [128, 128, 128],
    },
    {
        "run_id": 3,
        "learning_rate": 3e-4,
        "gamma": 0.99,
        "buffer_size": 100000,
        "batch_size": 128,
        "exploration_fraction": 0.3,
        "exploration_final_eps": 0.01,
        "target_update_interval": 1000,
        "net_arch": [200, 200, 200],
    },
    {
        "run_id": 4,
        "learning_rate": 1e-3,
        "gamma": 0.9,
        "buffer_size": 50000,
        "batch_size": 64,
        "exploration_fraction": 0.5,
        "exploration_final_eps": 0.1,
        "target_update_interval": 500,
        "net_arch": [128, 128],
    },
    {
        "run_id": 5,
        "learning_rate": 1e-4,
        "gamma": 0.9,
        "buffer_size": 100000,
        "batch_size": 128,
        "exploration_fraction": 0.6,
        "exploration_final_eps": 0.05,
        "target_update_interval": 2000,
        "net_arch": [256, 256, 256],
    },
    {
        "run_id": 6,
        "learning_rate": 5e-4,
        "gamma": 0.95,
        "buffer_size": 50000,
        "batch_size": 64,
        "exploration_fraction": 0.4,
        "exploration_final_eps": 0.02,
        "target_update_interval": 1000,
        "net_arch": [200, 200],
    },
    {
        "run_id": 7,
        "learning_rate": 3e-4,
        "gamma": 0.9,
        "buffer_size": 100000,
        "batch_size": 128,
        "exploration_fraction": 0.5,
        "exploration_final_eps": 0.05,
        "target_update_interval": 500,
        "net_arch": [128, 128],
    },
    {
        "run_id": 8,
        "learning_rate": 1e-3,
        "gamma": 0.95,
        "buffer_size": 50000,
        "batch_size": 64,
        "exploration_fraction": 0.3,
        "exploration_final_eps": 0.01,
        "target_update_interval": 1000,
        "net_arch": [64, 64],
    },
    {
        "run_id": 9,
        "learning_rate": 5e-3,
        "gamma": 0.99,
        "buffer_size": 100000,
        "batch_size": 128,
        "exploration_fraction": 0.3,
        "exploration_final_eps": 0.1,
        "target_update_interval": 1000,
        "net_arch": [256, 256],
    },
]


def setup_logger(run_id):
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger(f"dqn_run_{run_id}")
    logger.setLevel(logging.DEBUG)
    logger.handlers.clear()
    fh = logging.FileHandler(LOG_DIR / f"dqn_run_{run_id}.log", mode="a")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    logger.addHandler(fh)
    sh = logging.StreamHandler()
    sh.setLevel(logging.INFO)
    sh.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(sh)
    return logger


class DQNCheckpointCallback(BaseCallback):
    def __init__(self, run_id, checkpoint_dir, logger, verbose=1):
        super().__init__(verbose)
        self.run_id = run_id
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.ext_logger = logger
        self.last_checkpoint_step = 0
        self.last_checkpoint_time = time.time()
        self.episode_rewards = []
        self.episode_lengths = []
        self.current_episode_reward = 0.0
        self.current_episode_length = 0
        self.losses = []
        self.metrics_path = RESULTS_LOG_DIR / f"dqn_run_{run_id}.csv"
        RESULTS_LOG_DIR.mkdir(parents=True, exist_ok=True)
        self._init_metrics_csv()

    def _init_metrics_csv(self):
        if not self.metrics_path.exists():
            with open(self.metrics_path, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([
                    "timestep", "episode", "episode_reward", "episode_length",
                    "mean_reward_last_10", "loss", "exploration_rate",
                ])

    def _on_step(self):
        self.current_episode_reward += self.locals.get("rewards", [0.0])[0]
        self.current_episode_length += 1

        infos = self.locals.get("infos", [{}])
        dones = self.locals.get("dones", [False])

        if len(dones) > 0 and dones[0]:
            self.episode_rewards.append(self.current_episode_reward)
            self.episode_lengths.append(self.current_episode_length)
            self._write_metric_row()
            self.current_episode_reward = 0.0
            self.current_episode_length = 0

        if hasattr(self.model, "logger") and self.model.logger is not None:
            try:
                loss_val = self.model.logger.name_to_value.get("train/loss")
                if loss_val is not None:
                    self.losses.append(loss_val)
            except Exception:
                pass

        steps_since = self.num_timesteps - self.last_checkpoint_step
        time_since = time.time() - self.last_checkpoint_time
        if steps_since >= CHECKPOINT_STEP_INTERVAL or time_since >= CHECKPOINT_TIME_INTERVAL:
            self._save_checkpoint()

        return True

    def _write_metric_row(self):
        ep_num = len(self.episode_rewards)
        ep_reward = self.episode_rewards[-1]
        ep_length = self.episode_lengths[-1]
        last_10 = self.episode_rewards[-10:]
        mean_r = sum(last_10) / len(last_10)
        loss = self.losses[-1] if self.losses else 0.0
        exploration_rate = 0.0
        if hasattr(self.model, "exploration_rate"):
            exploration_rate = self.model.exploration_rate

        with open(self.metrics_path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                self.num_timesteps, ep_num, f"{ep_reward:.4f}", ep_length,
                f"{mean_r:.4f}", f"{loss:.6f}", f"{exploration_rate:.4f}",
            ])

        steps_since_print = self.num_timesteps - getattr(self, '_last_print_step', 0)
        if steps_since_print >= 10000:
            self._last_print_step = self.num_timesteps
            self.ext_logger.info(
                f"  step {self.num_timesteps} | episode {ep_num} | "
                f"reward {ep_reward:.2f} | mean(10) {mean_r:.2f} | "
                f"eps {exploration_rate:.3f}"
            )

    def _save_checkpoint(self):
        self.ext_logger.debug(f"saving checkpoint at step {self.num_timesteps}")
        cp_path = self.checkpoint_dir / "latest"
        cp_path.mkdir(parents=True, exist_ok=True)

        self.model.save(str(cp_path / "model"))

        try:
            self.model.save_replay_buffer(str(cp_path / "replay_buffer"))
        except Exception as e:
            self.ext_logger.debug(f"could not save replay buffer: {e}")

        meta = {
            "timestep": int(self.num_timesteps),
            "episode_count": len(self.episode_rewards),
            "episode_rewards": [float(r) for r in self.episode_rewards],
            "episode_lengths": [int(l) for l in self.episode_lengths],
            "losses": [float(l) for l in self.losses[-1000:]],
            "run_id": self.run_id,
            "wall_time": time.time(),
        }
        with open(cp_path / "meta.json", "w") as f:
            json.dump(meta, f)

        rng_state = {
            "numpy": np.random.get_state(),
            "torch": torch.random.get_rng_state(),
        }
        if torch.cuda.is_available():
            rng_state["torch_cuda"] = torch.cuda.get_rng_state_all()
        with open(cp_path / "rng_state.pkl", "wb") as f:
            pickle.dump(rng_state, f)

        self.last_checkpoint_step = self.num_timesteps
        self.last_checkpoint_time = time.time()
        self.ext_logger.debug(f"checkpoint saved at step {self.num_timesteps}")

    def get_summary(self):
        if not self.episode_rewards:
            return {"mean_reward": 0.0, "std_reward": 0.0, "episodes": 0, "total_steps": self.num_timesteps}
        return {
            "mean_reward": float(np.mean(self.episode_rewards)),
            "std_reward": float(np.std(self.episode_rewards)),
            "max_reward": float(np.max(self.episode_rewards)),
            "min_reward": float(np.min(self.episode_rewards)),
            "episodes": len(self.episode_rewards),
            "total_steps": self.num_timesteps,
            "mean_length": float(np.mean(self.episode_lengths)),
            "final_10_mean": float(np.mean(self.episode_rewards[-10:])),
        }


def load_checkpoint(run_id, checkpoint_dir, env, hyperparams, logger):
    cp_path = checkpoint_dir / "latest"
    if not (cp_path / "model.zip").exists():
        return None, None, 0

    logger.info(f"  resuming run {run_id} from checkpoint")

    model = DQN.load(
        str(cp_path / "model"),
        env=env,
        device=get_device(),
    )

    try:
        model.load_replay_buffer(str(cp_path / "replay_buffer"))
        logger.debug("replay buffer loaded")
    except Exception as e:
        logger.debug(f"could not load replay buffer: {e}")

    meta = {}
    if (cp_path / "meta.json").exists():
        with open(cp_path / "meta.json") as f:
            meta = json.load(f)

    if (cp_path / "rng_state.pkl").exists():
        with open(cp_path / "rng_state.pkl", "rb") as f:
            rng_state = pickle.load(f)
        np.random.set_state(rng_state["numpy"])
        torch.random.set_rng_state(rng_state["torch"])
        if torch.cuda.is_available() and "torch_cuda" in rng_state:
            torch.cuda.set_rng_state_all(rng_state["torch_cuda"])

    completed_steps = meta.get("timestep", 0)
    logger.info(f"  resumed at step {completed_steps}")
    return model, meta, completed_steps


def get_device():
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def run_single(run_id, params, resume=False):
    logger = setup_logger(run_id)
    logger.info(f"run {run_id} starting")
    logger.info(f"  hyperparams: {params}")
    logger.info(f"  device: {get_device()}")

    run_checkpoint_dir = CHECKPOINT_DIR / str(run_id)
    run_checkpoint_dir.mkdir(parents=True, exist_ok=True)

    completed_marker = run_checkpoint_dir / "COMPLETED"
    if completed_marker.exists():
        logger.info(f"  run {run_id} already completed, skipping")
        try:
            if (run_checkpoint_dir / "latest" / "meta.json").exists():
                with open(run_checkpoint_dir / "latest" / "meta.json") as f:
                    meta = json.load(f)
                return meta.get("summary", {})
        except (json.JSONDecodeError, Exception):
            pass
        return {}

    env = PowerGridEnv()
    device = get_device()

    model = None
    remaining_steps = TOTAL_TIMESTEPS
    meta = {}

    if resume:
        model, meta, completed_steps = load_checkpoint(
            run_id, run_checkpoint_dir, env, params, logger
        )
        if model is not None:
            remaining_steps = max(0, TOTAL_TIMESTEPS - completed_steps)
            if remaining_steps == 0:
                logger.info(f"  run {run_id} already finished all steps")
                completed_marker.touch()
                env.close()
                return {}

    if model is None:
        model = DQN(
            "MlpPolicy",
            env,
            learning_rate=params["learning_rate"],
            gamma=params["gamma"],
            buffer_size=params["buffer_size"],
            batch_size=params["batch_size"],
            exploration_fraction=params["exploration_fraction"],
            exploration_final_eps=params["exploration_final_eps"],
            target_update_interval=params["target_update_interval"],
            policy_kwargs={"net_arch": params["net_arch"]},
            verbose=0,
            device=device,
        )

    callback = DQNCheckpointCallback(
        run_id=run_id,
        checkpoint_dir=run_checkpoint_dir,
        logger=logger,
    )

    if meta:
        callback.episode_rewards = meta.get("episode_rewards", [])
        callback.episode_lengths = meta.get("episode_lengths", [])
        callback.losses = meta.get("losses", [])

    logger.info(f"  training for {remaining_steps} steps")
    start_time = time.time()

    try:
        model.learn(
            total_timesteps=remaining_steps,
            callback=callback,
            log_interval=10,
            reset_num_timesteps=False if meta else True,
        )
    except KeyboardInterrupt:
        logger.info("  interrupted, saving checkpoint")
        callback._save_checkpoint()
        env.close()
        raise

    elapsed = time.time() - start_time
    logger.info(f"  run {run_id} finished in {elapsed:.1f}s")

    FINAL_MODEL_DIR.mkdir(parents=True, exist_ok=True)
    model.save(str(FINAL_MODEL_DIR / f"dqn_run_{run_id}"))
    logger.info(f"  final model saved to {FINAL_MODEL_DIR / f'dqn_run_{run_id}'}")

    callback._save_checkpoint()

    summary = callback.get_summary()
    summary_meta = {
        "timestep": int(callback.num_timesteps),
        "episode_count": len(callback.episode_rewards),
        "episode_rewards": [float(r) for r in callback.episode_rewards],
        "episode_lengths": [int(l) for l in callback.episode_lengths],
        "losses": [float(l) for l in callback.losses[-1000:]],
        "run_id": run_id,
        "wall_time": time.time(),
        "summary": {k: float(v) if isinstance(v, (np.floating, np.integer)) else v for k, v in summary.items()},
        "elapsed_seconds": float(elapsed),
    }
    with open(run_checkpoint_dir / "latest" / "meta.json", "w") as f:
        json.dump(summary_meta, f)

    completed_marker.touch()
    logger.info(f"  run {run_id} marked complete")

    env.close()
    return summary


def save_sweep_results(all_results):
    RESULTS_TABLE_DIR.mkdir(parents=True, exist_ok=True)
    sweep_path = RESULTS_TABLE_DIR / "dqn_sweep.csv"

    fieldnames = [
        "run_id", "learning_rate", "gamma", "buffer_size", "batch_size",
        "exploration_fraction", "exploration_final_eps", "target_update_interval",
        "net_arch", "mean_reward", "std_reward", "max_reward", "min_reward",
        "episodes", "total_steps", "mean_length", "final_10_mean",
    ]

    with open(sweep_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for run_id, summary in all_results.items():
            params = HYPERPARAMS[run_id]
            row = {
                "run_id": run_id,
                "learning_rate": params["learning_rate"],
                "gamma": params["gamma"],
                "buffer_size": params["buffer_size"],
                "batch_size": params["batch_size"],
                "exploration_fraction": params["exploration_fraction"],
                "exploration_final_eps": params["exploration_final_eps"],
                "target_update_interval": params["target_update_interval"],
                "net_arch": str(params["net_arch"]),
            }
            row.update(summary)
            writer.writerow(row)

    print(f"sweep results saved to {sweep_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    print(f"dqn hyperparameter sweep: {len(HYPERPARAMS)} runs, {TOTAL_TIMESTEPS} steps each")
    print(f"device: {get_device()}")
    if args.resume:
        print("resume mode enabled")

    all_results = {}
    for params in HYPERPARAMS:
        run_id = params["run_id"]
        print(f"\nrun {run_id}/{len(HYPERPARAMS)-1} starting")
        try:
            summary = run_single(run_id, params, resume=args.resume)
            all_results[run_id] = summary
            if summary:
                print(
                    f"run {run_id} done: mean reward {summary.get('mean_reward', 0):.2f}, "
                    f"episodes {summary.get('episodes', 0)}"
                )
        except KeyboardInterrupt:
            print(f"\ninterrupted at run {run_id}, saving partial sweep results")
            save_sweep_results(all_results)
            sys.exit(1)
        except Exception as e:
            print(f"run {run_id} failed: {e}")
            logging.getLogger(f"dqn_run_{run_id}").exception("run failed")
            all_results[run_id] = {}

    save_sweep_results(all_results)
    print("\nall runs complete")

    if all_results:
        valid = {k: v for k, v in all_results.items() if v.get("mean_reward") is not None}
        if valid:
            best_id = max(valid, key=lambda k: valid[k]["mean_reward"])
            print(
                f"best run: {best_id} with mean reward {valid[best_id]['mean_reward']:.2f}"
            )


if __name__ == "__main__":
    main()
