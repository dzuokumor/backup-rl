import os
import sys
import csv
import json
import time
import random
import argparse
import logging
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical

from stable_baselines3 import PPO, A2C
from stable_baselines3.common.callbacks import BaseCallback

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from environment.custom_env import PowerGridEnv

PROJECT_ROOT = Path(__file__).resolve().parent.parent
TOTAL_TIMESTEPS = 100_000
CHECKPOINT_INTERVAL_STEPS = 10_000
CHECKPOINT_INTERVAL_SECONDS = 300
NUM_RUNS = 10

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[
        logging.FileHandler(PROJECT_ROOT / "logs" / "pg_training.log"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)


PPO_HYPERPARAM_GRID = {
    "learning_rate": [1e-4, 3e-4, 5e-4],
    "gamma": [0.9, 0.95, 0.99],
    "clip_range": [0.1, 0.2, 0.3],
    "n_steps": [512, 1024, 2048],
    "batch_size": [64, 128],
    "n_epochs": [10, 20, 40],
    "ent_coef": [0.01, 0.05, 0.1],
    "gae_lambda": [0.9, 0.95],
    "net_arch": [[200, 200, 200], [128, 128, 128], [256, 256]],
}

A2C_HYPERPARAM_GRID = {
    "learning_rate": [1e-4, 3e-4, 7e-4],
    "gamma": [0.9, 0.95, 0.99],
    "n_steps": [16, 32, 64],
    "ent_coef": [0.01, 0.05, 0.1],
    "vf_coef": [0.25, 0.5, 0.75],
    "max_grad_norm": [0.5, 1.0, 10.0],
    "net_arch": [[128, 128], [200, 200, 200]],
}

REINFORCE_HYPERPARAM_GRID = {
    "learning_rate": [1e-4, 3e-4, 5e-4, 1e-3],
    "gamma": [0.9, 0.95, 0.99],
    "net_arch": [[128, 128], [200, 200, 200], [256, 256]],
    "max_grad_norm": [1.0, 5.0, 10.0],
}


def sample_hyperparams(grid, num_samples, seed=42):
    rng = random.Random(seed)
    configs = []
    for i in range(num_samples):
        config = {}
        for key, values in grid.items():
            config[key] = rng.choice(values)
        configs.append(config)
    return configs


def ensure_dirs():
    dirs = [
        PROJECT_ROOT / "models" / "pg",
        PROJECT_ROOT / "models" / "checkpoints",
        PROJECT_ROOT / "results" / "logs",
        PROJECT_ROOT / "results" / "tables",
        PROJECT_ROOT / "logs",
    ]
    for d in dirs:
        d.mkdir(parents=True, exist_ok=True)


def checkpoint_dir(algorithm, run_id):
    d = PROJECT_ROOT / "models" / "checkpoints" / algorithm / str(run_id)
    d.mkdir(parents=True, exist_ok=True)
    return d


def is_completed(algorithm, run_id):
    return (checkpoint_dir(algorithm, run_id) / "COMPLETED").exists()


def mark_completed(algorithm, run_id):
    (checkpoint_dir(algorithm, run_id) / "COMPLETED").touch()


class PolicyNetwork(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden_layers):
        super().__init__()
        layers = []
        prev_dim = obs_dim
        for h in hidden_layers:
            layers.append(nn.Linear(prev_dim, h))
            layers.append(nn.ReLU())
            prev_dim = h
        layers.append(nn.Linear(prev_dim, act_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        logits = self.net(x)
        return torch.softmax(logits, dim=-1)

    def get_action(self, obs):
        obs_t = torch.FloatTensor(obs).unsqueeze(0).to(next(self.parameters()).device)
        probs = self.forward(obs_t)
        dist = Categorical(probs)
        action = dist.sample()
        return action.item(), dist.log_prob(action)


class ReinforceAgent:
    def __init__(self, obs_dim, act_dim, lr, gamma, net_arch, max_grad_norm):
        self.gamma = gamma
        self.max_grad_norm = max_grad_norm
        self.policy = PolicyNetwork(obs_dim, act_dim, net_arch).to(DEVICE)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)
        self.log_probs = []
        self.rewards = []
        self.total_steps = 0
        self.episode_count = 0
        self.episode_rewards_history = []

    def select_action(self, obs):
        action, log_prob = self.policy.get_action(obs)
        self.log_probs.append(log_prob)
        return action

    def store_reward(self, reward):
        self.rewards.append(reward)

    def update(self):
        returns = []
        g = 0
        for r in reversed(self.rewards):
            g = r + self.gamma * g
            returns.insert(0, g)

        returns = torch.tensor(returns, dtype=torch.float32, device=DEVICE)
        if len(returns) > 1:
            returns = (returns - returns.mean()) / (returns.std() + 1e-8)

        loss = 0
        for log_prob, ret in zip(self.log_probs, returns):
            loss -= log_prob * ret

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
        self.optimizer.step()

        self.log_probs = []
        self.rewards = []

    def save(self, path):
        state = {
            "policy_state_dict": self.policy.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "total_steps": self.total_steps,
            "episode_count": self.episode_count,
            "episode_rewards_history": self.episode_rewards_history,
            "gamma": self.gamma,
            "max_grad_norm": self.max_grad_norm,
        }
        torch.save(state, path)

    def load(self, path):
        state = torch.load(path, map_location=DEVICE, weights_only=False)
        self.policy.load_state_dict(state["policy_state_dict"])
        self.optimizer.load_state_dict(state["optimizer_state_dict"])
        self.total_steps = state["total_steps"]
        self.episode_count = state["episode_count"]
        self.episode_rewards_history = state["episode_rewards_history"]


class CheckpointCallback(BaseCallback):
    def __init__(self, algorithm, run_id, log_path, verbose=1):
        super().__init__(verbose)
        self.algorithm = algorithm
        self.run_id = run_id
        self.log_path = Path(log_path)
        self.ckpt_dir = checkpoint_dir(algorithm, run_id)
        self.last_save_time = time.time()
        self.episode_rewards = []
        self.current_episode_reward = 0.0
        self.episode_count = 0
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_csv()

    def _init_csv(self):
        if not self.log_path.exists():
            with open(self.log_path, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["timestep", "episode", "episode_reward", "mean_reward_100"])

    def _on_step(self):
        self.current_episode_reward += self.locals.get("rewards", [0])[0]

        dones = self.locals.get("dones", [False])
        if dones[0]:
            self.episode_count += 1
            self.episode_rewards.append(self.current_episode_reward)
            mean_100 = np.mean(self.episode_rewards[-100:])

            with open(self.log_path, "a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([
                    self.num_timesteps,
                    self.episode_count,
                    round(self.current_episode_reward, 4),
                    round(mean_100, 4),
                ])

            steps_since_print = self.num_timesteps - getattr(self, '_last_print_step', 0)
            if steps_since_print >= 10000:
                self._last_print_step = self.num_timesteps
                logger.info(
                    f"{self.algorithm} run {self.run_id} | "
                    f"ep {self.episode_count} | "
                    f"steps {self.num_timesteps} | "
                    f"reward {self.current_episode_reward:.2f} | "
                    f"mean100 {mean_100:.2f}"
                )

            self.current_episode_reward = 0.0

        now = time.time()
        should_save = (
            self.num_timesteps % CHECKPOINT_INTERVAL_STEPS == 0
            or (now - self.last_save_time) >= CHECKPOINT_INTERVAL_SECONDS
        )
        if should_save:
            self._save_checkpoint()
            self.last_save_time = now

        return True

    def _save_checkpoint(self):
        model_path = self.ckpt_dir / "model.zip"
        self.model.save(str(model_path))
        meta = {
            "timesteps": int(self.num_timesteps),
            "episode_count": int(self.episode_count),
            "episode_rewards": [float(r) for r in self.episode_rewards],
        }
        with open(self.ckpt_dir / "meta.json", "w") as f:
            json.dump(meta, f)
        logger.info(f"checkpoint saved: {self.algorithm} run {self.run_id} at step {self.num_timesteps}")


def train_sb3_algo(algo_class, algo_name, run_id, hyperparams, resume=False):
    ckpt = checkpoint_dir(algo_name, run_id)
    log_path = PROJECT_ROOT / "results" / "logs" / f"{algo_name}_run_{run_id}.csv"

    env = PowerGridEnv()

    policy_kwargs = {}
    model_params = {}

    for k, v in hyperparams.items():
        if k == "net_arch":
            policy_kwargs["net_arch"] = v
        else:
            model_params[k] = v

    if policy_kwargs:
        model_params["policy_kwargs"] = policy_kwargs

    remaining_timesteps = TOTAL_TIMESTEPS
    callback = CheckpointCallback(algo_name, run_id, log_path)

    if resume and (ckpt / "model.zip").exists():
        logger.info(f"resuming {algo_name} run {run_id} from checkpoint")
        model = algo_class.load(
            str(ckpt / "model.zip"),
            env=env,
            device=DEVICE,
        )
        if (ckpt / "meta.json").exists():
            with open(ckpt / "meta.json") as f:
                meta = json.load(f)
            completed_steps = meta.get("timesteps", 0)
            remaining_timesteps = max(0, TOTAL_TIMESTEPS - completed_steps)
            callback.episode_count = meta.get("episode_count", 0)
            callback.episode_rewards = meta.get("episode_rewards", [])
            logger.info(f"resuming from step {completed_steps}, {remaining_timesteps} steps remaining")
    else:
        model = algo_class(
            "MlpPolicy",
            env,
            verbose=0,
            device=DEVICE,
            **model_params,
        )

    if remaining_timesteps <= 0:
        logger.info(f"{algo_name} run {run_id} already completed all timesteps")
        env.close()
        return None

    logger.info(f"starting {algo_name} run {run_id} for {remaining_timesteps} steps with params: {hyperparams}")

    model.learn(total_timesteps=remaining_timesteps, callback=callback)

    final_path = PROJECT_ROOT / "models" / "pg" / f"{algo_name}_run_{run_id}"
    model.save(str(final_path))
    mark_completed(algo_name, run_id)

    mean_reward = np.mean(callback.episode_rewards[-100:]) if callback.episode_rewards else 0.0
    logger.info(f"finished {algo_name} run {run_id} | episodes: {callback.episode_count} | mean100: {mean_reward:.2f}")

    env.close()

    return {
        "run_id": run_id,
        "episodes": callback.episode_count,
        "mean_reward_100": round(float(mean_reward), 4),
        "total_episodes": len(callback.episode_rewards),
        **hyperparams,
    }


def train_reinforce(run_id, hyperparams, resume=False):
    ckpt = checkpoint_dir("reinforce", run_id)
    log_path = PROJECT_ROOT / "results" / "logs" / f"reinforce_run_{run_id}.csv"

    env = PowerGridEnv()
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.n

    agent = ReinforceAgent(
        obs_dim=obs_dim,
        act_dim=act_dim,
        lr=hyperparams["learning_rate"],
        gamma=hyperparams["gamma"],
        net_arch=hyperparams["net_arch"],
        max_grad_norm=hyperparams["max_grad_norm"],
    )

    start_step = 0

    if resume and (ckpt / "reinforce.pt").exists():
        logger.info(f"resuming reinforce run {run_id} from checkpoint")
        agent.load(str(ckpt / "reinforce.pt"))
        start_step = agent.total_steps
        logger.info(f"resuming from step {start_step}")

    if not log_path.exists():
        with open(log_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["timestep", "episode", "episode_reward", "mean_reward_100"])

    logger.info(f"starting reinforce run {run_id} with params: {hyperparams}")

    total_steps = start_step
    last_save_time = time.time()

    while total_steps < TOTAL_TIMESTEPS:
        obs, _ = env.reset()
        episode_reward = 0.0
        done = False

        while not done and total_steps < TOTAL_TIMESTEPS:
            action = agent.select_action(obs)
            obs, reward, terminated, truncated, _ = env.step(action)
            agent.store_reward(reward)
            episode_reward += reward
            total_steps += 1
            done = terminated or truncated

        agent.update()
        agent.total_steps = total_steps
        agent.episode_count += 1
        agent.episode_rewards_history.append(episode_reward)

        mean_100 = np.mean(agent.episode_rewards_history[-100:])

        with open(log_path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                total_steps,
                agent.episode_count,
                round(episode_reward, 4),
                round(float(mean_100), 4),
            ])

        steps_since_print = total_steps - getattr(agent, '_last_print_step', 0)
        if steps_since_print >= 10000:
            agent._last_print_step = total_steps
            logger.info(
                f"reinforce run {run_id} | "
                f"ep {agent.episode_count} | "
                f"steps {total_steps} | "
                f"reward {episode_reward:.2f} | "
                f"mean100 {mean_100:.2f}"
            )

        now = time.time()
        should_save = (
            total_steps % CHECKPOINT_INTERVAL_STEPS < 1000
            or (now - last_save_time) >= CHECKPOINT_INTERVAL_SECONDS
        )
        if should_save:
            agent.save(str(ckpt / "reinforce.pt"))
            last_save_time = now
            logger.info(f"checkpoint saved: reinforce run {run_id} at step {total_steps}")

    final_path = PROJECT_ROOT / "models" / "pg" / f"reinforce_run_{run_id}.pt"
    agent.save(str(final_path))
    mark_completed("reinforce", run_id)

    mean_reward = np.mean(agent.episode_rewards_history[-100:]) if agent.episode_rewards_history else 0.0
    logger.info(
        f"finished reinforce run {run_id} | "
        f"episodes: {agent.episode_count} | "
        f"mean100: {mean_reward:.2f}"
    )

    env.close()

    result = {
        "run_id": run_id,
        "episodes": agent.episode_count,
        "mean_reward_100": round(float(mean_reward), 4),
        "total_episodes": len(agent.episode_rewards_history),
    }
    for k, v in hyperparams.items():
        result[k] = str(v) if isinstance(v, list) else v
    return result


def save_sweep_results(algo_name, results):
    if not results:
        return
    table_path = PROJECT_ROOT / "results" / "tables" / f"{algo_name}_sweep.csv"
    fieldnames = list(results[0].keys())
    with open(table_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in results:
            writer.writerow(row)
    logger.info(f"sweep results saved to {table_path}")


def run_ppo_sweep(resume=False):
    configs = sample_hyperparams(PPO_HYPERPARAM_GRID, NUM_RUNS, seed=100)
    results = []
    for i, params in enumerate(configs):
        run_id = i
        if is_completed("ppo", run_id) and resume:
            logger.info(f"ppo run {run_id} already completed, skipping")
            continue
        result = train_sb3_algo(PPO, "ppo", run_id, params, resume=resume)
        if result:
            results.append(result)
    save_sweep_results("ppo", results)
    return results


def run_a2c_sweep(resume=False):
    configs = sample_hyperparams(A2C_HYPERPARAM_GRID, NUM_RUNS, seed=200)
    results = []
    for i, params in enumerate(configs):
        run_id = i
        if is_completed("a2c", run_id) and resume:
            logger.info(f"a2c run {run_id} already completed, skipping")
            continue
        result = train_sb3_algo(A2C, "a2c", run_id, params, resume=resume)
        if result:
            results.append(result)
    save_sweep_results("a2c", results)
    return results


def run_reinforce_sweep(resume=False):
    configs = sample_hyperparams(REINFORCE_HYPERPARAM_GRID, NUM_RUNS, seed=300)
    results = []
    for i, params in enumerate(configs):
        run_id = i
        if is_completed("reinforce", run_id) and resume:
            logger.info(f"reinforce run {run_id} already completed, skipping")
            continue
        result = train_reinforce(run_id, params, resume=resume)
        if result:
            results.append(result)
    save_sweep_results("reinforce", results)
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--algorithm",
        choices=["reinforce", "ppo", "a2c", "all"],
        default="all",
    )
    args = parser.parse_args()

    ensure_dirs()

    logger.info(f"device: {DEVICE}")
    logger.info(f"algorithm: {args.algorithm}")
    logger.info(f"resume: {args.resume}")
    logger.info(f"total timesteps per run: {TOTAL_TIMESTEPS}")
    logger.info(f"runs per algorithm: {NUM_RUNS}")

    if args.algorithm in ("reinforce", "all"):
        logger.info("starting reinforce sweep")
        run_reinforce_sweep(resume=args.resume)

    if args.algorithm in ("ppo", "all"):
        logger.info("starting ppo sweep")
        run_ppo_sweep(resume=args.resume)

    if args.algorithm in ("a2c", "all"):
        logger.info("starting a2c sweep")
        run_a2c_sweep(resume=args.resume)

    logger.info("all training complete")


if __name__ == "__main__":
    main()
