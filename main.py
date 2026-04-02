import argparse
import csv
import json
import logging
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

LOG_DIR = ROOT / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)

from environment.custom_env import PowerGridEnv, get_action_name


def setup_logging(agent_type):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = LOG_DIR / f"{agent_type}_{timestamp}.log"

    logger = logging.getLogger("main")
    logger.setLevel(logging.DEBUG)
    logger.handlers.clear()

    fh = logging.FileHandler(log_file)
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    logger.addHandler(fh)

    sh = logging.StreamHandler()
    sh.setLevel(logging.INFO)
    sh.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(sh)

    return logger, log_file, timestamp


def setup_step_csv(agent_type, episode_num, timestamp):
    csv_dir = LOG_DIR / agent_type
    csv_dir.mkdir(parents=True, exist_ok=True)
    csv_path = csv_dir / f"ep{episode_num}_{timestamp}.csv"
    f = open(csv_path, "w", newline="")
    writer = csv.writer(f)
    writer.writerow([
        "step", "action", "action_name", "action_category",
        "reward", "cumulative_reward", "load_served_pct",
        "total_demand", "total_generation",
        "num_overloaded", "num_disconnected",
        "time_of_day", "cascade_events",
    ])
    return f, writer, csv_path


def find_best_model():
    model_dirs = [ROOT / "models" / "dqn", ROOT / "models" / "pg"]
    best_path = None
    best_reward = -float("inf")

    for d in model_dirs:
        if not d.exists():
            continue
        for f in d.glob("*_final.zip"):
            meta = f.with_suffix(".meta.json")
            if meta.exists():
                with open(meta) as fh:
                    data = json.load(fh)
                if data.get("mean_reward", -float("inf")) > best_reward:
                    best_reward = data["mean_reward"]
                    best_path = f

    if best_path is None:
        for d in model_dirs:
            if not d.exists():
                continue
            files = sorted(d.glob("*_final.zip"))
            if files:
                best_path = files[0]
                break

    return best_path


def load_model(model_path):
    from stable_baselines3 import DQN, PPO, A2C

    path_str = str(model_path).lower()
    env = PowerGridEnv()

    if "dqn" in path_str:
        return DQN.load(model_path, env=env), env
    elif "ppo" in path_str:
        return PPO.load(model_path, env=env), env
    elif "a2c" in path_str:
        return A2C.load(model_path, env=env), env
    else:
        for cls in [DQN, PPO, A2C]:
            try:
                return cls.load(model_path, env=env), env
            except Exception:
                continue
        raise ValueError(f"could not load model from {model_path}")


def run_episode(env, model, episode_num, agent_type, timestamp, logger):
    obs, info = env.reset()
    total_reward = 0.0
    step = 0

    csv_file, csv_writer, csv_path = setup_step_csv(agent_type, episode_num, timestamp)
    logger.info(f"\nepisode {episode_num}")
    logger.debug(f"step log: {csv_path}")

    while True:
        if model is None:
            if step < 50:
                action = 0
            elif step < 150:
                action = env.action_space.sample() if env._rng.random() < 0.3 else 0
            else:
                action = env.action_space.sample()
        else:
            action, _ = model.predict(obs, deterministic=True)
            action = int(action)

        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        step += 1

        action_name, action_cat = get_action_name(action)
        cascades = info.get("cascade_events", [])

        csv_writer.writerow([
            step, action, action_name, action_cat,
            f"{reward:.4f}", f"{total_reward:.4f}", info["load_served_pct"],
            info["total_demand"], info["total_generation"],
            info["num_overloaded"], info["num_disconnected"],
            info["time_of_day"], "|".join(cascades) if cascades else "",
        ])

        logger.info(
            f"  step {step:4d} | {action_name:40s} | "
            f"served {info['load_served_pct']:5.1f}% | "
            f"overloaded {info['num_overloaded']} | "
            f"disconnected {info['num_disconnected']} | "
            f"reward {reward:+.3f} | total {total_reward:+.2f}"
        )

        if cascades:
            for c in cascades:
                logger.info(f"           cascade: {c}")
                logger.debug(f"cascade detail: {c}")

        logger.debug(
            f"step {step} | obs_min={obs.min():.3f} obs_max={obs.max():.3f} | "
            f"demand={info['total_demand']} gen={info['total_generation']} | "
            f"overloaded_lines={info.get('overloaded_lines', [])} | "
            f"disconnected_lines={info.get('disconnected_lines', [])}"
        )

        if terminated or truncated:
            break

    csv_file.close()

    status = "blackout" if terminated else "survived"
    logger.info(f"  finished: {status} | steps: {step} | total reward: {total_reward:.2f}")
    logger.debug(f"episode {episode_num} complete: {status}, {step} steps, reward {total_reward:.4f}")

    return total_reward, step, terminated


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument("--random", action="store_true")
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--no-render", action="store_true")
    args = parser.parse_args()

    render_mode = None if args.no_render else "human"

    if args.random:
        agent_type = "random"
        model = None
    elif args.model:
        model_path = Path(args.model)
        if not model_path.exists():
            print(f"model not found: {model_path}")
            sys.exit(1)
        agent_type = Path(args.model).stem
        model, _ = load_model(model_path)
    else:
        model_path = find_best_model()
        if model_path is None:
            agent_type = "random"
            model = None
        else:
            agent_type = model_path.stem
            model, _ = load_model(model_path)

    logger, log_file, timestamp = setup_logging(agent_type)
    logger.info(f"agent: {agent_type}")
    logger.info(f"log file: {log_file}")
    logger.debug(f"render_mode: {render_mode}, episodes: {args.episodes}")

    env = PowerGridEnv(render_mode=render_mode)

    rewards = []
    for ep in range(1, args.episodes + 1):
        total_reward, steps, terminated = run_episode(
            env, model, ep, agent_type, timestamp, logger
        )
        rewards.append(total_reward)

    if len(rewards) > 1:
        logger.info(f"\nsummary over {len(rewards)} episodes:")
        logger.info(f"  mean reward: {np.mean(rewards):.2f}")
        logger.info(f"  std reward: {np.std(rewards):.2f}")
        logger.info(f"  min: {np.min(rewards):.2f} max: {np.max(rewards):.2f}")

    summary = {
        "agent_type": agent_type,
        "episodes": len(rewards),
        "mean_reward": float(np.mean(rewards)),
        "std_reward": float(np.std(rewards)),
        "min_reward": float(np.min(rewards)),
        "max_reward": float(np.max(rewards)),
        "timestamp": timestamp,
    }
    summary_path = LOG_DIR / f"{agent_type}_{timestamp}_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    logger.debug(f"summary saved to {summary_path}")

    env.close()


if __name__ == "__main__":
    main()
