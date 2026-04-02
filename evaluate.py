import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from environment.custom_env import PowerGridEnv, get_action_name


def evaluate_agent(model, env, num_episodes=50, seed_offset=5000):
    results = []

    for ep in range(num_episodes):
        obs, info = env.reset(seed=seed_offset + ep)
        total_reward = 0.0
        steps = 0
        total_served = 0.0
        total_cascades = 0
        min_served = 100.0

        while True:
            if model is None:
                action = env.action_space.sample()
            elif model == "do_nothing":
                action = 0
            else:
                action, _ = model.predict(obs, deterministic=True)
                action = int(action)

            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            steps += 1
            total_served += info["load_served_pct"]
            min_served = min(min_served, info["load_served_pct"])
            total_cascades += len([e for e in info.get("cascade_events", []) if "auto-disconnect" in e])

            if terminated or truncated:
                break

        survived = not terminated
        results.append({
            "episode": ep,
            "seed": seed_offset + ep,
            "total_reward": total_reward,
            "steps": steps,
            "survived": survived,
            "mean_served_pct": total_served / steps,
            "min_served_pct": min_served,
            "cascades": total_cascades,
        })

    return results


def print_summary(name, results):
    rewards = [r["total_reward"] for r in results]
    steps = [r["steps"] for r in results]
    survived = [r["survived"] for r in results]
    served = [r["mean_served_pct"] for r in results]
    cascades = [r["cascades"] for r in results]

    print(f"\n{name}")
    print(f"  episodes:       {len(results)}")
    print(f"  survival rate:  {sum(survived) / len(survived):.1%}")
    print(f"  mean reward:    {np.mean(rewards):.2f} +/- {np.std(rewards):.2f}")
    print(f"  median reward:  {np.median(rewards):.2f}")
    print(f"  mean steps:     {np.mean(steps):.1f}")
    print(f"  mean served:    {np.mean(served):.1f}%")
    print(f"  mean cascades:  {np.mean(cascades):.2f}")
    print(f"  best episode:   reward={max(rewards):.2f} steps={steps[rewards.index(max(rewards))]}")
    print(f"  worst episode:  reward={min(rewards):.2f} steps={steps[rewards.index(min(rewards))]}")


def save_results(name, results, output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)

    csv_path = output_dir / f"{name}_eval.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)

    rewards = [r["total_reward"] for r in results]
    survived = [r["survived"] for r in results]
    served = [r["mean_served_pct"] for r in results]
    steps_list = [r["steps"] for r in results]
    cascades = [r["cascades"] for r in results]

    summary = {
        "agent": name,
        "episodes": len(results),
        "survival_rate": sum(survived) / len(survived),
        "mean_reward": float(np.mean(rewards)),
        "std_reward": float(np.std(rewards)),
        "median_reward": float(np.median(rewards)),
        "mean_steps": float(np.mean(steps_list)),
        "mean_served_pct": float(np.mean(served)),
        "mean_cascades": float(np.mean(cascades)),
    }
    json_path = output_dir / f"{name}_eval_summary.json"
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)

    return summary


def load_model(model_path):
    from stable_baselines3 import DQN, PPO, A2C

    path_str = str(model_path).lower()
    if "dqn" in path_str:
        return DQN.load(model_path), "dqn"
    elif "ppo" in path_str:
        return PPO.load(model_path), "ppo"
    elif "a2c" in path_str:
        return A2C.load(model_path), "a2c"
    else:
        for name, cls in [("dqn", DQN), ("ppo", PPO), ("a2c", A2C)]:
            try:
                return cls.load(model_path), name
            except Exception:
                continue
    raise ValueError(f"could not load {model_path}")


def find_best_models():
    models = {}
    for algo in ["dqn", "ppo", "a2c", "reinforce"]:
        sweep_path = ROOT / "results" / "tables" / f"{algo}_sweep.csv"
        if not sweep_path.exists():
            continue

        import pandas as pd
        df = pd.read_csv(sweep_path)
        if "mean_reward" not in df.columns:
            continue

        best_idx = df["mean_reward"].idxmax()
        run_id = int(df.loc[best_idx, "run_id"]) if "run_id" in df.columns else best_idx

        if algo == "dqn":
            model_dir = ROOT / "models" / "dqn"
        else:
            model_dir = ROOT / "models" / "pg"

        for pattern in [f"*run_{run_id}_final.zip", f"{algo}_run_{run_id}*.zip", f"run_{run_id}*.zip"]:
            matches = list(model_dir.glob(pattern))
            if matches:
                models[algo] = matches[0]
                break

    return models


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument("--episodes", type=int, default=50)
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--baselines", action="store_true")
    args = parser.parse_args()

    output_dir = ROOT / "results" / "evaluation"
    env = PowerGridEnv()
    all_summaries = []

    if args.baselines or args.all:
        print("evaluating random agent...")
        random_results = evaluate_agent(None, env, args.episodes)
        print_summary("random", random_results)
        all_summaries.append(save_results("random", random_results, output_dir))

        print("evaluating do-nothing agent...")
        donothing_results = evaluate_agent("do_nothing", env, args.episodes)
        print_summary("do_nothing", donothing_results)
        all_summaries.append(save_results("do_nothing", donothing_results, output_dir))

    if args.model:
        model_path = Path(args.model)
        model, algo = load_model(model_path)
        name = f"{algo}_{model_path.stem}"
        print(f"evaluating {name}...")
        results = evaluate_agent(model, env, args.episodes)
        print_summary(name, results)
        all_summaries.append(save_results(name, results, output_dir))

    if args.all:
        best_models = find_best_models()
        for algo, path in best_models.items():
            try:
                model, _ = load_model(path)
                name = f"{algo}_best"
                print(f"evaluating {name} ({path.name})...")
                results = evaluate_agent(model, env, args.episodes)
                print_summary(name, results)
                all_summaries.append(save_results(name, results, output_dir))
            except Exception as e:
                print(f"  failed to evaluate {algo}: {e}")

    if all_summaries:
        print("\n\ncomparison table:")
        print(f"{'agent':<25} {'survival':>10} {'mean_reward':>12} {'mean_steps':>12} {'served%':>10} {'cascades':>10}")
        for s in all_summaries:
            print(f"{s['agent']:<25} {s['survival_rate']:>9.1%} {s['mean_reward']:>12.2f} {s['mean_steps']:>12.1f} {s['mean_served_pct']:>9.1f}% {s['mean_cascades']:>10.2f}")

    env.close()


if __name__ == "__main__":
    main()
