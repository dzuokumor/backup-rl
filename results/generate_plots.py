import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

PLOTS_DIR = ROOT / "results" / "plots"
LOGS_DIR = ROOT / "results" / "logs"
TABLES_DIR = ROOT / "results" / "tables"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "figure.facecolor": "#f8f8f8",
    "axes.facecolor": "#ffffff",
    "axes.grid": True,
    "grid.alpha": 0.3,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 11,
    "legend.fontsize": 9,
    "figure.dpi": 150,
    "savefig.dpi": 150,
    "savefig.bbox": "tight",
})

ALGO_COLORS = {
    "dqn": "#2196F3",
    "ppo": "#4CAF50",
    "a2c": "#FF9800",
    "reinforce": "#9C27B0",
}

ALGO_LABELS = {
    "dqn": "DQN",
    "ppo": "PPO",
    "a2c": "A2C",
    "reinforce": "REINFORCE",
}


def load_training_logs(algo):
    logs = []
    for f in sorted(LOGS_DIR.glob(f"{algo}_run_*.csv")):
        try:
            df = pd.read_csv(f)
            run_id = f.stem.split("_")[-1]
            df["run_id"] = int(run_id)
            logs.append(df)
        except Exception:
            continue
    if logs:
        return pd.concat(logs, ignore_index=True)
    return None


def load_sweep_table(algo):
    path = TABLES_DIR / f"{algo}_sweep.csv"
    if path.exists():
        return pd.read_csv(path)
    return None


def smooth(values, window=20):
    if len(values) < window:
        return values
    return pd.Series(values).rolling(window, min_periods=1).mean().values


def plot_reward_curves():
    fig, axes = plt.subplots(2, 2, figsize=(14, 10), sharex=True)
    fig.suptitle("cumulative reward curves", fontsize=15, y=0.98)

    for idx, algo in enumerate(["dqn", "ppo", "a2c", "reinforce"]):
        ax = axes[idx // 2][idx % 2]
        df = load_training_logs(algo)

        if df is None:
            ax.set_title(f"{ALGO_LABELS[algo]} (no data)")
            continue

        for run_id in sorted(df["run_id"].unique()):
            run_data = df[df["run_id"] == run_id]
            if "episode_reward" in run_data.columns:
                rewards = run_data["episode_reward"].values
                smoothed = smooth(rewards)
                ax.plot(smoothed, alpha=0.5, color=ALGO_COLORS[algo], linewidth=0.8)

        ax.set_title(ALGO_LABELS[algo])
        ax.set_ylabel("episode reward")

    axes[1][0].set_xlabel("episode")
    axes[1][1].set_xlabel("episode")
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "reward_curves.png")
    plt.close()
    print("saved reward_curves.png")


def plot_dqn_loss():
    df = load_training_logs("dqn")
    if df is None or "loss" not in df.columns:
        print("no dqn loss data available")
        return

    fig, ax = plt.subplots(figsize=(10, 5))
    for run_id in sorted(df["run_id"].unique()):
        run_data = df[df["run_id"] == run_id]
        losses = run_data["loss"].dropna().values
        if len(losses) > 0:
            ax.plot(smooth(losses, 50), alpha=0.6, linewidth=0.8, label=f"run {run_id}")

    ax.set_title("DQN loss curves")
    ax.set_xlabel("training step")
    ax.set_ylabel("loss")
    ax.legend(ncol=2)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "dqn_loss.png")
    plt.close()
    print("saved dqn_loss.png")


def plot_entropy_curves():
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle("policy entropy over training", fontsize=14)

    for idx, algo in enumerate(["ppo", "a2c", "reinforce"]):
        ax = axes[idx]
        df = load_training_logs(algo)

        if df is None or "entropy" not in df.columns:
            ax.set_title(f"{ALGO_LABELS[algo]} (no data)")
            continue

        for run_id in sorted(df["run_id"].unique()):
            run_data = df[df["run_id"] == run_id]
            entropy = run_data["entropy"].dropna().values
            if len(entropy) > 0:
                ax.plot(smooth(entropy, 50), alpha=0.6, linewidth=0.8)

        ax.set_title(ALGO_LABELS[algo])
        ax.set_xlabel("training step")
        ax.set_ylabel("entropy")

    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "entropy_curves.png")
    plt.close()
    print("saved entropy_curves.png")


def plot_convergence_comparison():
    fig, ax = plt.subplots(figsize=(10, 6))

    for algo in ["dqn", "ppo", "a2c", "reinforce"]:
        sweep = load_sweep_table(algo)
        if sweep is None:
            continue

        if "mean_reward" not in sweep.columns:
            continue

        best_idx = sweep["mean_reward"].idxmax()
        best_run = int(sweep.loc[best_idx, "run_id"]) if "run_id" in sweep.columns else best_idx

        df = load_training_logs(algo)
        if df is None:
            continue

        run_data = df[df["run_id"] == best_run]
        if "episode_reward" in run_data.columns:
            rewards = run_data["episode_reward"].values
            smoothed = smooth(rewards, 30)
            ax.plot(smoothed, color=ALGO_COLORS[algo], linewidth=2, label=ALGO_LABELS[algo])

    ax.set_title("convergence comparison (best run per algorithm)")
    ax.set_xlabel("episode")
    ax.set_ylabel("episode reward (smoothed)")
    ax.legend()
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "convergence_comparison.png")
    plt.close()
    print("saved convergence_comparison.png")


def plot_best_run_comparison():
    fig, ax = plt.subplots(figsize=(8, 5))
    algo_names = []
    algo_rewards = []
    algo_stds = []
    colors = []

    for algo in ["dqn", "ppo", "a2c", "reinforce"]:
        sweep = load_sweep_table(algo)
        if sweep is None or "mean_reward" not in sweep.columns:
            continue

        best_idx = sweep["mean_reward"].idxmax()
        algo_names.append(ALGO_LABELS[algo])
        algo_rewards.append(sweep.loc[best_idx, "mean_reward"])
        algo_stds.append(sweep.loc[best_idx].get("std_reward", 0))
        colors.append(ALGO_COLORS[algo])

    if not algo_names:
        print("no sweep data for best run comparison")
        return

    bars = ax.bar(algo_names, algo_rewards, yerr=algo_stds, color=colors, capsize=5, alpha=0.85)
    ax.set_title("best run per algorithm")
    ax.set_ylabel("mean reward (last 100 episodes)")

    for bar, val in zip(bars, algo_rewards):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                f"{val:.1f}", ha="center", va="bottom", fontsize=10)

    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "best_run_comparison.png")
    plt.close()
    print("saved best_run_comparison.png")


def plot_generalization_test():
    from environment.custom_env import PowerGridEnv
    from stable_baselines3 import DQN, PPO, A2C

    fig, ax = plt.subplots(figsize=(10, 6))
    all_data = []
    labels = []
    colors = []

    model_dirs = {
        "dqn": (ROOT / "models" / "dqn", DQN),
        "ppo": (ROOT / "models" / "pg", PPO),
        "a2c": (ROOT / "models" / "pg", A2C),
    }

    for algo, (model_dir, cls) in model_dirs.items():
        if not model_dir.exists():
            continue

        sweep = load_sweep_table(algo)
        if sweep is None or "mean_reward" not in sweep.columns:
            continue

        best_idx = sweep["mean_reward"].idxmax()
        best_run = int(sweep.loc[best_idx, "run_id"]) if "run_id" in sweep.columns else best_idx

        model_path = model_dir / f"run_{best_run}_final.zip"
        if not model_path.exists():
            model_path = model_dir / f"{algo}_run_{best_run}_final.zip"
        if not model_path.exists():
            continue

        try:
            model = cls.load(model_path)
        except Exception:
            continue

        rewards = []
        env = PowerGridEnv()
        for seed in range(50):
            obs, _ = env.reset(seed=seed + 1000)
            total = 0.0
            for _ in range(1000):
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, term, trunc, _ = env.step(int(action))
                total += reward
                if term or trunc:
                    break
            rewards.append(total)
        env.close()

        all_data.append(rewards)
        labels.append(ALGO_LABELS[algo])
        colors.append(ALGO_COLORS[algo])

    if not all_data:
        print("no models available for generalization test")
        return

    bp = ax.boxplot(all_data, labels=labels, patch_artist=True)
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)

    ax.set_title("generalization test (50 episodes, different seeds)")
    ax.set_ylabel("total episode reward")
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "generalization_test.png")
    plt.close()
    print("saved generalization_test.png")


def main():
    print("generating plots...")
    plot_reward_curves()
    plot_dqn_loss()
    plot_entropy_curves()
    plot_convergence_comparison()
    plot_best_run_comparison()
    plot_generalization_test()
    print(f"all plots saved to {PLOTS_DIR}")


if __name__ == "__main__":
    main()
