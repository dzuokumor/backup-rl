import json
import sys
import os
import shutil
import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from environment.custom_env import PowerGridEnv, OBS_DIM, NUM_ACTIONS


@pytest.fixture
def tmp_dir():
    d = tempfile.mkdtemp()
    yield Path(d)
    shutil.rmtree(d, ignore_errors=True)


class TestDQNTraining:
    def test_hyperparams_count(self):
        from training.dqn_training import HYPERPARAMS
        assert len(HYPERPARAMS) == 10

    def test_hyperparams_have_required_keys(self):
        from training.dqn_training import HYPERPARAMS
        required = ["learning_rate", "gamma", "buffer_size", "batch_size",
                     "exploration_fraction", "exploration_final_eps",
                     "target_update_interval", "net_arch"]
        for cfg in HYPERPARAMS:
            for key in required:
                assert key in cfg, f"missing {key} in run {cfg.get('run_id')}"

    def test_hyperparams_vary_across_runs(self):
        from training.dqn_training import HYPERPARAMS
        lrs = set(cfg["learning_rate"] for cfg in HYPERPARAMS)
        gammas = set(cfg["gamma"] for cfg in HYPERPARAMS)
        assert len(lrs) > 1
        assert len(gammas) > 1

    def test_dqn_can_create_model(self):
        from stable_baselines3 import DQN
        env = PowerGridEnv()
        model = DQN("MlpPolicy", env, learning_rate=1e-4, buffer_size=1000,
                     batch_size=32, verbose=0, device="cpu")
        obs, _ = env.reset()
        action, _ = model.predict(obs)
        assert env.action_space.contains(int(action))
        env.close()

    def test_dqn_short_training(self):
        from stable_baselines3 import DQN
        env = PowerGridEnv()
        model = DQN("MlpPolicy", env, learning_rate=1e-3, buffer_size=500,
                     batch_size=32, learning_starts=100, verbose=0, device="cpu")
        model.learn(total_timesteps=200)
        obs, _ = env.reset()
        action, _ = model.predict(obs)
        assert env.action_space.contains(int(action))
        env.close()

    def test_checkpoint_save_load(self, tmp_dir):
        from stable_baselines3 import DQN
        env = PowerGridEnv()
        model = DQN("MlpPolicy", env, learning_rate=1e-3, buffer_size=500,
                     batch_size=32, verbose=0, device="cpu")
        model.learn(total_timesteps=200)

        model.save(str(tmp_dir / "test_model"))
        assert (tmp_dir / "test_model.zip").exists()

        loaded = DQN.load(str(tmp_dir / "test_model"), env=env)
        obs, _ = env.reset(seed=99)
        a1, _ = model.predict(obs, deterministic=True)
        a2, _ = loaded.predict(obs, deterministic=True)
        assert a1 == a2
        env.close()

    def test_replay_buffer_save_load(self, tmp_dir):
        from stable_baselines3 import DQN
        env = PowerGridEnv()
        model = DQN("MlpPolicy", env, buffer_size=500, batch_size=32,
                     learning_starts=100, verbose=0, device="cpu")
        model.learn(total_timesteps=200)

        model.save_replay_buffer(str(tmp_dir / "replay"))
        assert (tmp_dir / "replay.pkl").exists() or (tmp_dir / "replay_buffer.pkl").exists() or any(tmp_dir.glob("replay*"))

        env.close()

    def test_completed_marker(self, tmp_dir):
        marker = tmp_dir / "COMPLETED"
        marker.touch()
        assert marker.exists()


class TestPGTraining:
    def test_ppo_hyperparams_grid(self):
        from training.pg_training import PPO_HYPERPARAM_GRID
        assert "learning_rate" in PPO_HYPERPARAM_GRID
        assert "gamma" in PPO_HYPERPARAM_GRID
        assert "clip_range" in PPO_HYPERPARAM_GRID

    def test_a2c_hyperparams_grid(self):
        from training.pg_training import A2C_HYPERPARAM_GRID
        assert "learning_rate" in A2C_HYPERPARAM_GRID
        assert "n_steps" in A2C_HYPERPARAM_GRID

    def test_reinforce_hyperparams_grid(self):
        from training.pg_training import REINFORCE_HYPERPARAM_GRID
        assert "learning_rate" in REINFORCE_HYPERPARAM_GRID
        assert "gamma" in REINFORCE_HYPERPARAM_GRID
        assert "net_arch" in REINFORCE_HYPERPARAM_GRID

    def test_sample_hyperparams_reproducible(self):
        from training.pg_training import sample_hyperparams, PPO_HYPERPARAM_GRID
        c1 = sample_hyperparams(PPO_HYPERPARAM_GRID, 5, seed=42)
        c2 = sample_hyperparams(PPO_HYPERPARAM_GRID, 5, seed=42)
        assert c1 == c2

    def test_sample_hyperparams_count(self):
        from training.pg_training import sample_hyperparams, PPO_HYPERPARAM_GRID
        configs = sample_hyperparams(PPO_HYPERPARAM_GRID, 10, seed=42)
        assert len(configs) == 10

    def test_ppo_short_training(self):
        from stable_baselines3 import PPO
        env = PowerGridEnv()
        model = PPO("MlpPolicy", env, n_steps=64, batch_size=32,
                     n_epochs=2, verbose=0, device="cpu")
        model.learn(total_timesteps=128)
        obs, _ = env.reset()
        action, _ = model.predict(obs)
        assert env.action_space.contains(int(action))
        env.close()

    def test_a2c_short_training(self):
        from stable_baselines3 import A2C
        env = PowerGridEnv()
        model = A2C("MlpPolicy", env, n_steps=16, verbose=0, device="cpu")
        model.learn(total_timesteps=64)
        obs, _ = env.reset()
        action, _ = model.predict(obs)
        assert env.action_space.contains(int(action))
        env.close()


class TestREINFORCE:
    def test_reinforce_policy_network(self):
        from training.pg_training import PolicyNetwork
        net = PolicyNetwork(OBS_DIM, NUM_ACTIONS, [64, 64])
        obs = torch.randn(1, OBS_DIM)
        probs = net(obs)
        assert probs.shape == (1, NUM_ACTIONS)
        assert torch.allclose(probs.sum(dim=1), torch.tensor(1.0), atol=1e-5)

    def test_reinforce_policy_different_archs(self):
        from training.pg_training import PolicyNetwork
        for arch in [[64, 64], [128, 128], [256, 256]]:
            net = PolicyNetwork(OBS_DIM, NUM_ACTIONS, arch)
            obs = torch.randn(1, OBS_DIM)
            probs = net(obs)
            assert probs.shape == (1, NUM_ACTIONS)

    def test_reinforce_agent_init(self):
        from training.pg_training import ReinforceAgent
        agent = ReinforceAgent(OBS_DIM, NUM_ACTIONS, lr=1e-3, gamma=0.99,
                               net_arch=[64, 64], max_grad_norm=1.0)
        obs = np.random.randn(OBS_DIM).astype(np.float32)
        action = agent.select_action(obs)
        assert 0 <= action < NUM_ACTIONS

    def test_reinforce_save_load(self, tmp_dir):
        from training.pg_training import ReinforceAgent
        agent = ReinforceAgent(OBS_DIM, NUM_ACTIONS, lr=1e-3, gamma=0.99,
                               net_arch=[64, 64], max_grad_norm=1.0)

        obs = np.random.randn(OBS_DIM).astype(np.float32)
        a1 = agent.select_action(obs)

        agent.save(str(tmp_dir / "reinforce_test"))
        agent2 = ReinforceAgent(OBS_DIM, NUM_ACTIONS, lr=1e-3, gamma=0.99,
                                net_arch=[64, 64], max_grad_norm=1.0)
        agent2.load(str(tmp_dir / "reinforce_test"))

        torch.manual_seed(42)
        a1 = agent.select_action(obs)
        torch.manual_seed(42)
        a2 = agent2.select_action(obs)
        assert a1 == a2


class TestMetricsLogging:
    def test_csv_write_read(self, tmp_dir):
        import csv
        path = tmp_dir / "test_metrics.csv"
        with open(path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["step", "reward", "loss"])
            writer.writerow([100, 1.5, 0.01])
            writer.writerow([200, 2.0, 0.005])

        import pandas as pd
        df = pd.read_csv(path)
        assert len(df) == 2
        assert "reward" in df.columns

    def test_json_meta_roundtrip(self, tmp_dir):
        meta = {
            "timestep": 5000,
            "episode_count": 50,
            "episode_rewards": [1.0, 2.0, 1.5],
            "run_id": 3,
        }
        path = tmp_dir / "meta.json"
        with open(path, "w") as f:
            json.dump(meta, f)
        with open(path) as f:
            loaded = json.load(f)
        assert loaded["timestep"] == 5000
        assert loaded["run_id"] == 3


class TestDeviceDetection:
    def test_cuda_detection(self):
        has_cuda = torch.cuda.is_available()
        if has_cuda:
            assert torch.cuda.device_count() > 0
