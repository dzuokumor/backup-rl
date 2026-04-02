# CLAUDE.md — RL-for-PGM Project Constraints

## Identity
- **Student:** Dave
- **Platform:** Windows 11, Python 3.10+, NVIDIA GTX 1650Ti (CUDA)
- **Domain:** Power Grid Cascading Failure Mitigation RL agent (university assignment)
- **Capstone link:** DATMA (Dependency-Aware Temporal Memory Architecture)

## Current State (as of 2026-04-02)

### What is DONE:
- environment/ — custom Gymnasium env, IEEE 14-bus with Lagos/EKEDC naming, DC power flow, cascading failures. Passes check_env()
- environment/rendering.py — 3D OpenGL+Pygame nighttime Lagos city visualization
- training/dqn_training.py — 10-run hyperparameter sweep COMPLETE (100k steps each)
- training/pg_training.py — PPO/A2C/REINFORCE 10 runs each COMPLETE (100k steps each)
- training/train_1m.py — 1M step runs COMPLETE for DQN, PPO, A2C, REINFORCE
- training/train_ppo_optimized.py — NEVER RAN, env has configurable params for it
- evaluate.py — full evaluation script
- main.py — entry point with --random/--model/--episodes, logging to logs/
- api/serve.py — FastAPI POST /predict endpoint
- results/generate_plots.py — plot generation (needs to be run)
- tests/ — 65 tests all passing

### Training Results:
Best hyperparams (from 100k sweep):
- DQN run 6: lr=5e-4, gamma=0.95, [200,200]
- PPO run 7: lr=5e-4, gamma=0.95, clip=0.3, [128,128,128]
- A2C run 8: lr=7e-4, gamma=0.99, [200,200,200]
- REINFORCE run 8: lr=1e-4, gamma=0.9, [256,256]

1M results (final 100 mean): PPO=211.60, DQN=158.62, REINFORCE=62.10, A2C=32.16 (use 100k model at 88.34 instead)

Evaluation (50 eps): PPO 309.06 mean reward, 255 mean steps, 0.26 cascades. DQN 178.09/113 steps. Random -15.58/8 steps.

### What STILL NEEDS DOING:
1. train_ppo_optimized.py — uses env with fault_prob=0.001, recovery_steps=20, survival_bonus=0.2. Never ran.
2. Generate plots: python results/generate_plots.py
3. Visualization lagoon not visible from camera angle (low priority cosmetic)

### Environment is backwards compatible:
PowerGridEnv() — original defaults for all existing models
PowerGridEnv(fault_probability=0.001, max_disconnected_lines=10, fault_recovery_steps=20, survival_bonus=0.2) — optimized

## Code Style
- Minimal comments, lowercase prints, no ==== separators
- Training output every ~10k-50k steps
- Log to logs/ directory
- Use humanizer skill for documentation

## Git / Network
- Always unset HTTPS_PROXY before git/network commands
