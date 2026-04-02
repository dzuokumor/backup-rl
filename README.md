# RL-for-PGM

Reinforcement learning agent for power grid cascading failure mitigation. Built for a university assignment as part of the DATMA (Dependency-Aware Temporal Memory Architecture) capstone project.

The agent acts as a grid operator for a simulated Lagos/EKEDC distribution network. It detects overloads, sheds load, isolates faults, and redispatches generation to keep the lights on. The environment is based on the IEEE 14-bus system, adapted with real EKEDC district names and Nigerian grid characteristics.

Nigeria's national grid has collapsed 200+ times since 2010. EKEDC covers southern Lagos and operates 12 districts through 40 injection substations. The simulation reflects this, down to the evening demand peaks and random fault events that mirror actual outage patterns from published studies.

## Grid topology

14 substations connected by 20 transmission lines, with 5 generators. The topology follows the IEEE 14-bus layout but uses Lagos names (Egbin, Akangba, Lekki, Apapa, etc.) and Nigerian voltage levels (330kV/132kV/33kV).

Power flow uses the DC approximation (P = B * theta). Lines that exceed their thermal limit for 3 consecutive steps auto-disconnect, and the resulting flow redistribution can cascade into further overloads. This is how real cascading failures work, and it's the same mechanism used in Grid2Op and the L2RPN competitions.

## Setup

```
pip install -r requirements.txt
```

Needs Python 3.10+ and a GPU helps for training (tested on GTX 1650 Ti with CUDA). The visualization needs PyOpenGL and pygame, which are included in requirements.txt.

## Training

Four algorithms, 10 hyperparameter runs each:

```
python training/dqn_training.py
python training/pg_training.py --algorithm ppo
python training/pg_training.py --algorithm a2c
python training/pg_training.py --algorithm reinforce
python training/pg_training.py --algorithm all
```

All training scripts support `--resume` to pick up where they left off. Checkpoints save every 10k steps. This matters because training happens in Nigeria, where the grid actually does go down mid-run.

REINFORCE is implemented from scratch in PyTorch since SB3 doesn't have vanilla REINFORCE. The other three use Stable Baselines 3.

## Running the agent

```
python main.py                    # best trained model with visualization
python main.py --random           # random agent demo
python main.py --model path.zip   # specific model
python main.py --episodes 5       # multiple episodes
python main.py --no-render        # headless
```

The `--random` flag is required for the assignment's static demo.

## API

```
python -m api.serve
```

Starts a FastAPI server on port 8000. Send observations, get back actions:

```
POST /predict
{"observation": [0.85, 0.92, ...117 values...]}

Response:
{"action": 23, "action_name": "toggle_Akangba-Lekki 33kV", "action_category": "line_switching", "confidence": 0.87}
```

## Generating plots

```
python results/generate_plots.py
```

Outputs to results/plots/. Reward curves, loss curves, entropy, convergence comparison, best run bar chart, and a generalization test across 50 random seeds.

## Project structure

```
environment/         grid env, topology, OpenGL renderer
training/            DQN and policy gradient training scripts
models/              saved models and checkpoints
results/             plots, tables, training logs
api/                 FastAPI inference endpoint
main.py              entry point with visualization
```

## How this connects to DATMA

The power grid domain is one of DATMA's target applications. In a real deployment, DATMA would segment the temporal stream of grid observations into fault events and demand cycles, track physical dependencies between buses and lines, and provide the RL agent with structured context instead of flat observation vectors. This repo is the RL decision layer that DATMA's memory architecture would eventually sit behind.

## References

- Grid2Op framework (github.com/Grid2op/grid2op)
- RL2Grid benchmark (2025)
- Cascading failure mitigation by RL, Zhu 2021
- DRL-based load shedding with communication delay (2023), IEEE 14 and 30 bus
- Analysis of outage causes in Lagos distribution system (2023, FUOYE Journal)
- Lagos urban power profile (2023, Resilient Cities Network)
- Nigerian grid collapse statistics 2000-2022 (Covenant Journal of Engineering)
- EKEDC operations data (ekedp.com)
