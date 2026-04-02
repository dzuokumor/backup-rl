# Claude Code Brief: RL for Power Grid Management (v2 — Research-Grounded)

## Project Overview

Build a reinforcement learning project for a university assignment. The domain is **power grid cascading failure mitigation**, tied to a capstone project called DATMA (Dependency-Aware Temporal Memory Architecture) — which deals with structured temporal context in transformer models. The power grid domain is one of DATMA's target application areas (specifically EKEDC Lagos power distribution).

The RL agent acts as a real-time grid operator: detecting overloads, performing load shedding, isolating faults, and redispatching generation to prevent cascading failures — mirroring how published RL-for-power-grid research frames this problem (see: RL2Grid benchmark, Grid2Op L2RPN competitions, IEEE bus system cascading failure studies).

**Repo name:** `RL-for-PGM`
**Student name:** Dave
**Platform:** Windows, NVIDIA GTX 1650Ti (CUDA available)
**Python version:** 3.10+

---

## Nigeria Power Grid Context (USE THIS TO TAILOR THE ENVIRONMENT)

This environment is modeled after the real Lagos/EKEDC distribution network. The following Nigeria-specific details MUST inform environment design, naming, dynamics, and the report narrative.

### Real Grid Structure
- Nigeria's national grid operates at **330kV transmission**, stepped down to **132kV → 33kV → 11kV → 0.415kV** for distribution
- The grid is designed to operate at **50Hz ± 0.5Hz** (49.5-50.5Hz safe band) and **330kV ± 5%** voltage tolerance
- Nigeria has ~13,000MW installed generation capacity but the transmission network can barely wheel ~5,000MW — any additional load triggers collapse
- The grid has collapsed **200+ times between 2010-2022**, and **12 times in 2024 alone**. Three collapses in a single week have occurred.

### EKEDC Specifics (Our Target Network)
- **Eko Electricity Distribution Company (EKEDC)** covers **southern Lagos State** and Agbara (Ogun State)
- EKEDC operates **12 districts**: Lekki, Ibeju, Islands, Ajah, Ajele, Orile, Ijora, Apapa, Mushin, Festac, Ojo, Agbara
- Receives bulk power from **Akangba (330/132kV)** and **Ajah (330/132kV)** transmission stations
- Power flows through **10 transmission stations at 132/33kV** and **40 injection substations** (total 1,137.5MVA capacity)
- Receives **11-15% of total national grid allocation**
- EKEDC was the first DisCo to deploy **SCADA-DMS** (Supervisory Control and Data Acquisition) in Nigeria (2020)

### What Causes Failures (Model These in Dynamics)
- **Load shedding accounts for ~60% of all outages** — this is the primary control mechanism, not a last resort
- **Forced outages (equipment failure, weather, vandalism) account for ~30%** of outages
- **Planned maintenance outages ~10%** — last ~4 hours on average
- **Gas shortages** cripple gas-fired plants — generation can drop from 3,700MW to 2,000MW during crises
- **Cascading mechanism**: when a 330kV line trips (e.g., Omotosho–Ikeja West), load redistributes. If another critical line is on maintenance, the cascade causes widespread blackout across Lagos and Abuja. This happened in February 2025.
- **Frequency imbalance**: a spike from 50.33Hz to 51.44Hz caused a national grid collapse in November 2024
- **Physical damage**: trailers hitting tower bases, pipeline fires damaging 33kV feeders — model as random fault events

### Lagos-Specific Demand Patterns (For Load Profiles)
- **Evening peak** (17:00-22:00): residential + commercial overlap — highest demand
- **Morning ramp** (06:00-09:00): businesses opening
- **Overnight trough** (01:00-05:00): lowest demand
- **Band A feeders** are expected to provide 20 hours/day — model as high-priority loads
- **Band B-E feeders** get fewer hours — model as lower-priority (shed first)

### Naming Convention for Environment
Use Lagos/EKEDC naming for buses and districts to make the simulation authentic:
- Bus names should reference real EKEDC districts (Lekki, Ajah, Apapa, etc.)
- Transmission lines should reference real corridors (Omotosho–Ikeja West 330kV, Akangba–Ajah 132kV, etc.)
- Generator names should reference real power sources (Egbin thermal plant, Omotosho gas plant, etc.)
- This makes the visualization and report immediately recognizable as Nigeria-specific, not a generic IEEE benchmark

---

## CRITICAL: Training Checkpoint & Resume System

**The developer is training from Nigeria where power outages are frequent and unpredictable.** (Yes, the irony of building a power grid RL agent while experiencing grid failures is noted.)

ALL training scripts MUST implement robust checkpoint saving and resume:

### Requirements
1. **Auto-save checkpoints every N timesteps** (default: every 10,000 timesteps or every 5 minutes, whichever comes first)
2. **Save complete training state**, not just the model:
   - Model weights
   - Replay buffer (for DQN)
   - Optimizer state
   - Current timestep count
   - Episode count
   - All logged metrics so far (rewards, losses, etc.)
   - Hyperparameter config for this run
   - Random seed state (numpy + torch)
3. **Resume flag**: `python training/dqn_training.py --resume` should:
   - Detect the latest checkpoint for the current run
   - Load all state
   - Continue training from exactly where it stopped
   - Append to (not overwrite) existing logs
4. **Per-run checkpoints**: each hyperparameter run (1-10) saves independently so a crash during run 7 doesn't lose runs 1-6
5. **Completed run detection**: if a run already finished (reached total_timesteps), skip it on resume instead of retraining
6. **Checkpoint directory**: `models/checkpoints/{algorithm}/{run_id}/`

### Implementation Pattern
```python
# Pseudocode for what every training script should do:
for run_id, hyperparams in enumerate(sweep_configs):
    checkpoint_dir = f"models/checkpoints/{algo}/{run_id}/"
    
    # Skip if this run already completed
    if os.path.exists(f"{checkpoint_dir}/COMPLETED"):
        print(f"Run {run_id} already completed, skipping...")
        continue
    
    # Resume if checkpoint exists
    if os.path.exists(f"{checkpoint_dir}/latest_checkpoint.zip"):
        model = Algorithm.load(f"{checkpoint_dir}/latest_checkpoint")
        remaining_steps = total_timesteps - loaded_timestep_count
        print(f"Resuming run {run_id} from step {loaded_timestep_count}")
    else:
        model = Algorithm(env=env, **hyperparams)
        remaining_steps = total_timesteps
    
    # Train with periodic saving via callback
    model.learn(total_timesteps=remaining_steps, callback=checkpoint_callback)
    
    # Mark complete
    open(f"{checkpoint_dir}/COMPLETED", 'w').close()
    
    # Save final model + metrics
    model.save(f"models/{algo}/run_{run_id}_final")
    save_metrics(f"results/logs/{algo}_run_{run_id}.csv")
```

### SB3 Checkpoint Callback
Use `stable_baselines3.common.callbacks.CheckpointCallback` as base, but extend it to also save:
- The replay buffer (DQN): `model.save_replay_buffer()`
- Accumulated metrics
- A JSON file with the hyperparameter config + current progress

This is non-negotiable. Training 40 runs on a 1650Ti will take hours. Losing progress to a power cut is unacceptable.

---

## Project Structure (MANDATORY — follow exactly)

```
RL-for-PGM/
├── environment/
│   ├── custom_env.py            # Custom Gymnasium environment
│   ├── rendering.py             # OpenGL/3D visualization
│   ├── grid_topology.py         # Network topology definition (buses, lines, connections)
├── training/
│   ├── dqn_training.py          # DQN training + hyperparameter sweep
│   ├── pg_training.py           # REINFORCE, PPO, A2C training + sweeps
├── models/
│   ├── dqn/                     # Saved DQN models (best + all runs)
│   └── pg/                      # Saved PG models (best + all runs)
├── results/
│   ├── plots/                   # All generated plots (reward curves, loss, entropy, etc.)
│   ├── tables/                  # Hyperparameter sweep results as CSV
│   └── logs/                    # TensorBoard or CSV training logs
├── api/
│   └── serve.py                 # FastAPI endpoint for model inference
├── main.py                      # Entry point — runs best model with visualization
├── requirements.txt             # All dependencies (MANDATORY)
└── README.md                    # Project documentation
```

---

## Environment Design

### Domain: Power Grid Cascading Failure Mitigation

**Inspired by the IEEE 14-bus system** — the standard benchmark used in power grid RL research (used in Grid2Op L2RPN competitions, cascading failure mitigation papers, and the RL2Grid benchmark). We adapt it to represent a Lagos (EKEDC) distribution context.

**Grid topology:**
- **14 substations (buses)** connected by **20 transmission lines**
- **5 generators** at specific buses (buses 1, 2, 3, 6, 8)
- **11 load points** (consumer demand) at the remaining buses
- **Topology is a graph** — not fully connected. Lines connect specific bus pairs based on the IEEE 14-bus layout.
- Store the topology in `grid_topology.py` as adjacency lists / connection definitions

**Why IEEE 14-bus:** It's complex enough to demonstrate cascading behavior (20 interconnected lines, mixed generation/load) while being tractable for training on a GTX 1650Ti. This is the exact system used in published cascading failure RL papers.

### Observation Space

For each of the 20 transmission lines:
- `line_loading_ratio` (float, 0-1+): current flow / thermal limit. Above 1.0 = overloaded
- `line_status` (int, 0/1): connected or disconnected
- `time_since_overload` (int, 0-N): timesteps line has been in overload (lines auto-disconnect after exceeding thermal limit for too long, per Grid2Op convention)

For each of the 14 buses:
- `voltage_magnitude` (float, ~0.9-1.1 p.u.): bus voltage in per-unit
- `active_power_injection` (float): net power at bus (generation minus load)
- `is_load_shed` (int, 0/1): whether load shedding is active at this bus

For each of the 5 generators:
- `generator_output` (float, 0-1): current output as fraction of max capacity
- `generator_available` (int, 0/1): whether generator is online

Global features:
- `total_load_demand` (float): aggregate consumer demand
- `total_generation` (float): aggregate supply
- `time_of_day` (float, 0-1): normalized time (affects demand patterns)
- `num_overloaded_lines` (int): count of lines above thermal limit
- `num_disconnected_lines` (int): count of disconnected lines

Total observation: (20 × 3) + (14 × 3) + (5 × 2) + 5 = **117-dimensional continuous vector**
Use `gymnasium.spaces.Box` for the observation space.

**Rationale:** This mirrors the observation structure used in Grid2Op and published research — line flows, bus voltages, and generation status are the standard observable quantities in power grid RL.

### Action Space (Discrete)

The agent picks ONE action per step. Action categories based on established remedial actions from the literature:

**Category 1: Do Nothing (1 action)**
0. `DO_NOTHING` — maintain current topology (this is often the optimal action when the grid is stable)

**Category 2: Load Shedding — per load bus (11 actions)**
1-11. `SHED_LOAD_BUS_X` — reduce load by 20% at load bus X (standard shedding increment from literature). If already shed, sheds another 20%.

**Category 3: Restore Load — per load bus (11 actions)**
12-22. `RESTORE_LOAD_BUS_X` — restore previously shed load at bus X

**Category 4: Line Switching (20 actions)**
23-42. `TOGGLE_LINE_X` — disconnect or reconnect transmission line X (the primary topological action in Grid2Op / L2RPN)

**Category 5: Generator Redispatch (10 actions)**
43-47. `INCREASE_GEN_X` — increase output of generator X by 10% of capacity
48-52. `DECREASE_GEN_X` — decrease output of generator X by 10% of capacity

Total: **53 discrete actions**
Use `gymnasium.spaces.Discrete(53)` for compatibility with all SB3 algorithms.

**Constraints on actions (enforced in `step()`):**
- Line switching has a **cooldown** — a line that was just switched cannot be switched again for N steps (standard in Grid2Op)
- Load shedding cannot reduce a bus below 0
- Generator output is bounded by [min_output, max_output]
- Invalid actions (e.g., shedding at a bus with no load) are treated as DO_NOTHING with a small penalty

**Rationale:** These action categories directly match what's used in cascading failure RL research. Load shedding + line switching + generation redispatch are the three standard remedial action types. The "one action per step" constraint matches Grid2Op and most published work.

### Reward Structure

Based on published reward designs from cascading failure mitigation papers:

```python
reward = (
    # Primary: maximize power delivery (proportion of demand served)
    + 2.0 * (total_load_served / total_load_demand)

    # Penalty: line overloads (proportional to severity)
    - 1.0 * sum(max(0, line_loading_ratio - 1.0) for each line)

    # Penalty: load shedding (necessary evil, but penalized to minimize it)
    - 0.5 * (total_load_shed / total_load_demand)

    # Penalty: line disconnections (topology degradation)
    - 0.3 * (num_disconnected_lines / total_lines)

    # Heavy penalty: cascading failure (game over condition)
    - 10.0 * cascading_failure_occurred

    # Small penalty: action cost (discourages unnecessary interventions)
    - 0.1 * (action != DO_NOTHING)

    # Penalty: voltage violations
    - 0.5 * sum(max(0, |voltage - 1.0| - 0.05) for each bus)
)
```

**Rationale:** Published research combines operational costs with constraint violation penalties. The key insight from the literature is that reward components should be scaled to similar orders of magnitude so no single component dominates. Power delivery is the primary objective; everything else is a constraint/penalty.

### Dynamics / Transitions

**Load demand:**
- Follows sinusoidal daily pattern (peak at ~18:00, trough at ~04:00) + Gaussian noise + random demand spikes
- Each load bus has slightly different demand patterns and magnitudes
- Reflects Lagos-specific patterns: evening peak from residential + commercial overlap

**Power flow computation:**
- Use a simplified DC power flow model (standard in RL papers — DC approximation is computationally fast and sufficient for this scale)
- Compute line flows from bus injections and network admittance matrix
- If a line's flow exceeds its thermal limit, start counting overload timesteps

**Cascading failure mechanism:**
- A line that exceeds its thermal limit for `max_overload_steps` (e.g., 3 steps) is **automatically disconnected**
- When a line disconnects, power flow redistributes across remaining lines — this can overload OTHER lines (the cascade)
- If redistribution causes another line to exceed limits → that line also starts its overload countdown → potential chain reaction
- This is exactly how cascading failures work in Grid2Op and in real power systems

**Generator behavior:**
- Generators have ramp rate limits (can't change output instantaneously)
- If a generator is isolated from all loads (islanding), it trips offline
- Backup generator at bus 8 starts offline and can be brought online (limited fuel)

**Fault injection:**
- Random line faults occur with small probability each step (modeling weather, equipment failure)
- Fault probability increases for lines with high loading ratios
- Faults immediately disconnect the line

### Start State
- All lines connected
- All generators online at moderate output
- Load at moderate levels (time_of_day randomized)
- No active faults, no load shedding
- Small random perturbation to loading levels

### Terminal Conditions
- **Failure (terminated=True):** total load served drops below 30% of demand (widespread blackout) OR 8+ lines simultaneously disconnected
- **Success (truncated=True):** survive 1000 timesteps (represents one operational day at ~1.5 min per step)
- **Max steps:** 1000

### Environment Must:
- Subclass `gymnasium.Env`
- Implement `reset()`, `step()`, `render()`, `close()`
- Return proper `(obs, reward, terminated, truncated, info)` tuples
- Be compatible with Stable Baselines 3 `check_env()`
- Include a `render_mode` parameter supporting "human" and "rgb_array"
- The `info` dict should include human-readable details: action name, lines overloaded, load served %, faults active

---

## Visualization (OpenGL — Exemplary Tier)

Use **PyOpenGL + Pygame** (Pygame as the window manager, OpenGL for rendering). Build a high-quality 2.5D network visualization showing:

- **Network topology graph**: 14 buses as nodes, 20 lines as edges — laid out to match the IEEE 14-bus topology (this is a well-known graph layout, reference images available online)
- **Bus status**: color-coded nodes
  - Green = normal voltage
  - Yellow = voltage slightly out of range
  - Orange = load being shed at this bus
  - Red = severe voltage violation
  - Gray = islanded / disconnected
- **Line status**: color-coded edges
  - Green = normal loading (<80% thermal limit)
  - Yellow = approaching limit (80-100%)
  - Red = overloaded (>100%)
  - Dashed/dim = disconnected
  - Line thickness proportional to loading ratio
- **Generator indicators**: special markers at generator buses showing output level
- **Load bars**: small bar at each load bus showing demand vs served
- **Agent action highlight**: animate which element the agent is acting on (flash the line being switched, pulse the bus being shed)
- **Info panel** (rendered as overlay or side panel):
  - Current timestep / max steps
  - Total reward (cumulative)
  - Load served %
  - Lines overloaded count
  - Lines disconnected count
  - Current action name
  - Time of day

**Cascading effect animation**: when a line auto-disconnects from overload, show a visual ripple along connected lines to indicate flow redistribution.

The visualization must work in real-time as the agent takes actions.

Create a random-agent demo accessible via `python main.py --random` — this is explicitly required by the assignment.

---

## RL Training

Use **Stable Baselines 3** for all four algorithms. All must use the SAME environment.

### Algorithms

1. **DQN** (Value-Based)
2. **REINFORCE** — Note: SB3 does not have vanilla REINFORCE. Implement manually using PyTorch OR use A2C with specific settings that approximate REINFORCE (no value baseline, single-episode updates). Document this clearly in comments and README.
3. **PPO** (Policy Gradient) — the most commonly used algorithm in Grid2Op / L2RPN research
4. **A2C** (Actor-Critic)

### Hyperparameter Sweep

**10 runs per algorithm = 40 total runs.** Each run varies MULTIPLE hyperparameters, not just one.

For each run, log:
- All hyperparameter values
- Final mean reward (over last 100 episodes)
- Training time
- Convergence episode (if applicable)
- Reward curve data

Save results as CSV tables with columns for every hyperparameter + results.

**DQN hyperparameters to vary across 10 runs:**
- learning_rate: [1e-4, 3e-4, 5e-4, 1e-3, 5e-3]
- gamma: [0.95, 0.99, 0.999]
- buffer_size: [10000, 50000, 100000]
- batch_size: [32, 64, 128]
- exploration_fraction: [0.1, 0.2, 0.3]
- exploration_final_eps: [0.01, 0.05, 0.1]
- target_update_interval: [500, 1000, 5000]
- net_arch: [[64,64], [128,128], [256,256]]

**PPO hyperparameters to vary:**
- learning_rate: [1e-4, 3e-4, 5e-4, 1e-3]
- gamma: [0.95, 0.99, 0.999]
- clip_range: [0.1, 0.2, 0.3]
- n_steps: [256, 512, 1024, 2048]
- batch_size: [32, 64, 128]
- n_epochs: [3, 5, 10]
- ent_coef: [0.0, 0.01, 0.05]
- gae_lambda: [0.9, 0.95, 0.99]

**A2C hyperparameters to vary:**
- learning_rate: [1e-4, 3e-4, 7e-4, 1e-3]
- gamma: [0.95, 0.99, 0.999]
- n_steps: [5, 16, 32, 64]
- ent_coef: [0.0, 0.01, 0.05]
- vf_coef: [0.25, 0.5, 0.75]
- max_grad_norm: [0.5, 1.0, 5.0]
- net_arch: [[64,64], [128,128]]

**REINFORCE hyperparameters to vary:**
- learning_rate: [1e-4, 3e-4, 5e-4, 1e-3, 5e-3]
- gamma: [0.9, 0.95, 0.99, 0.999]
- net_arch: [[64,64], [128,128], [256,256]]
- max_grad_norm: [0.5, 1.0, 5.0]
- (if using baseline) baseline_coef: [0.5, 0.75, 1.0]

**Training duration**: Use enough timesteps for meaningful learning. Start with 100k total_timesteps per run. Increase if needed based on convergence. GPU (1650Ti) should handle this fine.

### Training Output

Each training script must:
- Save the trained model to `models/dqn/` or `models/pg/`
- Save training metrics to `results/logs/`
- Print progress during training
- Use callbacks to log episode rewards

---

## Plots to Generate (save to results/plots/)

All plots must be well-labeled with axis labels, titles, and legends.

1. **Cumulative reward curves** — all 4 algorithms on subplots (same figure, shared x-axis)
2. **DQN loss/objective curves** over training
3. **Policy gradient entropy curves** (PPO, A2C, REINFORCE) over training
4. **Convergence comparison** — episode where each algorithm stabilizes
5. **Best run per algorithm comparison** — bar chart or line chart of final performance
6. **Generalization test** — run each best model for 50 episodes with different random seeds, plot distribution of total rewards

Use matplotlib. Make them publication-quality (not default matplotlib styling — use seaborn style or custom rcParams).

---

## FastAPI Endpoint (api/serve.py)

Simple endpoint that:
- Loads the best trained model
- Accepts a POST request with the current observation (JSON)
- Returns the agent's action (JSON)
- Shows how this could integrate into a real SCADA/grid monitoring dashboard

```python
# Example API contract:
# POST /predict
# Body: {"observation": [0.85, 0.92, ...117 values...]}
# Response: {"action": 23, "action_name": "TOGGLE_LINE_3", "action_category": "line_switching", "confidence": 0.87}
```

---

## main.py

Entry point that:
1. Loads the best performing model (across all 4 algorithms)
2. Runs it in the environment with visualization ON
3. Prints verbose terminal output each step:
   - Step number
   - Action taken (name + category)
   - Lines overloaded (which ones, by how much)
   - Lines disconnected
   - Load served %
   - Step reward and cumulative reward
   - Any cascading events
4. Supports command line args:
   - `--model` to specify which model to run (default: best)
   - `--random` to run random agent (for the static demo requirement)
   - `--episodes` number of episodes to run

---

## requirements.txt

Must include all dependencies. Key ones:
- gymnasium
- stable-baselines3
- torch (with CUDA support)
- PyOpenGL
- pygame
- fastapi
- uvicorn
- matplotlib
- seaborn
- numpy
- pandas
- scipy (for power flow computation if needed)

---

## README.md

Include:
- Project description (Power Grid Cascading Failure Mitigation RL — connected to DATMA capstone)
- Grid topology description (IEEE 14-bus inspired, adapted for Lagos/EKEDC context)
- Setup instructions (pip install -r requirements.txt)
- How to train all models
- How to run the visualization (main.py with options)
- How to start the API
- Project structure explanation
- Brief results summary
- References to relevant research (Grid2Op, RL2Grid, IEEE bus systems)

---

## IMPORTANT NOTES

- **REINFORCE**: SB3 doesn't have vanilla REINFORCE. Either implement from scratch in PyTorch using the same env interface, or use a well-documented workaround. Either way, document the approach clearly.
- **All 4 algorithms MUST use the exact same environment** for fair comparison.
- **Random agent demo is required** — show the visualization working WITHOUT any trained model.
- **Environment must pass `check_env()`** from stable_baselines3.
- **GPU**: Code should auto-detect CUDA and use GPU when available.
- **Windows compatibility**: All file paths must use os.path or pathlib. No hardcoded unix paths.
- The assignment explicitly says the project will be **cloned and executed**, so everything must work out of the box after `pip install -r requirements.txt`.
- **DC Power Flow**: Use the simplified DC approximation (P = B × θ) for computing line flows. This is standard in RL research for power grids — it's fast and adequate. Do NOT try to implement full AC power flow.
- **IEEE 14-bus topology**: The bus connections and line parameters are well-documented. Use standard values but scale them to make the environment challenging enough for RL (i.e., thermal limits shouldn't be so generous that nothing ever overloads).

---

## Research References (for Claude Code context, README, and report)

These are the published works that inform this environment design:

1. **Grid2Op** — Open-source RL framework for power grid operations. Gymnasium-compatible. Uses IEEE bus systems. Our custom env mirrors its design patterns (line thermal limits, auto-disconnection, topology control, DC power flow).
   - github.com/Grid2op/grid2op

2. **RL2Grid Benchmark** (2025) — Benchmarking RL in power grid operations. Defines standard observation/action spaces, constraint formulations, and reward designs for grid RL.

3. **Cascading Failure Mitigation by RL** (Zhu, 2021 / ICML Workshop) — RL agent on IEEE 118-bus using DDPG/Actor-Critic for load shedding during cascading failures. Our reward structure draws from this work.

4. **DRL-Based Load Shedding with Communication Delay** (2023) — SAC agent on IEEE 14 and 30 bus systems. Actions = 20% load reduction. Observation = generator power, line flows, bus voltages.

5. **Graph RL for Power Grid Topology Control** (2025) — Distributed RL with GNN on Grid2Op. Shows how topology actions (line switching) are the primary control mechanism.

6. **Safe RL for Grid Voltage Control** (2021) — Reward includes voltage magnitude recovery targets + load shedding amounts + invalid action penalties.

### Nigeria-Specific References (for report and environment authenticity)

7. **"Analysis of the Causes of Outages in the Distribution System of Lagos State, Nigeria"** (2023, FUOYE Journal of Engineering) — EKEDC and IKEDC fault data. Key finding: transient faults (~357/month) and jumper cuts (~353/month) are the dominant outage causes. Monthly cost: ₦631m-₦1.66b. Monthly energy lost: 100,415 MWh. This directly informs our fault injection probabilities.

8. **"Lagos Urban Power Profile"** (2023, Resilient Cities Network) — Lagos accounts for 40% of Nigeria's electricity demand but receives only 24-26% of grid capacity (~1,000MW of the ~9,000MW needed). Average supply: ~4 hours/day. From 2010 to mid-2022, Nigeria's grid suffered 222 partial and total collapses. On July 20, 2022, transmission crashed from 3,921MW to 50MW in 6 hours.

9. **"Electric Grid Reliability: An Assessment of the Nigerian Power Grid"** (2023, Covenant Journal of Engineering) — Documents grid collapse statistics from 2000-2022. Identifies causes: generator outages causing voltage collapse, line tripping, frequency imbalances. Nigeria's grid is radial and weak with low voltage profiles, especially in the north.

10. **"Investigating Electricity Power Grid Collapses in Nigeria and Measures to Reduce its Effect"** (2025, NIPES Journal) — Uses a 48-bus transmission model of Nigeria. Evaluates voltage stability under various compensation schemes. Most recent study on Nigerian grid collapse mitigation.

11. **EKEDC Operations Data** (ekedp.com) — EKEDC receives bulk power from Akangba and Ajah 330/132kV stations, distributed through 10 transmission stations (132/33kV) and 40 injection substations (1,137.5MVA total). Covers 12 districts. Serves ~614,370 registered customers. Legacy infrastructure described as "dilapidated and unsafe."

12. **Nigerian National Grid Collapse Timeline 2024** (Guardian Nigeria, Intelpoint) — 12 collapses in 2024. Feb 4: grid dropped from 2,407MW to 31MW to zero. Oct 14-15: back-to-back collapses. Dec 29, 2025: another collapse to 50MW. Intervals between collapses as short as 1 day.

---

## What This Connects To (For README/Report context)

This power grid management agent simulates the type of temporal decision-making that DATMA (Dependency-Aware Temporal Memory Architecture) is designed to improve. In a real deployment:
- DATMA would segment the temporal stream of grid observations into meaningful episodes (fault events, demand cycles, cascade sequences)
- DATMA's dependency tracker maps directly to the physical dependencies between buses/lines — when line 3 trips, DATMA would track which other lines are now at risk
- DATMA's resolution detection would identify when a fault event is "over" and compress it into a sub-conclusion
- DATMA's predictive compression would retain what matters about past grid events for future prediction, not just reconstruct what happened
- The RL agent would use DATMA-structured observations (segments + dependencies + sub-conclusions) instead of flat observation vectors, enabling better long-horizon decision making

This is a proof-of-concept showing the RL decision layer that DATMA's memory architecture would eventually serve.