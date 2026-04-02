import gymnasium as gym
import numpy as np
from gymnasium import spaces

from environment.grid_topology import (
    NUM_BUSES, NUM_LINES, NUM_GENERATORS, NUM_LOADS,
    BUS_NAMES, GENERATOR_BUSES, GENERATOR_NAMES,
    GENERATOR_MAX_OUTPUT, GENERATOR_MIN_OUTPUT, GENERATOR_RAMP_RATE,
    GENERATOR_INIT_OUTPUT, GENERATOR_INIT_AVAILABLE,
    LOAD_BUSES_ALL, BASE_LOAD_DEMAND, LOAD_PRIORITY,
    LINE_CONNECTIONS, LINE_NAMES, LINE_REACTANCE,
    THERMAL_LIMITS, MAX_OVERLOAD_STEPS, LINE_SWITCH_COOLDOWN,
    LOAD_SHED_INCREMENT, GEN_REDISPATCH_INCREMENT,
    VOLTAGE_NOMINAL, VOLTAGE_TOLERANCE,
    BASE_FAULT_PROBABILITY, HIGH_LOAD_FAULT_MULTIPLIER,
    compute_dc_power_flow, compute_voltage_approximation, get_connected_buses,
)

NUM_LOAD_BUSES = len(LOAD_BUSES_ALL)
NUM_ACTIONS = 1 + NUM_LOAD_BUSES + NUM_LOAD_BUSES + NUM_LINES + NUM_GENERATORS * 2
OBS_DIM = NUM_LINES * 3 + NUM_BUSES * 3 + NUM_GENERATORS * 2 + 5

MAX_STEPS = 1000
BLACKOUT_THRESHOLD = 0.30
MAX_DISCONNECTED_LINES = 8


def get_action_name(action):
    if action == 0:
        return "do_nothing", "idle"
    elif 1 <= action <= NUM_LOAD_BUSES:
        idx = action - 1
        bus = LOAD_BUSES_ALL[idx]
        return f"shed_load_{BUS_NAMES[bus]}", "load_shedding"
    elif NUM_LOAD_BUSES + 1 <= action <= 2 * NUM_LOAD_BUSES:
        idx = action - NUM_LOAD_BUSES - 1
        bus = LOAD_BUSES_ALL[idx]
        return f"restore_load_{BUS_NAMES[bus]}", "load_restore"
    elif 2 * NUM_LOAD_BUSES + 1 <= action <= 2 * NUM_LOAD_BUSES + NUM_LINES:
        idx = action - 2 * NUM_LOAD_BUSES - 1
        return f"toggle_{LINE_NAMES[idx]}", "line_switching"
    elif 2 * NUM_LOAD_BUSES + NUM_LINES + 1 <= action <= 2 * NUM_LOAD_BUSES + NUM_LINES + NUM_GENERATORS:
        idx = action - 2 * NUM_LOAD_BUSES - NUM_LINES - 1
        return f"increase_gen_{GENERATOR_NAMES[idx]}", "gen_redispatch"
    else:
        idx = action - 2 * NUM_LOAD_BUSES - NUM_LINES - NUM_GENERATORS - 1
        return f"decrease_gen_{GENERATOR_NAMES[idx]}", "gen_redispatch"


class PowerGridEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 10}

    def __init__(self, render_mode=None, seed=None, fault_probability=None,
                 max_disconnected_lines=None, fault_recovery_steps=None,
                 survival_bonus=0.0):
        super().__init__()
        self.render_mode = render_mode

        self._fault_probability = fault_probability or BASE_FAULT_PROBABILITY
        self._max_disconnected = max_disconnected_lines or MAX_DISCONNECTED_LINES
        self._fault_recovery_steps = fault_recovery_steps or 0
        self._survival_bonus = survival_bonus

        self.action_space = spaces.Discrete(NUM_ACTIONS)
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(OBS_DIM,), dtype=np.float32
        )

        self._renderer = None
        self._step_count = 0
        self._rng = np.random.default_rng(seed)

        self.line_status = np.ones(NUM_LINES, dtype=np.int32)
        self.line_loading = np.zeros(NUM_LINES, dtype=np.float32)
        self.overload_timer = np.zeros(NUM_LINES, dtype=np.int32)
        self.line_cooldown = np.zeros(NUM_LINES, dtype=np.int32)
        self.fault_recovery_timer = np.zeros(NUM_LINES, dtype=np.int32)

        self.gen_output = GENERATOR_INIT_OUTPUT.copy()
        self.gen_available = GENERATOR_INIT_AVAILABLE.copy()

        self.load_shed_fraction = np.zeros(NUM_BUSES, dtype=np.float32)
        self.bus_injections = np.zeros(NUM_BUSES, dtype=np.float32)
        self.voltages = np.ones(NUM_BUSES, dtype=np.float32)

        self.time_of_day = 0.0
        self.cumulative_reward = 0.0

        self._last_action_name = "none"
        self._last_action_category = "none"
        self._cascade_events = []

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self._rng = np.random.default_rng(seed)

        self._step_count = 0
        self.cumulative_reward = 0.0

        self.line_status = np.ones(NUM_LINES, dtype=np.int32)
        self.line_loading = np.zeros(NUM_LINES, dtype=np.float32)
        self.overload_timer = np.zeros(NUM_LINES, dtype=np.int32)
        self.line_cooldown = np.zeros(NUM_LINES, dtype=np.int32)
        self.fault_recovery_timer = np.zeros(NUM_LINES, dtype=np.int32)

        self.gen_output = GENERATOR_INIT_OUTPUT.copy()
        self.gen_available = GENERATOR_INIT_AVAILABLE.copy()

        self.load_shed_fraction = np.zeros(NUM_BUSES, dtype=np.float32)

        self.time_of_day = self._rng.uniform(0.0, 1.0)

        for i in range(NUM_GENERATORS):
            if self.gen_available[i]:
                noise = self._rng.uniform(-0.05, 0.05)
                self.gen_output[i] = np.clip(
                    self.gen_output[i] + noise,
                    GENERATOR_MIN_OUTPUT[i],
                    GENERATOR_MAX_OUTPUT[i]
                )

        self._update_power_flow()
        self._last_action_name = "none"
        self._last_action_category = "none"
        self._cascade_events = []

        obs = self._get_observation()
        info = self._get_info()
        return obs, info

    def step(self, action):
        assert self.action_space.contains(action), f"invalid action: {action}"

        self._step_count += 1
        self._cascade_events = []
        action_penalty = 0.0

        self.line_cooldown = np.maximum(0, self.line_cooldown - 1)
        self.time_of_day = (self.time_of_day + 1.0 / MAX_STEPS) % 1.0

        valid = self._execute_action(action)
        if not valid:
            action_penalty = 0.05

        self._update_load_demand()
        self._inject_faults()
        self._update_power_flow()
        self._handle_overloads()
        self._check_generator_islanding()

        reward = self._compute_reward(action, action_penalty)
        self.cumulative_reward += reward

        terminated, truncated = self._check_terminal()

        obs = self._get_observation()
        info = self._get_info()

        if self.render_mode == "human":
            self.render()

        return obs, reward, terminated, truncated, info

    def _execute_action(self, action):
        if action == 0:
            self._last_action_name, self._last_action_category = "do_nothing", "idle"
            return True

        name, category = get_action_name(action)
        self._last_action_name = name
        self._last_action_category = category

        if 1 <= action <= NUM_LOAD_BUSES:
            idx = action - 1
            bus = LOAD_BUSES_ALL[idx]
            if bus not in BASE_LOAD_DEMAND or BASE_LOAD_DEMAND[bus] <= 0:
                return False
            self.load_shed_fraction[bus] = min(1.0, self.load_shed_fraction[bus] + LOAD_SHED_INCREMENT)
            return True

        elif NUM_LOAD_BUSES + 1 <= action <= 2 * NUM_LOAD_BUSES:
            idx = action - NUM_LOAD_BUSES - 1
            bus = LOAD_BUSES_ALL[idx]
            if self.load_shed_fraction[bus] <= 0:
                return False
            self.load_shed_fraction[bus] = max(0.0, self.load_shed_fraction[bus] - LOAD_SHED_INCREMENT)
            return True

        elif 2 * NUM_LOAD_BUSES + 1 <= action <= 2 * NUM_LOAD_BUSES + NUM_LINES:
            line_idx = action - 2 * NUM_LOAD_BUSES - 1
            if self.line_cooldown[line_idx] > 0:
                return False
            self.line_status[line_idx] = 1 - self.line_status[line_idx]
            self.line_cooldown[line_idx] = LINE_SWITCH_COOLDOWN
            if self.line_status[line_idx] == 1:
                self.overload_timer[line_idx] = 0
            return True

        elif 2 * NUM_LOAD_BUSES + NUM_LINES + 1 <= action <= 2 * NUM_LOAD_BUSES + NUM_LINES + NUM_GENERATORS:
            gen_idx = action - 2 * NUM_LOAD_BUSES - NUM_LINES - 1
            if not self.gen_available[gen_idx]:
                return False
            new_output = self.gen_output[gen_idx] + GEN_REDISPATCH_INCREMENT * GENERATOR_MAX_OUTPUT[gen_idx]
            new_output = min(new_output, self.gen_output[gen_idx] + GENERATOR_RAMP_RATE[gen_idx])
            if new_output > GENERATOR_MAX_OUTPUT[gen_idx]:
                return False
            self.gen_output[gen_idx] = new_output
            return True

        else:
            gen_idx = action - 2 * NUM_LOAD_BUSES - NUM_LINES - NUM_GENERATORS - 1
            if not self.gen_available[gen_idx]:
                return False
            new_output = self.gen_output[gen_idx] - GEN_REDISPATCH_INCREMENT * GENERATOR_MAX_OUTPUT[gen_idx]
            new_output = max(new_output, self.gen_output[gen_idx] - GENERATOR_RAMP_RATE[gen_idx])
            if new_output < GENERATOR_MIN_OUTPUT[gen_idx]:
                return False
            self.gen_output[gen_idx] = new_output
            return True

    def _get_demand_multiplier(self):
        return max(0.4, 0.7 + 0.3 * np.sin(2 * np.pi * (self.time_of_day - 0.25)))

    def _update_load_demand(self):
        demand_multiplier = self._get_demand_multiplier()

        for bus in LOAD_BUSES_ALL:
            base = BASE_LOAD_DEMAND[bus]
            noise = self._rng.normal(0, 0.02)
            spike = 0.0
            if self._rng.random() < 0.01:
                spike = self._rng.uniform(0.05, 0.15)
            self.bus_injections[bus] = -(base * demand_multiplier + noise + spike) * (1.0 - self.load_shed_fraction[bus])

    def _inject_faults(self):
        for k in range(NUM_LINES):
            if self.line_status[k] == 0:
                if self._fault_recovery_steps > 0 and self.fault_recovery_timer[k] > 0:
                    self.fault_recovery_timer[k] -= 1
                    if self.fault_recovery_timer[k] == 0:
                        self.line_status[k] = 1
                        self.overload_timer[k] = 0
                        self._cascade_events.append(f"auto-recovery {LINE_NAMES[k]}")
                continue
            fault_prob = self._fault_probability
            if abs(self.line_loading[k]) > 0.8:
                fault_prob *= HIGH_LOAD_FAULT_MULTIPLIER
            if self._rng.random() < fault_prob:
                self.line_status[k] = 0
                self.overload_timer[k] = 0
                if self._fault_recovery_steps > 0:
                    self.fault_recovery_timer[k] = self._fault_recovery_steps
                self._cascade_events.append(f"fault on {LINE_NAMES[k]}")

    def _update_power_flow(self):
        self.bus_injections[:] = 0.0

        for i, bus in enumerate(GENERATOR_BUSES):
            if self.gen_available[i]:
                self.bus_injections[bus] += self.gen_output[i]

        demand_multiplier = self._get_demand_multiplier()
        for bus in LOAD_BUSES_ALL:
            base = BASE_LOAD_DEMAND[bus]
            self.bus_injections[bus] -= base * demand_multiplier * (1.0 - self.load_shed_fraction[bus])

        line_flows = compute_dc_power_flow(self.bus_injections, self.line_status)
        for k in range(NUM_LINES):
            if self.line_status[k] == 1 and THERMAL_LIMITS[k] > 0:
                self.line_loading[k] = abs(line_flows[k]) / THERMAL_LIMITS[k]
            else:
                self.line_loading[k] = 0.0

        self.voltages = compute_voltage_approximation(self.bus_injections, self.line_status)

    def _handle_overloads(self):
        cascade_occurred = True
        cascade_rounds = 0

        while cascade_occurred and cascade_rounds < 5:
            cascade_occurred = False
            cascade_rounds += 1

            for k in range(NUM_LINES):
                if self.line_status[k] == 0:
                    continue
                if self.line_loading[k] > 1.0:
                    self.overload_timer[k] += 1
                    if self.overload_timer[k] >= MAX_OVERLOAD_STEPS:
                        self.line_status[k] = 0
                        self.overload_timer[k] = 0
                        self._cascade_events.append(f"auto-disconnect {LINE_NAMES[k]} (thermal)")
                        cascade_occurred = True
                else:
                    self.overload_timer[k] = max(0, self.overload_timer[k] - 1)

            if cascade_occurred:
                self._update_power_flow()

    def _check_generator_islanding(self):
        connected = get_connected_buses(self.line_status)
        for i, bus in enumerate(GENERATOR_BUSES):
            if self.gen_available[i] and bus not in connected:
                self.gen_available[i] = 0
                self.gen_output[i] = 0.0
                self._cascade_events.append(f"generator {GENERATOR_NAMES[i]} tripped (islanded)")
            elif (not self.gen_available[i] and bus in connected
                  and self._fault_recovery_steps > 0
                  and GENERATOR_INIT_AVAILABLE[i] == 1):
                self.gen_available[i] = 1
                self.gen_output[i] = GENERATOR_MIN_OUTPUT[i]
                self._cascade_events.append(f"generator {GENERATOR_NAMES[i]} restarted")

    def _compute_reward(self, action, action_penalty):
        demand_multiplier = self._get_demand_multiplier()
        total_demand = sum(BASE_LOAD_DEMAND[bus] for bus in LOAD_BUSES_ALL) * demand_multiplier

        total_served = 0.0
        total_shed = 0.0
        for bus in LOAD_BUSES_ALL:
            base = BASE_LOAD_DEMAND[bus] * demand_multiplier
            total_served += base * (1.0 - self.load_shed_fraction[bus])
            total_shed += base * self.load_shed_fraction[bus]

        serve_ratio = total_served / total_demand if total_demand > 0 else 1.0
        shed_ratio = total_shed / total_demand if total_demand > 0 else 0.0

        overload_penalty = sum(max(0, self.line_loading[k] - 1.0) for k in range(NUM_LINES) if self.line_status[k])
        disconnect_ratio = np.sum(self.line_status == 0) / NUM_LINES

        voltage_penalty = 0.0
        for bus in range(NUM_BUSES):
            voltage_penalty += max(0, abs(self.voltages[bus] - VOLTAGE_NOMINAL) - VOLTAGE_TOLERANCE)

        cascade_penalty = 1.0 if any("auto-disconnect" in e for e in self._cascade_events) else 0.0

        return float(
            + 2.0 * serve_ratio
            - 1.0 * overload_penalty
            - 0.5 * shed_ratio
            - 0.3 * disconnect_ratio
            - 10.0 * cascade_penalty
            - 0.1 * (action != 0)
            - 0.5 * voltage_penalty
            - action_penalty
            + self._survival_bonus
        )

    def _check_terminal(self):
        demand_multiplier = self._get_demand_multiplier()
        total_demand = sum(BASE_LOAD_DEMAND[bus] for bus in LOAD_BUSES_ALL) * demand_multiplier

        total_served = 0.0
        for bus in LOAD_BUSES_ALL:
            total_served += BASE_LOAD_DEMAND[bus] * demand_multiplier * (1.0 - self.load_shed_fraction[bus])

        serve_ratio = total_served / total_demand if total_demand > 0 else 1.0
        num_disconnected = int(np.sum(self.line_status == 0))

        terminated = serve_ratio < BLACKOUT_THRESHOLD or num_disconnected >= self._max_disconnected
        truncated = self._step_count >= MAX_STEPS

        return terminated, truncated

    def _get_observation(self):
        obs = np.zeros(OBS_DIM, dtype=np.float32)
        idx = 0

        for k in range(NUM_LINES):
            obs[idx] = self.line_loading[k]
            obs[idx + 1] = float(self.line_status[k])
            obs[idx + 2] = float(self.overload_timer[k])
            idx += 3

        for b in range(NUM_BUSES):
            obs[idx] = self.voltages[b]
            obs[idx + 1] = self.bus_injections[b]
            obs[idx + 2] = float(self.load_shed_fraction[b] > 0)
            idx += 3

        for i in range(NUM_GENERATORS):
            obs[idx] = self.gen_output[i] / GENERATOR_MAX_OUTPUT[i] if GENERATOR_MAX_OUTPUT[i] > 0 else 0.0
            obs[idx + 1] = float(self.gen_available[i])
            idx += 2

        demand_multiplier = self._get_demand_multiplier()
        total_demand = sum(BASE_LOAD_DEMAND[bus] for bus in LOAD_BUSES_ALL) * demand_multiplier
        total_gen = sum(self.gen_output[i] for i in range(NUM_GENERATORS) if self.gen_available[i])

        obs[idx] = total_demand
        obs[idx + 1] = total_gen
        obs[idx + 2] = self.time_of_day
        obs[idx + 3] = float(np.sum(self.line_loading > 1.0))
        obs[idx + 4] = float(np.sum(self.line_status == 0))

        return obs

    def _get_info(self):
        demand_multiplier = self._get_demand_multiplier()
        total_demand = sum(BASE_LOAD_DEMAND[bus] for bus in LOAD_BUSES_ALL) * demand_multiplier

        total_served = 0.0
        for bus in LOAD_BUSES_ALL:
            total_served += BASE_LOAD_DEMAND[bus] * demand_multiplier * (1.0 - self.load_shed_fraction[bus])

        serve_pct = (total_served / total_demand * 100) if total_demand > 0 else 100.0

        overloaded_lines = [
            LINE_NAMES[k] for k in range(NUM_LINES)
            if self.line_status[k] == 1 and self.line_loading[k] > 1.0
        ]
        disconnected_lines = [
            LINE_NAMES[k] for k in range(NUM_LINES)
            if self.line_status[k] == 0
        ]

        return {
            "step": self._step_count,
            "action_name": self._last_action_name,
            "action_category": self._last_action_category,
            "load_served_pct": round(serve_pct, 1),
            "total_demand": round(total_demand, 3),
            "total_generation": round(sum(self.gen_output[i] for i in range(NUM_GENERATORS) if self.gen_available[i]), 3),
            "overloaded_lines": overloaded_lines,
            "disconnected_lines": disconnected_lines,
            "num_overloaded": len(overloaded_lines),
            "num_disconnected": len(disconnected_lines),
            "cascade_events": self._cascade_events.copy(),
            "time_of_day": round(self.time_of_day * 24, 1),
            "cumulative_reward": round(self.cumulative_reward, 2),
        }

    def render(self):
        if self.render_mode == "human":
            if self._renderer is None:
                from environment.rendering import GridRenderer
                self._renderer = GridRenderer(self)
            self._renderer.render(self)
            return None
        elif self.render_mode == "rgb_array":
            if self._renderer is None:
                from environment.rendering import GridRenderer
                self._renderer = GridRenderer(self, offscreen=True)
            return self._renderer.render_to_array(self)
        return None

    def close(self):
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None
