import numpy as np
import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from environment.custom_env import (
    PowerGridEnv, NUM_ACTIONS, OBS_DIM, NUM_LOAD_BUSES, get_action_name,
    MAX_STEPS, BLACKOUT_THRESHOLD, MAX_DISCONNECTED_LINES,
)
from environment.grid_topology import (
    NUM_BUSES, NUM_LINES, NUM_GENERATORS,
    LOAD_BUSES_ALL, BASE_LOAD_DEMAND, GENERATOR_BUSES,
    GENERATOR_MAX_OUTPUT, GENERATOR_MIN_OUTPUT,
    LINE_CONNECTIONS, LINE_NAMES, THERMAL_LIMITS,
    LINE_SWITCH_COOLDOWN, LOAD_SHED_INCREMENT,
    build_admittance_matrix, compute_dc_power_flow,
    compute_voltage_approximation, get_connected_buses,
)


@pytest.fixture
def env():
    e = PowerGridEnv()
    e.reset(seed=42)
    return e


class TestSpaces:
    def test_observation_shape(self, env):
        obs, _ = env.reset()
        assert obs.shape == (OBS_DIM,)
        assert obs.shape == (117,)

    def test_observation_dtype(self, env):
        obs, _ = env.reset()
        assert obs.dtype == np.float32

    def test_action_space_size(self, env):
        assert env.action_space.n == NUM_ACTIONS
        assert env.action_space.n == 53

    def test_observation_in_space(self, env):
        obs, _ = env.reset()
        assert env.observation_space.contains(obs)

    def test_all_actions_valid(self, env):
        for a in range(NUM_ACTIONS):
            assert env.action_space.contains(a)


class TestReset:
    def test_reset_returns_tuple(self, env):
        result = env.reset()
        assert len(result) == 2
        obs, info = result
        assert isinstance(info, dict)

    def test_reset_clears_state(self, env):
        for _ in range(10):
            env.step(env.action_space.sample())
        obs, info = env.reset()
        assert info["step"] == 0
        assert info["num_disconnected"] == 0

    def test_reset_seed_reproducibility(self):
        env1 = PowerGridEnv()
        env2 = PowerGridEnv()
        obs1, _ = env1.reset(seed=123)
        obs2, _ = env2.reset(seed=123)
        np.testing.assert_array_equal(obs1, obs2)

    def test_reset_different_seeds_differ(self):
        env1 = PowerGridEnv()
        env2 = PowerGridEnv()
        obs1, _ = env1.reset(seed=1)
        obs2, _ = env2.reset(seed=2)
        assert not np.array_equal(obs1, obs2)


class TestStep:
    def test_step_returns_five_tuple(self, env):
        result = env.step(0)
        assert len(result) == 5
        obs, reward, terminated, truncated, info = result
        assert isinstance(reward, float)
        assert isinstance(terminated, bool)
        assert isinstance(truncated, bool)

    def test_do_nothing_action(self, env):
        obs, reward, term, trunc, info = env.step(0)
        assert info["action_name"] == "do_nothing"
        assert info["action_category"] == "idle"

    def test_step_increments_count(self, env):
        env.step(0)
        env.step(0)
        _, _, _, _, info = env.step(0)
        assert info["step"] == 3

    def test_time_advances(self, env):
        _, info1 = env.reset(seed=42)
        t0 = info1["time_of_day"]
        for _ in range(100):
            env.step(0)
        _, _, _, _, info2 = env.step(0)
        assert info2["time_of_day"] != t0


class TestLoadShedding:
    def test_shed_load_reduces_served(self, env):
        _, info_before = env.reset(seed=42)
        pct_before = info_before["load_served_pct"]
        for _ in range(5):
            env.step(1)
        _, _, _, _, info_after = env.step(1)
        assert info_after["load_served_pct"] < pct_before

    def test_restore_load_after_shed(self, env):
        env.step(1)
        env.step(1)
        bus = LOAD_BUSES_ALL[0]
        assert env.load_shed_fraction[bus] == pytest.approx(2 * LOAD_SHED_INCREMENT)
        restore_action = NUM_LOAD_BUSES + 1
        env.step(restore_action)
        assert env.load_shed_fraction[bus] == pytest.approx(LOAD_SHED_INCREMENT)

    def test_shed_caps_at_100_percent(self, env):
        bus = LOAD_BUSES_ALL[0]
        for _ in range(10):
            env.step(1)
        assert env.load_shed_fraction[bus] == pytest.approx(1.0)

    def test_restore_without_shed_is_invalid(self, env):
        restore_action = NUM_LOAD_BUSES + 1
        obs_before = env._get_observation().copy()
        env.step(restore_action)
        # shed fraction should still be 0
        bus = LOAD_BUSES_ALL[0]
        assert env.load_shed_fraction[bus] == 0.0


class TestLineSwitching:
    def test_toggle_line_disconnects(self, env):
        line_action = 2 * NUM_LOAD_BUSES + 1
        assert env.line_status[0] == 1
        env.step(line_action)
        assert env.line_status[0] == 0

    def test_toggle_line_reconnects(self, env):
        line_action = 2 * NUM_LOAD_BUSES + 1
        env.step(line_action)
        assert env.line_status[0] == 0
        for _ in range(LINE_SWITCH_COOLDOWN):
            env.step(0)
        env.step(line_action)
        assert env.line_status[0] == 1

    def test_cooldown_prevents_immediate_retoggle(self, env):
        line_action = 2 * NUM_LOAD_BUSES + 1
        env.step(line_action)
        assert env.line_status[0] == 0
        env.step(line_action)
        assert env.line_status[0] == 0


class TestGeneratorRedispatch:
    def test_increase_generator(self, env):
        gen_increase_action = 2 * NUM_LOAD_BUSES + NUM_LINES + 1
        output_before = env.gen_output[0]
        env.step(gen_increase_action)
        assert env.gen_output[0] > output_before

    def test_decrease_generator(self, env):
        gen_decrease_action = 2 * NUM_LOAD_BUSES + NUM_LINES + NUM_GENERATORS + 1
        output_before = env.gen_output[0]
        env.step(gen_decrease_action)
        assert env.gen_output[0] < output_before

    def test_unavailable_generator_action_invalid(self, env):
        gen_idx = 4
        assert env.gen_available[gen_idx] == 0
        increase_action = 2 * NUM_LOAD_BUSES + NUM_LINES + 1 + gen_idx
        env.step(increase_action)


class TestCascadingFailure:
    def test_overload_timer_increments(self, env):
        env.line_loading[0] = 1.5
        env.line_status[0] = 1
        env._handle_overloads()
        assert env.overload_timer[0] >= 1

    def test_line_auto_disconnects_after_max_overload(self, env):
        env.overload_timer[5] = 2
        env.line_loading[5] = 2.0
        env.line_status[5] = 1
        env._handle_overloads()
        assert env.line_status[5] == 0


class TestTerminalConditions:
    def test_blackout_terminates(self, env):
        for bus in LOAD_BUSES_ALL:
            env.load_shed_fraction[bus] = 0.8
        _, _, terminated, _, _ = env.step(0)
        assert terminated

    def test_too_many_disconnected_lines_terminates(self, env):
        for k in range(MAX_DISCONNECTED_LINES):
            env.line_status[k] = 0
        _, _, terminated, _, _ = env.step(0)
        assert terminated

    def test_truncation_at_max_steps(self, env):
        env._step_count = MAX_STEPS - 1
        _, _, _, truncated, _ = env.step(0)
        assert truncated

    def test_normal_operation_does_not_terminate(self, env):
        _, _, terminated, truncated, _ = env.step(0)
        assert not terminated
        assert not truncated


class TestReward:
    def test_do_nothing_positive_reward(self, env):
        _, reward, _, _, _ = env.step(0)
        assert reward > 0

    def test_massive_shedding_reduces_reward(self, env):
        _, reward_before, _, _, _ = env.step(0)
        env.reset(seed=42)
        for bus in LOAD_BUSES_ALL:
            env.load_shed_fraction[bus] = 0.5
        _, reward_after, _, _, _ = env.step(0)
        assert reward_after < reward_before


class TestActionNames:
    def test_do_nothing_name(self):
        name, cat = get_action_name(0)
        assert name == "do_nothing"

    def test_shed_load_name(self):
        name, cat = get_action_name(1)
        assert "shed_load" in name
        assert cat == "load_shedding"

    def test_toggle_line_name(self):
        action = 2 * NUM_LOAD_BUSES + 1
        name, cat = get_action_name(action)
        assert "toggle" in name
        assert cat == "line_switching"

    def test_all_actions_have_names(self):
        for a in range(NUM_ACTIONS):
            name, cat = get_action_name(a)
            assert len(name) > 0
            assert len(cat) > 0


class TestPowerFlow:
    def test_admittance_matrix_symmetric(self):
        B = build_admittance_matrix()
        np.testing.assert_array_almost_equal(B, B.T)

    def test_admittance_matrix_diagonal_positive(self):
        B = build_admittance_matrix()
        for i in range(NUM_BUSES):
            assert B[i, i] >= 0

    def test_dc_power_flow_returns_correct_shape(self):
        injections = np.zeros(NUM_BUSES, dtype=np.float64)
        injections[0] = 1.0
        injections[3] = -0.5
        injections[7] = -0.5
        flows = compute_dc_power_flow(injections)
        assert flows.shape == (NUM_LINES,)

    def test_disconnected_lines_have_zero_flow(self):
        injections = np.zeros(NUM_BUSES, dtype=np.float64)
        injections[0] = 1.0
        injections[3] = -1.0
        line_status = np.ones(NUM_LINES, dtype=np.int32)
        line_status[0] = 0
        flows = compute_dc_power_flow(injections, line_status)
        assert flows[0] == 0.0

    def test_voltage_approximation_shape(self):
        injections = np.zeros(NUM_BUSES, dtype=np.float64)
        voltages = compute_voltage_approximation(injections)
        assert voltages.shape == (NUM_BUSES,)

    def test_connected_buses_all_connected_default(self):
        line_status = np.ones(NUM_LINES, dtype=np.int32)
        connected = get_connected_buses(line_status)
        assert len(connected) == NUM_BUSES


class TestInfo:
    def test_info_has_required_keys(self, env):
        _, info = env.reset()
        required = [
            "step", "action_name", "action_category", "load_served_pct",
            "total_demand", "total_generation", "overloaded_lines",
            "disconnected_lines", "num_overloaded", "num_disconnected",
            "cascade_events", "time_of_day", "cumulative_reward",
        ]
        for key in required:
            assert key in info

    def test_load_served_starts_at_100(self, env):
        _, info = env.reset()
        assert info["load_served_pct"] == pytest.approx(100.0, abs=1.0)
