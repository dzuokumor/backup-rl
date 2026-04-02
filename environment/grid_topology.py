import numpy as np


NUM_BUSES = 14
NUM_LINES = 20
NUM_GENERATORS = 5
NUM_LOADS = 11

BUS_NAMES = [
    "Egbin",
    "Akangba",
    "Ajah-TS",
    "Lekki",
    "Ibeju",
    "Omotosho",
    "Islands",
    "Apapa",
    "Aes-Barge",
    "Mushin",
    "Festac",
    "Ojo",
    "Orile",
    "Agbara",
]

GENERATOR_BUSES = [0, 1, 2, 5, 8]

GENERATOR_NAMES = [
    "Egbin Thermal",
    "Akangba Sync. Condenser",
    "Ajah Gas Turbine",
    "Omotosho Gas Plant",
    "AES Barge (Backup)",
]

GENERATOR_MAX_OUTPUT = np.array([3.32, 0.40, 0.40, 0.40, 0.25], dtype=np.float32)
GENERATOR_MIN_OUTPUT = np.array([0.10, 0.05, 0.05, 0.05, 0.00], dtype=np.float32)
GENERATOR_RAMP_RATE = np.array([0.10, 0.08, 0.08, 0.08, 0.05], dtype=np.float32)
GENERATOR_INIT_OUTPUT = np.array([1.50, 0.20, 0.20, 0.20, 0.00], dtype=np.float32)
GENERATOR_INIT_AVAILABLE = np.array([1, 1, 1, 1, 0], dtype=np.int32)

LOAD_BUSES = [3, 4, 6, 7, 9, 10, 11, 12, 13]
LOAD_BUSES_ALL = [1, 3, 4, 5, 6, 7, 9, 10, 11, 12, 13]

BASE_LOAD_DEMAND = {
    1:  0.217,
    3:  0.478,
    4:  0.076,
    5:  0.112,
    6:  0.295,
    7:  0.389,
    9:  0.350,
    10: 0.290,
    11: 0.175,
    12: 0.135,
    13: 0.149,
}

LOAD_PRIORITY = {
    1:  "A",
    3:  "A",
    4:  "C",
    5:  "B",
    6:  "A",
    7:  "A",
    9:  "B",
    10: "B",
    11: "D",
    12: "C",
    13: "D",
}

LINE_CONNECTIONS = [
    (0, 1),
    (0, 4),
    (1, 2),
    (1, 3),
    (1, 4),
    (2, 3),
    (3, 4),
    (3, 6),
    (3, 8),
    (4, 5),
    (5, 10),
    (5, 11),
    (5, 12),
    (6, 7),
    (6, 8),
    (8, 13),
    (9, 3),
    (9, 10),
    (10, 11),
    (12, 13),
]

LINE_NAMES = [
    "Egbin-Akangba 330kV",
    "Egbin-Ibeju 132kV",
    "Akangba-Ajah 132kV",
    "Akangba-Lekki 33kV",
    "Akangba-Ibeju 33kV",
    "Ajah-Lekki 33kV",
    "Lekki-Ibeju 33kV",
    "Lekki-Islands 33kV",
    "Lekki-AES Barge 33kV",
    "Ibeju-Omotosho 132kV",
    "Omotosho-Festac 33kV",
    "Omotosho-Ojo 33kV",
    "Omotosho-Orile 33kV",
    "Islands-Apapa 33kV",
    "Islands-AES Barge 33kV",
    "AES Barge-Agbara 33kV",
    "Mushin-Lekki 33kV",
    "Mushin-Festac 33kV",
    "Festac-Ojo 33kV",
    "Orile-Agbara 33kV",
]

LINE_REACTANCE = np.array([
    0.05917, 0.22304, 0.19797, 0.17632, 0.17388,
    0.17103, 0.04211, 0.20912, 0.55618, 0.25202,
    0.19890, 0.25581, 0.13027, 0.17615, 0.11001,
    0.20640, 0.11001, 0.08450, 0.20912, 0.19207,
], dtype=np.float32)

THERMAL_LIMITS = np.array([
    1.50, 0.60, 0.60, 0.50, 0.50,
    0.40, 0.30, 0.35, 0.20, 0.50,
    0.35, 0.30, 0.30, 0.35, 0.25,
    0.20, 0.35, 0.40, 0.30, 0.25,
], dtype=np.float32)

MAX_OVERLOAD_STEPS = 3
LINE_SWITCH_COOLDOWN = 5
LOAD_SHED_INCREMENT = 0.20
GEN_REDISPATCH_INCREMENT = 0.10

VOLTAGE_NOMINAL = 1.0
VOLTAGE_TOLERANCE = 0.05
VOLTAGE_MIN = 0.90
VOLTAGE_MAX = 1.10

BASE_FAULT_PROBABILITY = 0.005
HIGH_LOAD_FAULT_MULTIPLIER = 3.0

BUS_POSITIONS = {
    0:  (0.85, 0.70),
    1:  (0.45, 0.65),
    2:  (0.75, 0.40),
    3:  (0.65, 0.50),
    4:  (0.80, 0.55),
    5:  (0.15, 0.60),
    6:  (0.55, 0.30),
    7:  (0.40, 0.25),
    8:  (0.70, 0.25),
    9:  (0.50, 0.50),
    10: (0.30, 0.45),
    11: (0.20, 0.40),
    12: (0.35, 0.55),
    13: (0.10, 0.35),
}


def build_admittance_matrix(line_status=None):
    if line_status is None:
        line_status = np.ones(NUM_LINES, dtype=np.int32)

    B = np.zeros((NUM_BUSES, NUM_BUSES), dtype=np.float64)

    for k in range(NUM_LINES):
        if line_status[k] == 0:
            continue
        i, j = LINE_CONNECTIONS[k]
        b = 1.0 / LINE_REACTANCE[k]
        B[i, j] -= b
        B[j, i] -= b
        B[i, i] += b
        B[j, j] += b

    return B


def compute_dc_power_flow(bus_injections, line_status=None):
    if line_status is None:
        line_status = np.ones(NUM_LINES, dtype=np.int32)

    B = build_admittance_matrix(line_status)

    B_reduced = B[1:, 1:]
    P_reduced = bus_injections[1:]

    try:
        if np.linalg.matrix_rank(B_reduced) < B_reduced.shape[0]:
            theta_reduced = np.linalg.lstsq(B_reduced, P_reduced, rcond=None)[0]
        else:
            theta_reduced = np.linalg.solve(B_reduced, P_reduced)
    except np.linalg.LinAlgError:
        theta_reduced = np.zeros(NUM_BUSES - 1, dtype=np.float64)

    theta = np.zeros(NUM_BUSES, dtype=np.float64)
    theta[1:] = theta_reduced

    line_flows = np.zeros(NUM_LINES, dtype=np.float32)
    for k in range(NUM_LINES):
        if line_status[k] == 0:
            line_flows[k] = 0.0
            continue
        i, j = LINE_CONNECTIONS[k]
        line_flows[k] = (theta[i] - theta[j]) / LINE_REACTANCE[k]

    return line_flows


def compute_voltage_approximation(bus_injections, line_status=None):
    voltages = np.ones(NUM_BUSES, dtype=np.float32)

    for bus in range(NUM_BUSES):
        injection = bus_injections[bus]
        voltages[bus] += 0.02 * injection
        voltages[bus] = np.clip(voltages[bus], 0.80, 1.15)

    if line_status is not None:
        connected = get_connected_buses(line_status)
        for bus in range(NUM_BUSES):
            if bus not in connected:
                voltages[bus] = 0.85

    return voltages


def get_connected_buses(line_status):
    adjacency = {i: set() for i in range(NUM_BUSES)}
    for k in range(NUM_LINES):
        if line_status[k] == 1:
            i, j = LINE_CONNECTIONS[k]
            adjacency[i].add(j)
            adjacency[j].add(i)

    visited = set()
    stack = [0]
    while stack:
        node = stack.pop()
        if node in visited:
            continue
        visited.add(node)
        for neighbor in adjacency[node]:
            if neighbor not in visited:
                stack.append(neighbor)

    return visited
