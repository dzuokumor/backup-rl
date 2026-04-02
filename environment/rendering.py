import math
import time
import random as stdlib_random

import numpy as np

try:
    import pygame
    from pygame.locals import DOUBLEBUF, OPENGL, QUIT, KEYDOWN, K_ESCAPE
    from OpenGL.GL import *
    from OpenGL.GLU import *
    HAS_OPENGL = True
except ImportError:
    HAS_OPENGL = False

from environment.grid_topology import (
    NUM_BUSES, NUM_LINES, NUM_GENERATORS,
    BUS_NAMES, BUS_POSITIONS, GENERATOR_BUSES, GENERATOR_NAMES,
    GENERATOR_MAX_OUTPUT,
    LINE_CONNECTIONS, LINE_NAMES,
    LOAD_BUSES_ALL, BASE_LOAD_DEMAND,
    get_connected_buses,
)

WINDOW_WIDTH = 1400
WINDOW_HEIGHT = 900
PANEL_WIDTH = 320
VIEWPORT_WIDTH = WINDOW_WIDTH - PANEL_WIDTH

WORLD_SCALE = 14.0
CAM_HEIGHT = 18.0
CAM_ANGLE = 45.0
CAM_CENTER_X = 7.0
CAM_CENTER_Z = 5.0
CAM_LOOK_FROM_FRONT = True

WATER_Y = -0.02
GROUND_Y = 0.0

WINDOW_COLORS = [
    (1.0, 0.92, 0.65),
    (0.95, 0.85, 0.55),
    (0.9, 0.95, 1.0),
    (0.7, 0.85, 1.0),
    (1.0, 0.8, 0.5),
    (0.85, 0.75, 0.5),
]

BUILDING_SHADES = [
    (0.08, 0.08, 0.10),
    (0.10, 0.10, 0.12),
    (0.07, 0.07, 0.09),
    (0.09, 0.08, 0.10),
    (0.11, 0.10, 0.13),
    (0.06, 0.06, 0.08),
]


def _bus_world_pos(bus_id):
    x, y = BUS_POSITIONS[bus_id]
    return x * WORLD_SCALE, GROUND_Y, (1 - y) * WORLD_SCALE


class BuildingData:
    __slots__ = ['x', 'z', 'w', 'd', 'h', 'shade', 'win_color', 'win_brightness',
                 'windows', 'has_rooftop_light', 'rooftop_color', 'bus_id']

    def __init__(self, x, z, w, d, h, rng, bus_id):
        self.x = x
        self.z = z
        self.w = w
        self.d = d
        self.h = h
        self.bus_id = bus_id
        self.shade = BUILDING_SHADES[rng.integers(0, len(BUILDING_SHADES))]
        self.win_color = WINDOW_COLORS[rng.integers(0, len(WINDOW_COLORS))]
        self.win_brightness = rng.uniform(0.4, 1.0)
        self.has_rooftop_light = h > 0.6 and rng.random() < 0.4
        self.rooftop_color = (rng.uniform(0.8, 1.0), rng.uniform(0.1, 0.4), rng.uniform(0.0, 0.15))

        self.windows = []
        if h > 0.12:
            floor_h = 0.08
            win_w_unit = 0.06
            num_floors = max(1, int(h / floor_h))
            cols_front = max(1, int(w / win_w_unit))
            cols_side = max(1, int(d / win_w_unit))

            lit_prob = rng.uniform(0.3, 0.85)
            for floor in range(num_floors):
                wy = floor * floor_h + floor_h * 0.3
                if wy > h - 0.03:
                    break
                for col in range(cols_front):
                    if rng.random() < lit_prob:
                        wx = (col + 0.3) * w / (cols_front + 0.2)
                        ww = w / (cols_front + 0.2) * 0.45
                        wh = floor_h * 0.55
                        bright = rng.uniform(0.2, 1.0) * self.win_brightness
                        self.windows.append(('front', wx, wy, ww, wh, bright))
                for col in range(cols_side):
                    if rng.random() < lit_prob:
                        wz = (col + 0.3) * d / (cols_side + 0.2)
                        ww = d / (cols_side + 0.2) * 0.45
                        wh = floor_h * 0.55
                        bright = rng.uniform(0.15, 0.9) * self.win_brightness
                        self.windows.append(('side', wz, wy, ww, wh, bright))


class DistrictData:
    def __init__(self, bus_id, rng):
        wx, _, wz = _bus_world_pos(bus_id)
        self.bus_id = bus_id
        self.buildings = []
        self.road_quads = []

        if bus_id in LOAD_BUSES_ALL:
            demand = BASE_LOAD_DEMAND.get(bus_id, 0.1)
            density = 0.6 + demand * 3.0
            height_scale = 0.4 + demand * 2.5
            district_radius = 0.6 + demand * 1.2
        elif bus_id in GENERATOR_BUSES:
            density = 0.3
            height_scale = 0.25
            district_radius = 0.4
        else:
            density = 0.4
            height_scale = 0.4
            district_radius = 0.5

        if bus_id in [6, 7, 3]:
            height_scale *= 2.2
            density *= 1.4

        grid_cells = max(2, int(density * 2.5))
        cell_size = district_radius * 2 / grid_cells
        road_width = cell_size * 0.15
        start_x = wx - district_radius
        start_z = wz - district_radius

        for row in range(grid_cells + 1):
            rz = start_z + row * cell_size
            self.road_quads.append((start_x, rz - road_width * 0.5,
                                    start_x + district_radius * 2, rz + road_width * 0.5))
        for col in range(grid_cells + 1):
            rx = start_x + col * cell_size
            self.road_quads.append((rx - road_width * 0.5, start_z,
                                    rx + road_width * 0.5, start_z + district_radius * 2))

        for row in range(grid_cells):
            for col in range(grid_cells):
                if rng.random() < 0.1:
                    continue
                cx = start_x + col * cell_size + cell_size * 0.5
                cz = start_z + row * cell_size + cell_size * 0.5

                num_bldgs = rng.integers(1, 4)
                for _ in range(num_bldgs):
                    lot_margin = cell_size * 0.12
                    bw = rng.uniform(cell_size * 0.15, cell_size * 0.5 - lot_margin)
                    bd = rng.uniform(cell_size * 0.15, cell_size * 0.5 - lot_margin)
                    bh = rng.exponential(0.2 * height_scale) + 0.06
                    bh = min(bh, 3.0 * height_scale)
                    bx = cx + rng.uniform(-cell_size * 0.3, cell_size * 0.3 - bw)
                    bz = cz + rng.uniform(-cell_size * 0.3, cell_size * 0.3 - bd)
                    self.buildings.append(BuildingData(bx, bz, bw, bd, bh, rng, bus_id))


def _draw_building(bldg, dim):
    x, z, w, d, h = bldg.x, bldg.z, bldg.w, bldg.d, bldg.h
    sr, sg, sb = bldg.shade

    base_dim = max(dim, 0.15)

    glColor4f(sr * base_dim * 1.4, sg * base_dim * 1.4, sb * base_dim * 1.4, 1.0)
    glBegin(GL_QUADS)
    glVertex3f(x, h, z)
    glVertex3f(x + w, h, z)
    glVertex3f(x + w, h, z + d)
    glVertex3f(x, h, z + d)
    glEnd()

    faces = [
        ((x, 0, z), (x + w, 0, z), (x + w, h, z), (x, h, z), 0.7),
        ((x, 0, z + d), (x + w, 0, z + d), (x + w, h, z + d), (x, h, z + d), 0.65),
        ((x, 0, z), (x, 0, z + d), (x, h, z + d), (x, h, z), 0.5),
        ((x + w, 0, z), (x + w, 0, z + d), (x + w, h, z + d), (x + w, h, z), 0.55),
    ]
    for v0, v1, v2, v3, shade_mult in faces:
        f = shade_mult * base_dim
        glColor4f(sr * f, sg * f, sb * f, 1.0)
        glBegin(GL_QUADS)
        glVertex3f(*v0)
        glVertex3f(*v1)
        glVertex3f(*v2)
        glVertex3f(*v3)
        glEnd()

    if h > 0.08:
        edge_b = base_dim * 0.2
        glColor4f(edge_b, edge_b, edge_b + 0.01, 0.3)
        glLineWidth(1.0)
        glBegin(GL_LINE_LOOP)
        glVertex3f(x, h + 0.001, z)
        glVertex3f(x + w, h + 0.001, z)
        glVertex3f(x + w, h + 0.001, z + d)
        glVertex3f(x, h + 0.001, z + d)
        glEnd()

    if dim < 0.05:
        return

    wr, wg, wb = bldg.win_color
    for win in bldg.windows:
        face, u, wy, ww, wh, bright = win
        b = bright * dim
        if b < 0.02:
            continue

        glColor4f(wr * b, wg * b, wb * b, min(1.0, b * 1.2))
        if face == 'front':
            glBegin(GL_QUADS)
            glVertex3f(x + u, wy, z - 0.002)
            glVertex3f(x + u + ww, wy, z - 0.002)
            glVertex3f(x + u + ww, wy + wh, z - 0.002)
            glVertex3f(x + u, wy + wh, z - 0.002)
            glEnd()

            glBegin(GL_QUADS)
            glVertex3f(x + u, wy, z + d + 0.002)
            glVertex3f(x + u + ww, wy, z + d + 0.002)
            glVertex3f(x + u + ww, wy + wh, z + d + 0.002)
            glVertex3f(x + u, wy + wh, z + d + 0.002)
            glEnd()
        else:
            glBegin(GL_QUADS)
            glVertex3f(x + w + 0.002, wy, z + u)
            glVertex3f(x + w + 0.002, wy, z + u + ww)
            glVertex3f(x + w + 0.002, wy + wh, z + u + ww)
            glVertex3f(x + w + 0.002, wy + wh, z + u)
            glEnd()

            glBegin(GL_QUADS)
            glVertex3f(x - 0.002, wy, z + u)
            glVertex3f(x - 0.002, wy, z + u + ww)
            glVertex3f(x - 0.002, wy + wh, z + u + ww)
            glVertex3f(x - 0.002, wy + wh, z + u)
            glEnd()

    if bldg.has_rooftop_light and dim > 0.2:
        rc = bldg.rooftop_color
        rb = dim * 0.8
        glColor4f(rc[0] * rb, rc[1] * rb, rc[2] * rb, rb)
        glPointSize(3.0)
        glBegin(GL_POINTS)
        glVertex3f(x + w * 0.5, h + 0.02, z + d * 0.5)
        glEnd()


class GridRenderer:
    def __init__(self, env, offscreen=False):
        if not HAS_OPENGL:
            raise ImportError("PyOpenGL and pygame required")

        self.offscreen = offscreen
        self._frame = 0
        self._start_time = time.time()
        self._cascade_flash_timer = 0
        self._cascade_lines = []
        self._ripple_rings = []

        if not offscreen:
            pygame.init()
            pygame.display.set_caption("RL-for-PGM  |  Lagos/EKEDC power grid")
            self._screen = pygame.display.set_mode(
                (WINDOW_WIDTH, WINDOW_HEIGHT), DOUBLEBUF | OPENGL
            )
        else:
            pygame.init()
            self._screen = pygame.Surface((WINDOW_WIDTH, WINDOW_HEIGHT))

        self._font = pygame.font.SysFont("consolas", 13)
        self._font_small = pygame.font.SysFont("consolas", 10)
        self._font_title = pygame.font.SysFont("consolas", 15, bold=True)
        self._font_big = pygame.font.SysFont("consolas", 28, bold=True)

        self._build_world()
        self._setup_gl()

    def _setup_gl(self):
        glEnable(GL_DEPTH_TEST)
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        glEnable(GL_LINE_SMOOTH)
        glEnable(GL_POINT_SMOOTH)
        glHint(GL_LINE_SMOOTH_HINT, GL_NICEST)

        glEnable(GL_FOG)
        glFogi(GL_FOG_MODE, GL_EXP2)
        glFogfv(GL_FOG_COLOR, [0.005, 0.005, 0.015, 1.0])
        glFogf(GL_FOG_DENSITY, 0.018)

    def _build_world(self):
        rng = np.random.default_rng(42)
        self._districts = {}
        for b in range(NUM_BUSES):
            self._districts[b] = DistrictData(b, rng)

        self._lagoon_shoreline = []
        shore_z_base = WORLD_SCALE * 0.50
        for i in range(60):
            x = -3 + i * (WORLD_SCALE + 6) / 59
            z = shore_z_base + 0.4 * math.sin(x * 0.8) + rng.uniform(-0.15, 0.15)
            self._lagoon_shoreline.append((x, z))

        self._stars = []
        for _ in range(200):
            sx = rng.uniform(-5, WORLD_SCALE + 5)
            sy = rng.uniform(8, 25)
            sz = rng.uniform(-10, -2)
            brightness = rng.uniform(0.15, 0.7)
            twinkle = rng.uniform(0.5, 4.0)
            self._stars.append((sx, sy, sz, brightness, twinkle))

    def _set_3d_camera(self):
        glViewport(0, 0, VIEWPORT_WIDTH, WINDOW_HEIGHT)
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        gluPerspective(42.0, VIEWPORT_WIDTH / WINDOW_HEIGHT, 0.05, 80.0)
        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()

        angle_rad = math.radians(CAM_ANGLE)
        cam_y = CAM_HEIGHT * math.sin(angle_rad)
        cam_z_off = CAM_HEIGHT * math.cos(angle_rad)

        gluLookAt(
            CAM_CENTER_X, cam_y, CAM_CENTER_Z - cam_z_off,
            CAM_CENTER_X, 0.0, CAM_CENTER_Z + 3.0,
            0.0, 1.0, 0.0,
        )

    def _set_2d_overlay(self):
        glViewport(0, 0, WINDOW_WIDTH, WINDOW_HEIGHT)
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        gluOrtho2D(0, WINDOW_WIDTH, WINDOW_HEIGHT, 0)
        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()
        glDisable(GL_DEPTH_TEST)

    def render(self, env):
        for event in pygame.event.get():
            if event.type == QUIT:
                self.close()
                return
            if event.type == KEYDOWN and event.key == K_ESCAPE:
                self.close()
                return

        self._draw_frame(env)
        pygame.display.flip()
        pygame.time.wait(150)

    def render_to_array(self, env):
        self._draw_frame(env)
        buf = glReadPixels(0, 0, WINDOW_WIDTH, WINDOW_HEIGHT, GL_RGB, GL_UNSIGNED_BYTE)
        img = np.frombuffer(buf, dtype=np.uint8).reshape(WINDOW_HEIGHT, WINDOW_WIDTH, 3)
        return np.flipud(img)

    def _draw_frame(self, env):
        self._frame += 1
        t = time.time() - self._start_time

        self._update_cascade(env)

        glClearColor(0.005, 0.005, 0.018, 1.0)
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

        self._set_3d_camera()
        glEnable(GL_DEPTH_TEST)
        glEnable(GL_FOG)

        self._draw_sky(t)
        self._draw_ground(env)
        self._draw_lagoon(t, env)
        self._draw_roads(env)
        self._draw_buildings(env)
        self._draw_transmission_lines(env, t)
        self._draw_substations(env, t)
        self._draw_ripples(t)

        glDisable(GL_FOG)
        self._set_2d_overlay()
        self._draw_panel(env, t)

    def _draw_sky(self, t):
        glDisable(GL_FOG)

        moon_sky_x = WORLD_SCALE * 0.75
        moon_sky_y = 10.0
        moon_sky_z = WORLD_SCALE + 5
        moon_r = 0.6
        segments = 24
        glBegin(GL_TRIANGLE_FAN)
        glColor4f(0.75, 0.78, 0.85, 0.9)
        glVertex3f(moon_sky_x, moon_sky_y, moon_sky_z)
        for s in range(segments + 1):
            angle = 2 * math.pi * s / segments
            glColor4f(0.4, 0.42, 0.5, 0.0)
            glVertex3f(moon_sky_x + moon_r * math.cos(angle),
                       moon_sky_y + moon_r * math.sin(angle), moon_sky_z)
        glEnd()
        glBegin(GL_TRIANGLE_FAN)
        glColor4f(0.85, 0.87, 0.92, 1.0)
        glVertex3f(moon_sky_x, moon_sky_y, moon_sky_z)
        inner_r = moon_r * 0.35
        for s in range(segments + 1):
            angle = 2 * math.pi * s / segments
            glColor4f(0.65, 0.68, 0.75, 0.5)
            glVertex3f(moon_sky_x + inner_r * math.cos(angle),
                       moon_sky_y + inner_r * math.sin(angle), moon_sky_z)
        glEnd()

        for sx, sy, sz, brightness, twinkle in self._stars:
            b = brightness * (0.5 + 0.5 * math.sin(t * twinkle))
            glColor4f(0.8 * b, 0.85 * b, 1.0 * b, b)
            glPointSize(1.5)
            glBegin(GL_POINTS)
            glVertex3f(sx, sy, sz)
            glEnd()
        glEnable(GL_FOG)

    def _draw_ground(self, env):
        connected = get_connected_buses(env.line_status)

        glColor4f(0.012, 0.012, 0.016, 1.0)
        glBegin(GL_QUADS)
        glVertex3f(-3, GROUND_Y - 0.01, -3)
        glVertex3f(WORLD_SCALE + 3, GROUND_Y - 0.01, -3)
        glVertex3f(WORLD_SCALE + 3, GROUND_Y - 0.01, WORLD_SCALE + 3)
        glVertex3f(-3, GROUND_Y - 0.01, WORLD_SCALE + 3)
        glEnd()

        glDepthMask(GL_FALSE)
        for b in range(NUM_BUSES):
            wx, _, wz = _bus_world_pos(b)
            if b in LOAD_BUSES_ALL:
                demand = BASE_LOAD_DEMAND.get(b, 0.1)
                if b not in connected:
                    intensity = 0.005
                else:
                    intensity = (1 - env.load_shed_fraction[b]) * 0.08
                radius = 0.5 + demand * 1.5
            else:
                intensity = 0.03
                radius = 0.4

            if intensity < 0.003:
                continue

            segments = 20
            glBegin(GL_TRIANGLE_FAN)
            glColor4f(0.9 * intensity, 0.7 * intensity, 0.35 * intensity, intensity * 2)
            glVertex3f(wx, GROUND_Y + 0.001, wz)
            glColor4f(0, 0, 0, 0)
            for s in range(segments + 1):
                angle = 2 * math.pi * s / segments
                glVertex3f(wx + radius * math.cos(angle), GROUND_Y + 0.001,
                           wz + radius * math.sin(angle))
            glEnd()
        glDepthMask(GL_TRUE)

    def _draw_lagoon(self, t, env):
        connected = get_connected_buses(env.line_status)

        shore = self._lagoon_shoreline
        far_z = WORLD_SCALE + 3

        glBegin(GL_TRIANGLE_STRIP)
        for sx, sz in shore:
            glColor4f(0.06, 0.10, 0.20, 1.0)
            glVertex3f(sx, WATER_Y, sz)
            glColor4f(0.04, 0.07, 0.16, 1.0)
            glVertex3f(sx, WATER_Y, far_z)
        glEnd()

        moon_x = WORLD_SCALE * 0.7
        moon_z = (shore[0][1] + far_z) * 0.55
        moon_radius = 2.5
        shimmer = 0.85 + 0.15 * math.sin(t * 0.2)
        glDepthMask(GL_FALSE)
        segments = 20
        glBegin(GL_TRIANGLE_FAN)
        glColor4f(0.12 * shimmer, 0.14 * shimmer, 0.22 * shimmer, 0.35 * shimmer)
        glVertex3f(moon_x, WATER_Y + 0.002, moon_z)
        glColor4f(0, 0, 0, 0)
        for s in range(segments + 1):
            angle = 2 * math.pi * s / segments
            glVertex3f(moon_x + moon_radius * 0.6 * math.cos(angle), WATER_Y + 0.002,
                       moon_z + moon_radius * 1.4 * math.sin(angle))
        glEnd()

        for stripe in range(8):
            sx_off = (stripe - 3.5) * 0.12
            s_shimmer = 0.7 + 0.3 * math.sin(t * 0.4 + stripe * 0.8)
            bright = 0.08 * s_shimmer
            glColor4f(bright * 0.8, bright * 0.9, bright * 1.4, bright * 1.5)
            glBegin(GL_QUADS)
            glVertex3f(moon_x + sx_off - 0.03, WATER_Y + 0.003, shore[0][1] + 0.2)
            glVertex3f(moon_x + sx_off + 0.03, WATER_Y + 0.003, shore[0][1] + 0.2)
            glVertex3f(moon_x + sx_off * 1.3 + 0.05, WATER_Y + 0.003, far_z - 0.5)
            glVertex3f(moon_x + sx_off * 1.3 - 0.05, WATER_Y + 0.003, far_z - 0.5)
            glEnd()
        glDepthMask(GL_TRUE)

        glColor4f(0.10, 0.14, 0.22, 0.5)
        glLineWidth(1.5)
        glBegin(GL_LINE_STRIP)
        for sx, sz in shore:
            glVertex3f(sx, WATER_Y + 0.003, sz)
        glEnd()

        for i in range(22):
            z_base = shore[0][1] + 0.25 + i * 0.35
            if z_base > far_z:
                break
            fade = 1.0 - (z_base - shore[0][1]) / (far_z - shore[0][1])
            dist_to_moon = abs(z_base - moon_z) / moon_radius
            moon_boost = max(0, 1.0 - dist_to_moon * 0.4) * 0.04
            c = 0.07 + fade * 0.05 + moon_boost
            glColor4f(c * 0.5, c * 0.8, c * 1.5, 0.3 * fade + 0.15)
            glLineWidth(1.0)
            glBegin(GL_LINE_STRIP)
            for j in range(50):
                x = -3 + j * (WORLD_SCALE + 6) / 49
                wz = z_base + 0.1 * math.sin(x * 1.0 + t * 0.3 + i * 0.6)
                glVertex3f(x, WATER_Y + 0.001, wz)
            glEnd()

        glDepthMask(GL_FALSE)
        for b in range(NUM_BUSES):
            wx, _, wz = _bus_world_pos(b)
            if wz < shore[0][1] - 0.5:
                continue
            if b in LOAD_BUSES_ALL:
                if b not in connected:
                    intensity = 0.002
                else:
                    intensity = (1 - env.load_shed_fraction[b]) * 0.025
            else:
                intensity = 0.012

            if intensity < 0.001:
                continue

            ref_z = wz + 0.8
            ref_z = min(ref_z, far_z - 1.0)
            shimmer = 0.8 + 0.2 * math.sin(t * 0.5 + wx * 2)
            ri = intensity * shimmer * 1.5

            glBegin(GL_TRIANGLE_FAN)
            glColor4f(0.9 * ri, 0.7 * ri, 0.3 * ri, ri * 0.8)
            glVertex3f(wx, WATER_Y + 0.002, ref_z)
            glColor4f(0, 0, 0, 0)
            for s in range(16):
                angle = 2 * math.pi * s / 16
                stretch_x = 0.3
                stretch_z = 1.2
                glVertex3f(wx + stretch_x * math.cos(angle), WATER_Y + 0.002,
                           ref_z + stretch_z * math.sin(angle))
            glEnd()

            glBegin(GL_QUADS)
            stripe_w = 0.06
            stripe_ri = ri * 0.4
            glColor4f(0.9 * stripe_ri, 0.7 * stripe_ri, 0.3 * stripe_ri, stripe_ri)
            glVertex3f(wx - stripe_w, WATER_Y + 0.003, ref_z - 1.0)
            glVertex3f(wx + stripe_w, WATER_Y + 0.003, ref_z - 1.0)
            glColor4f(0, 0, 0, 0)
            glVertex3f(wx + stripe_w * 1.5, WATER_Y + 0.003, ref_z + 2.0)
            glVertex3f(wx - stripe_w * 1.5, WATER_Y + 0.003, ref_z + 2.0)
            glEnd()
        glDepthMask(GL_TRUE)

    def _draw_roads(self, env):
        connected = get_connected_buses(env.line_status)

        for b in range(NUM_BUSES):
            if b in LOAD_BUSES_ALL and b not in connected:
                dim = 0.08
            elif b in LOAD_BUSES_ALL:
                dim = 0.3 + 0.7 * (1 - env.load_shed_fraction[b])
            else:
                dim = 0.4

            district = self._districts[b]
            for rx1, rz1, rx2, rz2 in district.road_quads:
                c = 0.04 * dim
                glColor4f(c + 0.005, c + 0.003, c, 1.0)
                glBegin(GL_QUADS)
                glVertex3f(rx1, GROUND_Y + 0.001, rz1)
                glVertex3f(rx2, GROUND_Y + 0.001, rz1)
                glVertex3f(rx2, GROUND_Y + 0.001, rz2)
                glVertex3f(rx1, GROUND_Y + 0.001, rz2)
                glEnd()

            if dim > 0.2:
                for rx1, rz1, rx2, rz2 in district.road_quads:
                    is_horizontal = abs(rx2 - rx1) > abs(rz2 - rz1)
                    length = abs(rx2 - rx1) if is_horizontal else abs(rz2 - rz1)
                    num_lights = int(length / 0.35)
                    for li in range(num_lights):
                        frac = (li + 0.5) / max(1, num_lights)
                        if is_horizontal:
                            lx = rx1 + (rx2 - rx1) * frac
                            lz = (rz1 + rz2) * 0.5
                        else:
                            lx = (rx1 + rx2) * 0.5
                            lz = rz1 + (rz2 - rz1) * frac
                        lb = 0.15 * dim
                        glColor4f(lb, lb * 0.85, lb * 0.5, 0.7 * dim)
                        glPointSize(1.5)
                        glBegin(GL_POINTS)
                        glVertex3f(lx, 0.04, lz)
                        glEnd()

    def _draw_buildings(self, env):
        connected = get_connected_buses(env.line_status)

        for b in range(NUM_BUSES):
            if b in LOAD_BUSES_ALL:
                if b not in connected:
                    dim = 0.02
                else:
                    dim = 1.0 - env.load_shed_fraction[b]
            elif b in GENERATOR_BUSES:
                gen_idx = GENERATOR_BUSES.index(b)
                dim = 0.5 if env.gen_available[gen_idx] else 0.02
            else:
                dim = 0.4

            for bldg in self._districts[b].buildings:
                _draw_building(bldg, dim)

    def _draw_transmission_lines(self, env, t):
        line_y = 0.08

        for k in range(NUM_LINES):
            i, j = LINE_CONNECTIONS[k]
            x1, _, z1 = _bus_world_pos(i)
            x2, _, z2 = _bus_world_pos(j)

            if env.line_status[k] == 0:
                glColor4f(0.08, 0.05, 0.05, 0.25)
                glLineWidth(1.0)
                dx, dz = x2 - x1, z2 - z1
                length = math.sqrt(dx * dx + dz * dz)
                segs = max(1, int(length / 0.25))
                for s in range(segs):
                    t0 = s / segs
                    t1 = (s + 0.4) / segs
                    glBegin(GL_LINES)
                    glVertex3f(x1 + dx * t0, line_y, z1 + dz * t0)
                    glVertex3f(x1 + dx * t1, line_y, z1 + dz * t1)
                    glEnd()
                continue

            loading = env.line_loading[k]

            if loading > 1.0:
                pulse = 0.7 + 0.3 * math.sin(t * 8)
                r, g, b = 1.0 * pulse, 0.12 * pulse, 0.08 * pulse
                glow_w = 2.0
                core_w = 1.0
            elif loading > 0.8:
                r, g, b = 0.9, 0.7, 0.08
                glow_w = 1.5
                core_w = 0.8
            else:
                r, g, b = 0.08, 0.5, 0.2
                glow_w = 1.2
                core_w = 0.7

            if k in self._cascade_lines and self._cascade_flash_timer > 0:
                flash = self._cascade_flash_timer / 15.0
                r = r * (1 - flash) + 1.0 * flash
                g = g * (1 - flash) + 0.25 * flash
                b = b * (1 - flash) + 0.04 * flash

            glColor4f(r * 0.4, g * 0.4, b * 0.4, 0.15)
            glLineWidth(glow_w + 2)
            glBegin(GL_LINES)
            glVertex3f(x1, line_y, z1)
            glVertex3f(x2, line_y, z2)
            glEnd()

            glColor4f(r, g, b, 0.8)
            glLineWidth(core_w)
            glBegin(GL_LINES)
            glVertex3f(x1, line_y, z1)
            glVertex3f(x2, line_y, z2)
            glEnd()

            if loading > 0.3:
                dx, dz = x2 - x1, z2 - z1
                length = math.sqrt(dx * dx + dz * dz)
                if length > 0:
                    flow_pos = (t * 1.2 * loading) % length
                    fx = x1 + dx * flow_pos / length
                    fz = z1 + dz * flow_pos / length
                    glColor4f(r, g, b, 0.9)
                    glPointSize(3.0)
                    glBegin(GL_POINTS)
                    glVertex3f(fx, line_y + 0.005, fz)
                    glEnd()

    def _draw_substations(self, env, t):
        connected = get_connected_buses(env.line_status)

        for b in range(NUM_BUSES):
            wx, _, wz = _bus_world_pos(b)
            is_gen = b in GENERATOR_BUSES

            if b not in connected:
                color = (0.2, 0.2, 0.2)
                glow = 0.1
            elif env.load_shed_fraction[b] > 0:
                shed = env.load_shed_fraction[b]
                color = (0.9, 0.35 * (1 - shed), 0.04)
                glow = 0.6 + 0.4 * math.sin(t * 4)
            elif abs(env.voltages[b] - 1.0) > 0.08:
                color = (0.9, 0.06, 0.06)
                glow = 0.5 + 0.5 * math.sin(t * 6)
            elif abs(env.voltages[b] - 1.0) > 0.04:
                color = (0.85, 0.7, 0.08)
                glow = 0.7 + 0.3 * math.sin(t * 2)
            else:
                color = (0.08, 0.7, 0.3)
                glow = 0.8 + 0.2 * math.sin(t * 1.5 + b)

            size = 0.1 if is_gen else 0.06
            h = 0.18 if is_gen else 0.1

            cr, cg, cb = color
            f = glow * 0.6
            glColor4f(cr * f, cg * f, cb * f, 1.0)
            glBegin(GL_QUADS)
            glVertex3f(wx - size, h, wz - size)
            glVertex3f(wx + size, h, wz - size)
            glVertex3f(wx + size, h, wz + size)
            glVertex3f(wx - size, h, wz + size)
            glEnd()
            for v0, v1, v2, v3, sm in [
                ((wx - size, 0, wz - size), (wx + size, 0, wz - size),
                 (wx + size, h, wz - size), (wx - size, h, wz - size), 0.5),
                ((wx - size, 0, wz + size), (wx + size, 0, wz + size),
                 (wx + size, h, wz + size), (wx - size, h, wz + size), 0.45),
                ((wx - size, 0, wz - size), (wx - size, 0, wz + size),
                 (wx - size, h, wz + size), (wx - size, h, wz - size), 0.35),
                ((wx + size, 0, wz - size), (wx + size, 0, wz + size),
                 (wx + size, h, wz + size), (wx + size, h, wz - size), 0.4),
            ]:
                ff = glow * sm
                glColor4f(cr * ff, cg * ff, cb * ff, 1.0)
                glBegin(GL_QUADS)
                glVertex3f(*v0)
                glVertex3f(*v1)
                glVertex3f(*v2)
                glVertex3f(*v3)
                glEnd()

            glDepthMask(GL_FALSE)
            gc = glow * 0.06
            segments = 12
            glBegin(GL_TRIANGLE_FAN)
            glColor4f(cr * gc, cg * gc, cb * gc, glow * 0.3)
            glVertex3f(wx, GROUND_Y + 0.002, wz)
            glColor4f(0, 0, 0, 0)
            for s in range(segments + 1):
                angle = 2 * math.pi * s / segments
                glVertex3f(wx + 0.25 * math.cos(angle), GROUND_Y + 0.002,
                           wz + 0.25 * math.sin(angle))
            glEnd()
            glDepthMask(GL_TRUE)

            if is_gen:
                gen_idx = GENERATOR_BUSES.index(b)
                if env.gen_available[gen_idx]:
                    output_frac = env.gen_output[gen_idx] / max(0.01, GENERATOR_MAX_OUTPUT[gen_idx])
                    tower_h = 0.12 + output_frac * 0.3
                    tf = 0.6
                    glColor4f(0.12 * tf, 0.12 * tf, 0.14 * tf, 1.0)
                    glBegin(GL_QUADS)
                    for dx_off, dz_off in [(-0.02, -0.02), (0.02, -0.02), (0.02, 0.02), (-0.02, 0.02)]:
                        glVertex3f(wx + dx_off, 0, wz + size + 0.05 + dz_off)
                    glEnd()
                    glBegin(GL_QUADS)
                    for dx_off, dz_off in [(-0.02, -0.02), (0.02, -0.02), (0.02, 0.02), (-0.02, 0.02)]:
                        glVertex3f(wx + dx_off, tower_h, wz + size + 0.05 + dz_off)
                    glEnd()

                    tip_b = output_frac * glow
                    glColor4f(0.25 * tip_b, 0.5 * tip_b, 1.0 * tip_b, tip_b)
                    glPointSize(4.0)
                    glBegin(GL_POINTS)
                    glVertex3f(wx, tower_h + 0.015, wz + size + 0.05)
                    glEnd()

    def _update_cascade(self, env):
        if hasattr(env, '_cascade_events') and env._cascade_events:
            new_cascades = [e for e in env._cascade_events if "auto-disconnect" in e]
            if new_cascades:
                self._cascade_flash_timer = 15
                self._cascade_lines = []
                for e in new_cascades:
                    for k, name in enumerate(LINE_NAMES):
                        if name in e:
                            self._cascade_lines.append(k)
                            i, j = LINE_CONNECTIONS[k]
                            x1, _, z1 = _bus_world_pos(i)
                            x2, _, z2 = _bus_world_pos(j)
                            self._ripple_rings.append(
                                ((x1 + x2) / 2, (z1 + z2) / 2, time.time()))

        if self._cascade_flash_timer > 0:
            self._cascade_flash_timer -= 1

    def _draw_ripples(self, t):
        alive = []
        for rx, rz, start_t in self._ripple_rings:
            age = time.time() - start_t
            if age > 2.5:
                continue
            alive.append((rx, rz, start_t))
            progress = age / 2.5
            radius = 0.15 + progress * 1.8
            alpha = (1 - progress) * 0.5

            glColor4f(1.0, 0.3, 0.04, alpha)
            glLineWidth(1.5)
            glBegin(GL_LINE_LOOP)
            for s in range(20):
                angle = 2 * math.pi * s / 20
                glVertex3f(rx + radius * math.cos(angle), 0.01,
                           rz + radius * math.sin(angle))
            glEnd()
        self._ripple_rings = alive

    def _draw_panel(self, env, t):
        px = VIEWPORT_WIDTH

        glColor4f(0.03, 0.03, 0.06, 0.95)
        glBegin(GL_QUADS)
        glVertex2f(px, 0)
        glVertex2f(WINDOW_WIDTH, 0)
        glVertex2f(WINDOW_WIDTH, WINDOW_HEIGHT)
        glVertex2f(px, WINDOW_HEIGHT)
        glEnd()

        glColor4f(0.06, 0.1, 0.18, 0.5)
        glLineWidth(1.0)
        glBegin(GL_LINES)
        glVertex2f(px, 0)
        glVertex2f(px, WINDOW_HEIGHT)
        glEnd()

        info = env._get_info()
        y = 18
        sp = 20

        self._draw_text("EKEDC GRID MONITOR", px + 16, y, (0.45, 0.65, 1.0), title=True)
        y += sp + 4

        glColor4f(0.1, 0.12, 0.2, 0.4)
        glBegin(GL_QUADS)
        glVertex2f(px + 10, y - 2)
        glVertex2f(WINDOW_WIDTH - 10, y - 2)
        glVertex2f(WINDOW_WIDTH - 10, y + sp * 2 + 8)
        glVertex2f(px + 10, y + sp * 2 + 8)
        glEnd()

        self._draw_text(f"step {info['step']}/1000", px + 16, y, (0.65, 0.65, 0.75))
        y += sp
        hour = info['time_of_day']
        h_int = int(hour)
        m_int = int((hour - h_int) * 60)
        self._draw_text(f"time {h_int:02d}:{m_int:02d}", px + 16, y, (0.65, 0.65, 0.75))
        y += sp + 12

        served = info['load_served_pct']
        if served > 90:
            sc = (0.15, 0.85, 0.35)
        elif served > 70:
            sc = (0.9, 0.75, 0.12)
        elif served > 50:
            sc = (0.95, 0.45, 0.08)
        else:
            sc = (0.95, 0.12, 0.08)

        self._draw_text(f"{served:.1f}%", px + 16, y, sc, big=True)
        y += 32
        self._draw_text("load served", px + 16, y, (0.45, 0.45, 0.55), small=True)
        y += sp + 8

        bar_x = px + 16
        bar_w = PANEL_WIDTH - 32
        bar_h = 6
        glColor4f(0.08, 0.08, 0.12, 0.8)
        glBegin(GL_QUADS)
        glVertex2f(bar_x, y)
        glVertex2f(bar_x + bar_w, y)
        glVertex2f(bar_x + bar_w, y + bar_h)
        glVertex2f(bar_x, y + bar_h)
        glEnd()
        fill = served / 100.0
        glColor4f(*sc, 0.85)
        glBegin(GL_QUADS)
        glVertex2f(bar_x, y)
        glVertex2f(bar_x + bar_w * fill, y)
        glVertex2f(bar_x + bar_w * fill, y + bar_h)
        glVertex2f(bar_x, y + bar_h)
        glEnd()
        y += bar_h + 14

        stats = [
            ("demand", f"{info['total_demand']:.3f} pu", (0.55, 0.55, 0.65)),
            ("generation", f"{info['total_generation']:.3f} pu", (0.55, 0.55, 0.65)),
            ("overloaded", str(info['num_overloaded']),
             (0.95, 0.25, 0.15) if info['num_overloaded'] > 0 else (0.45, 0.45, 0.55)),
            ("disconnected", str(info['num_disconnected']),
             (0.95, 0.25, 0.15) if info['num_disconnected'] > 0 else (0.45, 0.45, 0.55)),
        ]
        for label, val, color in stats:
            self._draw_text(label, px + 16, y, (0.4, 0.4, 0.5), small=True)
            self._draw_text(val, px + PANEL_WIDTH - 16, y, color, small=True, right=True)
            y += sp - 2

        y += 8
        glColor4f(0.12, 0.12, 0.2, 0.3)
        glBegin(GL_LINES)
        glVertex2f(px + 16, y)
        glVertex2f(WINDOW_WIDTH - 16, y)
        glEnd()
        y += 10

        self._draw_text(f"reward  {info['cumulative_reward']:.2f}", px + 16, y, (0.55, 0.65, 0.85))
        y += sp

        action_text = info['action_name']
        if len(action_text) > 30:
            action_text = action_text[:28] + ".."
        self._draw_text(f"action  {action_text}", px + 16, y, (0.6, 0.6, 0.7), small=True)
        y += sp - 2
        self._draw_text(f"type    {info['action_category']}", px + 16, y, (0.45, 0.45, 0.55), small=True)
        y += sp + 8

        if info["cascade_events"]:
            glColor4f(0.25, 0.06, 0.04, 0.5)
            box_h = min(len(info["cascade_events"]), 4) * (sp - 4) + 24
            glBegin(GL_QUADS)
            glVertex2f(px + 10, y - 4)
            glVertex2f(WINDOW_WIDTH - 10, y - 4)
            glVertex2f(WINDOW_WIDTH - 10, y + box_h)
            glVertex2f(px + 10, y + box_h)
            glEnd()

            self._draw_text("cascade events", px + 16, y, (0.95, 0.35, 0.15))
            y += sp
            for event in info["cascade_events"][:4]:
                if len(event) > 35:
                    event = event[:33] + ".."
                self._draw_text(event, px + 20, y, (0.9, 0.45, 0.25), small=True)
                y += sp - 4
            y += 8

        if info["overloaded_lines"]:
            self._draw_text("overloaded", px + 16, y, (0.95, 0.25, 0.15), small=True)
            y += sp - 4
            for ln in info["overloaded_lines"][:3]:
                if len(ln) > 35:
                    ln = ln[:33] + ".."
                self._draw_text(ln, px + 20, y, (0.85, 0.35, 0.25), small=True)
                y += sp - 4
            y += 6

        y = WINDOW_HEIGHT - 105
        glColor4f(0.06, 0.06, 0.1, 0.6)
        glBegin(GL_QUADS)
        glVertex2f(px + 10, y - 4)
        glVertex2f(WINDOW_WIDTH - 10, y - 4)
        glVertex2f(WINDOW_WIDTH - 10, WINDOW_HEIGHT - 10)
        glVertex2f(px + 10, WINDOW_HEIGHT - 10)
        glEnd()

        self._draw_text("legend", px + 16, y, (0.45, 0.45, 0.55), small=True)
        y += sp - 2
        legend = [
            ((0.08, 0.7, 0.3), "normal"),
            ((0.85, 0.7, 0.08), "voltage warning"),
            ((0.9, 0.35, 0.04), "load shedding"),
            ((0.9, 0.06, 0.06), "critical"),
            ((0.2, 0.2, 0.2), "islanded"),
        ]
        for color, label in legend:
            glColor4f(*color, 0.9)
            glBegin(GL_QUADS)
            glVertex2f(px + 20, y + 2)
            glVertex2f(px + 28, y + 2)
            glVertex2f(px + 28, y + 10)
            glVertex2f(px + 20, y + 10)
            glEnd()
            self._draw_text(label, px + 36, y, (0.45, 0.45, 0.55), small=True)
            y += sp - 4

    def _draw_text(self, text, x, y, color, center=False, small=False,
                   title=False, big=False, right=False):
        if big:
            font = self._font_big
        elif title:
            font = self._font_title
        elif small:
            font = self._font_small
        else:
            font = self._font

        r = int(min(1.0, color[0]) * 255)
        g = int(min(1.0, color[1]) * 255)
        b = int(min(1.0, color[2]) * 255)
        surface = font.render(str(text), True, (r, g, b))
        text_data = pygame.image.tostring(surface, "RGBA", True)
        w, h = surface.get_size()

        if center:
            x -= w / 2
        elif right:
            x -= w

        glRasterPos2f(x, y + h)
        glDrawPixels(w, h, GL_RGBA, GL_UNSIGNED_BYTE, text_data)

    def close(self):
        pygame.quit()
