"""Sumo Environment module.

Implements differential drive kinematics, polygon bounding boxes, photoelectric raycasting
with 75-degree specular loss (Omron E3Z-D62 standard), QRE1113 thermal noise edge sensing, Coulomb friction collisions,
and discrete-time match execution with synchronized observation-action logging.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from config.config import (
    EDGE_GLITCH_PROB,
    EDGE_NOISE_STD,
    KIT_1KG_CONFIG,
    PWM_MAX,
    PWM_MIN,
    SIMULATION_DT_S,
    RobotClassConfig,
)

__all__ = [
    "SumoBot",
    "SumoArena",
    "SumoEnvironment",
    "compute_starting_placement",
    "run_simulation_match",
]


# ==============================================================================
# 1. ROBOT PHYSICAL ENTITY
# ==============================================================================
class SumoBot:
    """Autonomous Sumo Robot with differential drive kinematics and physical sensors."""

    def __init__(
        self,
        bot_id: str,
        config: RobotClassConfig,
        x: float,
        y: float,
        heading_rad: float,
    ) -> None:
        self.bot_id = bot_id
        self.config = config
        self.x = float(x)
        self.y = float(y)
        self.heading_rad = float(heading_rad)
        self.v_cms = 0.0
        self.omega_rads = 0.0
        self.pwm_left = 0
        self.pwm_right = 0
        self.current_state = "SEARCH"

    def reset(self, x: float, y: float, heading_rad: float) -> None:
        self.x = float(x)
        self.y = float(y)
        self.heading_rad = float(heading_rad)
        self.v_cms = 0.0
        self.omega_rads = 0.0
        self.pwm_left = 0
        self.pwm_right = 0
        self.current_state = "SEARCH"

    def get_wheel_positions(self) -> Tuple[Tuple[float, float], Tuple[float, float]]:
        """Return global coordinates of (Left_Wheel, Right_Wheel) using rear axle offset."""
        half_wb = self.config.wheelbase_cm / 2.0
        axle_x = self.config.wheel_offset_x_cm
        cos_h = math.cos(self.heading_rad)
        sin_h = math.sin(self.heading_rad)

        # Left wheel (lx = axle_x, ly = +half_wb)
        wx_l = self.x + (axle_x * cos_h - half_wb * sin_h)
        wy_l = self.y + (axle_x * sin_h + half_wb * cos_h)

        # Right wheel (lx = axle_x, ly = -half_wb)
        wx_r = self.x + (axle_x * cos_h - (-half_wb) * sin_h)
        wy_r = self.y + (axle_x * sin_h + (-half_wb) * cos_h)

        return (wx_l, wy_l), (wx_r, wy_r)

    def get_wheel_traction(self) -> Tuple[float, float]:
        """Return (traction_left, traction_right) in [0.0, 1.0].
        Wheels that have exited past the elevated Dohyo radius lose ground contact and drop to 0.0 traction.
        """
        (wx_l, wy_l), (wx_r, wy_r) = self.get_wheel_positions()
        d_l = math.hypot(wx_l, wy_l)
        d_r = math.hypot(wx_r, wy_r)
        r_max = self.config.dohyo_radius_cm

        trac_l = 1.0 if d_l <= r_max else 0.0
        trac_r = 1.0 if d_r <= r_max else 0.0
        return trac_l, trac_r

    def update_kinematics(self, pwm_left: int, pwm_right: int, dt: float = SIMULATION_DT_S) -> None:
        """Step differential drive kinematics with physical wheel drop-off ground traction."""
        self.pwm_left = int(np.clip(pwm_left, PWM_MIN, PWM_MAX))
        self.pwm_right = int(np.clip(pwm_right, PWM_MIN, PWM_MAX))

        trac_l, trac_r = self.get_wheel_traction()

        # Effective ground-driven individual wheel velocities
        v_l = (self.pwm_left / 255.0) * self.config.v_max_cms * trac_l
        v_r = (self.pwm_right / 255.0) * self.config.v_max_cms * trac_r

        # Body linear and angular velocities
        self.v_cms = (v_r + v_l) / 2.0
        self.omega_rads = (v_r - v_l) / self.config.wheelbase_cm

        # Discrete-time kinematic integration
        self.heading_rad = (self.heading_rad + self.omega_rads * dt) % (2.0 * math.pi)
        self.x += self.v_cms * math.cos(self.heading_rad) * dt
        self.y += self.v_cms * math.sin(self.heading_rad) * dt

    @property
    def heading_deg(self) -> float:
        return math.degrees(self.heading_rad) % 360.0

    @property
    def distance_to_center(self) -> float:
        return math.hypot(self.x, self.y)

    def get_bounding_box_corners(self) -> List[Tuple[float, float]]:
        """Return 4 corners in global coordinates: [FL, FR, RR, RL]."""
        half_l = self.config.robot_length_cm / 2.0
        half_w = self.config.robot_width_cm / 2.0
        cos_h = math.cos(self.heading_rad)
        sin_h = math.sin(self.heading_rad)

        local_corners = [
            (half_l, half_w),   # Front-Left
            (half_l, -half_w),  # Front-Right
            (-half_l, -half_w), # Rear-Right
            (-half_l, half_w),  # Rear-Left
        ]

        global_corners = []
        for lx, ly in local_corners:
            gx = self.x + (lx * cos_h - ly * sin_h)
            gy = self.y + (lx * sin_h + ly * cos_h)
            global_corners.append((gx, gy))

        return global_corners

    def get_edge_sensor_positions(self) -> Dict[str, Tuple[float, float]]:
        """Compute global coordinates for downward-facing edge reflectance sensors."""
        cos_h = math.cos(self.heading_rad)
        sin_h = math.sin(self.heading_rad)
        positions = {}

        for name, (off_x, off_y) in zip(
            self.config.edge_sensor_names, self.config.edge_sensor_offsets_cm
        ):
            gx = self.x + (off_x * cos_h - off_y * sin_h)
            gy = self.y + (off_x * sin_h + off_y * cos_h)
            positions[name] = (gx, gy)

        return positions


# ==============================================================================
# 2. TOURNAMENT ARENA & SENSOR PHYSICS ENGINE
# ==============================================================================
class SumoArena:
    """Official Circular Dohyo with white border ring and Shikiri starting lines."""

    def __init__(self, config: RobotClassConfig) -> None:
        self.config = config

    def sample_edge_sensors(
        self, bot: SumoBot, rng: np.random.Generator
    ) -> Dict[str, float]:
        """Compute reflectance with thermal Gaussian noise and 1% transient glitch."""
        positions = bot.get_edge_sensor_positions()
        readings: Dict[str, float] = {}

        for name, (gx, gy) in positions.items():
            r = math.hypot(gx, gy)
            # Ground truth: 1.0 on white outer ring, 0.0 on black dohyo surface
            ground_truth = 1.0 if r >= self.config.inner_ring_radius_cm else 0.0

            # Thermal Gaussian noise N(0, 0.03)
            noise = rng.normal(0.0, EDGE_NOISE_STD)
            measured = ground_truth + noise

            # 1% transient electrical glitch returning uniform random float in [0.0, 1.0]
            if rng.uniform(0.0, 1.0) < EDGE_GLITCH_PROB:
                measured = rng.uniform(0.0, 1.0)

            readings[name] = float(np.clip(measured, 0.0, 1.0))

        return readings

    def cast_opponent_sensors(
        self, bot: SumoBot, opponent: SumoBot, rng: np.random.Generator
    ) -> Dict[str, float]:
        """2D multi-ray raycasting against opponent bounding box with 45-deg specular loss."""
        readings: Dict[str, float] = {}
        opp_corners = opponent.get_bounding_box_corners()

        opp_segments = [
            (opp_corners[0], opp_corners[1]),
            (opp_corners[1], opp_corners[2]),
            (opp_corners[2], opp_corners[3]),
            (opp_corners[3], opp_corners[0]),
        ]

        cos_bot = math.cos(bot.heading_rad)
        sin_bot = math.sin(bot.heading_rad)

        for sensor in self.config.opponent_sensors:
            gx_mount = bot.x + (sensor.offset_x_cm * cos_bot - sensor.offset_y_cm * sin_bot)
            gy_mount = bot.y + (sensor.offset_x_cm * sin_bot + sensor.offset_y_cm * cos_bot)

            ray_heading_rad = bot.heading_rad + math.radians(sensor.angle_deg)
            dir_x = math.cos(ray_heading_rad)
            dir_y = math.sin(ray_heading_rad)

            closest_distance = float("inf")
            valid_hit_detected = False

            for (p1_x, p1_y), (p2_x, p2_y) in opp_segments:
                hit_dist, normal_x, normal_y = self._ray_segment_intersection(
                    gx_mount, gy_mount, dir_x, dir_y, p1_x, p1_y, p2_x, p2_y
                )

                if hit_dist is not None and hit_dist <= sensor.max_range_cm:
                    dot = dir_x * normal_x + dir_y * normal_y
                    cos_incidence = np.clip(abs(dot), 0.0, 1.0)
                    incidence_angle_deg = math.degrees(math.acos(cos_incidence))

                    if incidence_angle_deg <= sensor.specular_max_angle_deg:
                        if hit_dist < closest_distance:
                            closest_distance = hit_dist
                            valid_hit_detected = True

            if valid_hit_detected:
                sigma = 0.01 + 0.02 * closest_distance
                noisy_distance = closest_distance + rng.normal(0.0, sigma)
                readings[sensor.name] = float(np.clip(noisy_distance, 0.0, sensor.max_range_cm))
            else:
                readings[sensor.name] = -1.0

        return readings

    @staticmethod
    def _ray_segment_intersection(
        ray_x: float,
        ray_y: float,
        dir_x: float,
        dir_y: float,
        p1_x: float,
        p1_y: float,
        p2_x: float,
        p2_y: float,
    ) -> Tuple[Optional[float], float, float]:
        """Compute parametric intersection of ray with line segment, returning (dist, norm_x, norm_y)."""
        seg_dx = p2_x - p1_x
        seg_dy = p2_y - p1_y

        denom = dir_x * seg_dy - dir_y * seg_dx
        if abs(denom) < 1e-8:
            return None, 0.0, 0.0

        diff_x = p1_x - ray_x
        diff_y = p1_y - ray_y

        t = (diff_x * seg_dy - diff_y * seg_dx) / denom
        u = (diff_x * dir_y - diff_y * dir_x) / denom

        if t >= 0.0 and 0.0 <= u <= 1.0:
            seg_len = math.hypot(seg_dx, seg_dy)
            if seg_len < 1e-8:
                return None, 0.0, 0.0
            norm_x = seg_dy / seg_len
            norm_y = -seg_dx / seg_len
            return t, norm_x, norm_y

        return None, 0.0, 0.0

    def resolve_collisions(
        self, bot_a: SumoBot, bot_b: SumoBot, rng: Optional[np.random.Generator] = None
    ) -> bool:
        """Rigid-body Oriented Bounding Box (OBB) Separating Axis Theorem (SAT) collision resolution
        with Coulomb forward thrust transfer, contact yaw torque, and strictly zero interpenetration.
        """
        # 1. SAT Collision Detection & Minimum Translation Vector (MTV)
        axes = [
            (math.cos(bot_a.heading_rad), math.sin(bot_a.heading_rad)),
            (-math.sin(bot_a.heading_rad), math.cos(bot_a.heading_rad)),
            (math.cos(bot_b.heading_rad), math.sin(bot_b.heading_rad)),
            (-math.sin(bot_b.heading_rad), math.cos(bot_b.heading_rad)),
        ]

        corners_a = bot_a.get_bounding_box_corners()
        corners_b = bot_b.get_bounding_box_corners()

        min_overlap = float("inf")
        collision_normal: Optional[Tuple[float, float]] = None

        dx = bot_b.x - bot_a.x
        dy = bot_b.y - bot_a.y

        for ax, ay in axes:
            l = math.hypot(ax, ay)
            if l < 1e-8:
                continue
            ax, ay = ax / l, ay / l

            proj_a = [cx * ax + cy * ay for cx, cy in corners_a]
            proj_b = [cx * ax + cy * ay for cx, cy in corners_b]

            min_a, max_a = min(proj_a), max(proj_a)
            min_b, max_b = min(proj_b), max(proj_b)

            if max_a < min_b or max_b < min_a:
                return False  # Separating axis found -> zero collision

            overlap = min(max_a, max_b) - max(min_a, min_b)
            if overlap < min_overlap:
                min_overlap = overlap
                # Ensure normal vector points from Bot A towards Bot B
                if (dx * ax + dy * ay) < 0:
                    ax, ay = -ax, -ay
                collision_normal = (ax, ay)

        if collision_normal is not None and min_overlap > 1e-6:
            nx, ny = collision_normal

            # 1. Positional geometric separation: resolve exact OBB overlap with a tiny 0.05cm buffer
            sep = (min_overlap / 2.0) + 0.05
            bot_a.x -= nx * sep
            bot_a.y -= ny * sep
            bot_b.x += nx * sep
            bot_b.y += ny * sep

            # 2. Drive thrust vector projection along collision normal with dynamic traction micro-slip
            v_ax = bot_a.v_cms * math.cos(bot_a.heading_rad)
            v_ay = bot_a.v_cms * math.sin(bot_a.heading_rad)
            v_bx = bot_b.v_cms * math.cos(bot_b.heading_rad)
            v_by = bot_b.v_cms * math.sin(bot_b.heading_rad)

            slip_a = rng.uniform(0.96, 1.04) if rng is not None else 1.0
            slip_b = rng.uniform(0.96, 1.04) if rng is not None else 1.0
            net_push_v = (v_ax * slip_a + v_bx * slip_b) * nx + (v_ay * slip_a + v_by * slip_b) * ny

            push_disp = net_push_v * self.config.dt * self.config.coulomb_mu * 0.5
            bot_a.x += nx * push_disp
            bot_a.y += ny * push_disp
            bot_b.x += nx * push_disp
            bot_b.y += ny * push_disp

            # 3. True Physical Contact Moment-Arm Normal Torque (Wedge & Bumper Squaring)
            # Compute the actual physical contact point between the two colliding OBBs
            proj_a = [c[0] * nx + c[1] * ny for c in corners_a]
            proj_b = [c[0] * (-nx) + c[1] * (-ny) for c in corners_b]
            contact_a = corners_a[int(np.argmax(proj_a))]
            contact_b = corners_b[int(np.argmax(proj_b))]
            c_px = (contact_a[0] + contact_b[0]) * 0.5
            c_py = (contact_a[1] + contact_b[1]) * 0.5

            r_ax = c_px - bot_a.x
            r_ay = c_py - bot_a.y
            r_bx = c_px - bot_b.x
            r_by = c_py - bot_b.y

            f_scale = max(abs(bot_a.v_cms), abs(bot_b.v_cms), 40.0)
            # Contact normal force on A from B is in -n direction: F_Ax = -nx * f_scale, F_Ay = -ny * f_scale
            # Contact normal force on B from A is in +n direction: F_Bx = +nx * f_scale, F_By = +ny * f_scale
            raw_tau_a = (r_ax * (-ny * f_scale) - r_ay * (-nx * f_scale)) / (self.config.robot_length_cm * 2.0)
            raw_tau_b = (r_bx * (+ny * f_scale) - r_by * (+nx * f_scale)) / (self.config.robot_length_cm * 2.0)

            # Tire ground traction limits maximum angular slip rate during high-power forward drive (max ~1.5 rad/s = 85 deg/s)
            max_contact_yaw_rate = 1.50  # rad/s
            tau_a = float(np.clip(raw_tau_a, -max_contact_yaw_rate, max_contact_yaw_rate))
            tau_b = float(np.clip(raw_tau_b, -max_contact_yaw_rate, max_contact_yaw_rate))

            noise_t = rng.normal(0.0, 0.05) if rng is not None else 0.0
            bot_a.heading_rad = (bot_a.heading_rad + (tau_a + noise_t) * self.config.dt) % (2.0 * math.pi)
            bot_b.heading_rad = (bot_b.heading_rad + (tau_b - noise_t) * self.config.dt) % (2.0 * math.pi)

            return True

        return False

    def is_ring_out(self, bot: SumoBot) -> bool:
        """Check if robot centroid has exited the dohyo diameter."""
        return bot.distance_to_center > self.config.dohyo_radius_cm


def compute_starting_placement(
    config: RobotClassConfig,
    bot_id: str,
    formation: str = "HEAD_ON",
    rng: Optional[np.random.Generator] = None,
) -> Tuple[float, float, float, str]:
    """Calculate realistic tournament (x, y, heading_rad, resolved_formation) strictly behind the Shikiri start line."""
    half_sep = config.shikiri_separation_cm / 2.0
    half_bot_l = config.robot_length_cm / 2.0
    half_bot_w = config.robot_width_cm / 2.0
    r_chassis = math.hypot(half_bot_l, half_bot_w)
    r_dohyo = config.dohyo_radius_cm

    # Base contact position touching the Shikiri line
    base_x = -(half_sep + half_bot_l) if bot_id == "Bot_A" else +(half_sep + half_bot_l)
    base_heading = 0.0 if bot_id == "Bot_A" else math.pi

    if formation == "RANDOM_MIX":
        canonical_options = ["HEAD_ON", "ANGLED_INWARD", "LATERAL_OFFSET", "SIDE_START"]
        if rng is not None:
            formation = str(rng.choice(canonical_options))
        else:
            formation = "HEAD_ON"

    # Max permissible setback depth in contestant's quadrant
    max_quadrant_depth = max(0.0, (r_dohyo - r_chassis - 1.5) - abs(base_x))
    max_y_shift = max(1.0, (config.shikiri_length_cm / 2.0) - (config.robot_width_cm / 4.0))

    if rng is not None:
        # Realistic variations based on tournament practices
        if formation == "HEAD_ON":
            # Direct charge: setback 0 to 4cm (or 0-8cm for mega) for motor runway acceleration
            depth_max = min(max_quadrant_depth, 4.0 if config.class_name == "KIT_1KG" else 8.0)
            depth_setback = rng.uniform(0.0, depth_max)
            x = -(abs(base_x) + depth_setback) if bot_id == "Bot_A" else +(abs(base_x) + depth_setback)
            y = rng.uniform(-1.0, 1.0)
            angle_jitter = math.radians(rng.uniform(-2.0, 2.0))
            heading_rad = base_heading + angle_jitter
        elif formation == "ANGLED_INWARD":
            # Angled strike: 25° to 35° inward angle with 0 to 3cm setback
            depth_max = min(max_quadrant_depth, 3.0 if config.class_name == "KIT_1KG" else 6.0)
            depth_setback = rng.uniform(0.0, depth_max)
            x = -(abs(base_x) + depth_setback) if bot_id == "Bot_A" else +(abs(base_x) + depth_setback)
            y = rng.uniform(-1.5, 1.5)
            angle_delta = math.radians(rng.uniform(25.0, 35.0))
            heading_rad = (base_heading - angle_delta) if bot_id == "Bot_A" else (base_heading + angle_delta)
        elif formation == "LATERAL_OFFSET":
            # Lateral shift along Shikiri line: 75% to 115% of max shift, 0.5 to 5cm setback
            depth_max = min(max_quadrant_depth, 4.0 if config.class_name == "KIT_1KG" else 8.0)
            depth_setback = rng.uniform(0.5, depth_max)
            x = -(abs(base_x) + depth_setback) if bot_id == "Bot_A" else +(abs(base_x) + depth_setback)
            y_sign = 1.0 if bot_id == "Bot_A" else -1.0
            y = y_sign * rng.uniform(max_y_shift * 0.75, max_y_shift * 1.15)
            angle_jitter = math.radians(rng.uniform(-3.0, 3.0))
            heading_rad = base_heading + angle_jitter
        elif formation == "SIDE_START":
            # Sideways placement facing ring perimeter: 90° ± 5°
            depth_max = min(max_quadrant_depth, 3.5 if config.class_name == "KIT_1KG" else 7.0)
            depth_setback = rng.uniform(0.5, depth_max)
            x = -(abs(base_x) + depth_setback) if bot_id == "Bot_A" else +(abs(base_x) + depth_setback)
            y_sign = 1.0 if bot_id == "Bot_A" else -1.0
            y = y_sign * rng.uniform(max_y_shift * 0.75, max_y_shift * 1.15)
            angle_jitter = math.radians(rng.uniform(-5.0, 5.0))
            heading_rad = (math.pi / 2.0 + angle_jitter) if bot_id == "Bot_A" else (-math.pi / 2.0 + angle_jitter)
        else:
            x = base_x
            y = 0.0
            heading_rad = base_heading
    else:
        # Deterministic nominal default
        if formation == "HEAD_ON":
            x, y, heading_rad = base_x, 0.0, base_heading
        elif formation == "ANGLED_INWARD":
            angle_delta = math.radians(30.0)
            x, y = base_x, 0.0
            heading_rad = (base_heading - angle_delta) if bot_id == "Bot_A" else (base_heading + angle_delta)
        elif formation == "LATERAL_OFFSET":
            x = base_x
            y = +max_y_shift if bot_id == "Bot_A" else -max_y_shift
            heading_rad = base_heading
        elif formation == "SIDE_START":
            x = base_x
            y = +max_y_shift if bot_id == "Bot_A" else -max_y_shift
            heading_rad = (math.pi / 2.0) if bot_id == "Bot_A" else (-math.pi / 2.0)
        else:
            x, y, heading_rad = base_x, 0.0, base_heading

    # Enforce strict boundary safety (must be inside dohyo radius with safety margin)
    dist_c = math.hypot(x, y)
    if dist_c > (r_dohyo - r_chassis - 0.5):
        scale = (r_dohyo - r_chassis - 0.5) / max(0.1, dist_c)
        x *= scale
        y *= scale

    # Enforce strict Shikiri line legality (must be behind line)
    if bot_id == "Bot_A" and x > -(half_sep + half_bot_l):
        x = -(half_sep + half_bot_l)
    elif bot_id == "Bot_B" and x < +(half_sep + half_bot_l):
        x = +(half_sep + half_bot_l)

    return float(x), float(y), float(heading_rad % (2.0 * math.pi)), str(formation)


# ==============================================================================
# 3. COMPLETE SUMO SIMULATION ENVIRONMENT
# ==============================================================================
class SumoEnvironment:
    """Deterministic, high-performance Sim-to-Real 2D Sumo Match Environment."""

    def __init__(
        self,
        config: RobotClassConfig = KIT_1KG_CONFIG,
        seed: int = 42,
        formation_a: str = "HEAD_ON",
        formation_b: str = "HEAD_ON",
    ) -> None:
        self.config = config
        self.arena = SumoArena(config)
        self.rng = np.random.default_rng(seed)
        self.seed = seed
        self.current_tick = 0
        self.match_id = "M_00001"
        self.match_status = "IN_PROGRESS"
        self.formation_a = formation_a
        self.formation_b = formation_b

        xa, ya, ha, res_a = compute_starting_placement(config, "Bot_A", formation_a, self.rng)
        xb, yb, hb, res_b = compute_starting_placement(config, "Bot_B", formation_b, self.rng)
        self.actual_formation_a = res_a
        self.actual_formation_b = res_b

        self.bot_a = SumoBot("Bot_A", config, xa, ya, ha)
        self.bot_b = SumoBot("Bot_B", config, xb, yb, hb)

    def reset(
        self,
        match_id: str = "M_00001",
        seed: Optional[int] = None,
        formation_a: Optional[str] = None,
        formation_b: Optional[str] = None,
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Reset match to initial Shikiri start lines with designated opening formations."""
        if seed is not None:
            self.seed = seed
            self.rng = np.random.default_rng(seed)

        if formation_a is not None:
            self.formation_a = formation_a
        if formation_b is not None:
            self.formation_b = formation_b

        self.match_id = match_id
        self.current_tick = 0
        self.match_status = "IN_PROGRESS"

        xa, ya, ha, res_a = compute_starting_placement(self.config, "Bot_A", self.formation_a, self.rng)
        xb, yb, hb, res_b = compute_starting_placement(self.config, "Bot_B", self.formation_b, self.rng)
        self.actual_formation_a = res_a
        self.actual_formation_b = res_b

        self.bot_a.reset(xa, ya, ha)
        self.bot_b.reset(xb, yb, hb)

        obs_a = self._get_observation(self.bot_a, self.bot_b)
        obs_b = self._get_observation(self.bot_b, self.bot_a)

        return obs_a, obs_b

    def _get_observation(self, bot: SumoBot, opponent: SumoBot) -> Dict[str, Any]:
        """Compile ground-truth and local sensor dictionary for a robot."""
        edge_readings = self.arena.sample_edge_sensors(bot, self.rng)
        opp_readings = self.arena.cast_opponent_sensors(bot, opponent, self.rng)

        target_visible = any(val > 0.0 for val in opp_readings.values())
        bot_form = self.actual_formation_a if bot.bot_id == "Bot_A" else self.actual_formation_b

        obs: Dict[str, Any] = {
            "Match_ID": self.match_id,
            "Tick": self.current_tick,
            "Timestamp_ms": int(self.current_tick * self.config.dt * 1000),
            "Bot_ID": bot.bot_id,
            "Weight_Class": self.config.class_name,
            "Starting_Formation": bot_form,
            "Pos_X": float(bot.x),
            "Pos_Y": float(bot.y),
            "Heading_Deg": float(bot.heading_deg),
            "Dist_To_Center": float(bot.distance_to_center),
            "Target_Visible": bool(target_visible),
            "Current_State": bot.current_state,
            "Action_PWM_Left": bot.pwm_left,
            "Action_PWM_Right": bot.pwm_right,
            "Match_Status": self.match_status,
        }
        obs.update(edge_readings)
        obs.update(opp_readings)
        return obs
        obs.update(edge_readings)
        obs.update(opp_readings)
        return obs

    def step(
        self,
        action_a: Tuple[int, int, str],
        action_b: Tuple[int, int, str],
    ) -> Tuple[Dict[str, Any], Dict[str, Any], bool, Dict[str, Any]]:
        """Advance simulation by one 50ms discrete tick."""
        self.current_tick += 1
        pwm_la, pwm_ra, state_a = action_a
        pwm_lb, pwm_rb, state_b = action_b

        self.bot_a.current_state = state_a
        self.bot_b.current_state = state_b

        # Update differential kinematics
        self.bot_a.update_kinematics(pwm_la, pwm_ra, self.config.dt)
        self.bot_b.update_kinematics(pwm_lb, pwm_rb, self.config.dt)

        # Resolve body collisions
        self.arena.resolve_collisions(self.bot_a, self.bot_b, self.rng)

        # Check ring-out terminal condition
        ring_out_a = self.arena.is_ring_out(self.bot_a)
        ring_out_b = self.arena.is_ring_out(self.bot_b)

        done = False
        if ring_out_a and ring_out_b:
            self.match_status = "DRAW"
            done = True
        elif ring_out_a:
            self.match_status = "BOT_B_WIN"
            done = True
        elif ring_out_b:
            self.match_status = "BOT_A_WIN"
            done = True
        elif self.current_tick >= self.config.max_ticks:
            self.match_status = "DRAW"
            done = True

        obs_a = self._get_observation(self.bot_a, self.bot_b)
        obs_b = self._get_observation(self.bot_b, self.bot_a)

        obs_a["Match_Status"] = self.match_status
        obs_b["Match_Status"] = self.match_status

        info = {
            "winner": "Bot_A" if self.match_status == "BOT_A_WIN" else ("Bot_B" if self.match_status == "BOT_B_WIN" else "None"),
            "total_ticks": self.current_tick,
            "duration_s": self.current_tick * self.config.dt,
        }

        return obs_a, obs_b, done, info


# ==============================================================================
# 4. SYNCHRONIZED BATCH SIMULATION RUNNER
# ==============================================================================
def run_simulation_match(
    env: SumoEnvironment,
    agent_a_policy: Any,
    agent_b_policy: Any,
    match_id: str,
    formation_a: str = "HEAD_ON",
    formation_b: str = "HEAD_ON",
) -> List[Dict[str, Any]]:
    """Execute match with strictly synchronized observation-action-state recording."""
    obs_a, obs_b = env.reset(match_id=match_id, formation_a=formation_a, formation_b=formation_b)
    agent_a_policy.reset()
    agent_b_policy.reset()

    strat_a = getattr(agent_a_policy, "strategy_name", type(agent_a_policy).__name__).upper()
    strat_b = getattr(agent_b_policy, "strategy_name", type(agent_b_policy).__name__).upper()

    telemetry: List[Dict[str, Any]] = []
    done = False

    while not done:
        action_a = agent_a_policy.compute_action(obs_a)
        action_b = agent_b_policy.compute_action(obs_b)

        # Synchronously record sensory observation paired with the resulting action & state
        rec_a = dict(obs_a)
        rec_a["Strategy_Profile"] = strat_a
        rec_a["Current_State"] = action_a[2]
        rec_a["Action_PWM_Left"] = action_a[0]
        rec_a["Action_PWM_Right"] = action_a[1]

        rec_b = dict(obs_b)
        rec_b["Strategy_Profile"] = strat_b
        rec_b["Current_State"] = action_b[2]
        rec_b["Action_PWM_Left"] = action_b[0]
        rec_b["Action_PWM_Right"] = action_b[1]

        telemetry.append(rec_a)
        telemetry.append(rec_b)

        obs_a, obs_b, done, _ = env.step(action_a, action_b)

    # Append terminal outcome update on final frame
    if telemetry:
        telemetry[-2]["Match_Status"] = env.match_status
        telemetry[-1]["Match_Status"] = env.match_status

    return telemetry
