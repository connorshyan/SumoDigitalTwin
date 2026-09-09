"""Configuration module for the Autonomous Sumo Robot Digital Twin.

Contains immutable dataclasses, competition parameters, sensor specifications,
physics constants, and standardized directory paths for Kit (1kg) and Mega (3kg) classes.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

# ==============================================================================
# 1. DIRECTORY PATH DEFINITIONS (Strict Sim-to-Real Schema)
# ==============================================================================
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
AGENT_DATA_DIR = DATA_DIR / "agent"
OBSERVER_DATA_DIR = DATA_DIR / "observer"
DT_DATA_DIR = DATA_DIR / "dt"

# Ensure all partitioned data directories exist
for directory in [DATA_DIR, RAW_DATA_DIR, PROCESSED_DATA_DIR, AGENT_DATA_DIR, OBSERVER_DATA_DIR, DT_DATA_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

# ==============================================================================
# 2. GLOBAL SIMULATION & PHYSICS CONSTANTS
# ==============================================================================
SIMULATION_DT_S: float = 0.05                # 50 ms simulation clock tick
DEFAULT_MATCH_MAX_TICKS: int = 3600          # 180.0 s (3 minutes) match duration limit per official RoboGames rules
COULOMB_FRICTION_MU: float = 0.60            # Friction coefficient for body collisions
SPECULAR_MAX_ANGLE_DEG: float = 75.0         # Max optical incidence angle for diffuse IR sensors (E3Z-D62)
SENSOR_MAX_RANGE_CM: float = 50.0            # Maximum operational range in cm
EDGE_LINE_THRESHOLD: float = 0.70            # Normalized threshold for white border detection
EDGE_SENSOR_STANDOFF_MIN_MM: float = 0.5     # Minimum downward sensing clearance in mm (wedge ground scrape)
EDGE_SENSOR_STANDOFF_MAX_MM: float = 10.0    # Maximum downward operational standoff range in mm
EDGE_SENSOR_STANDOFF_NOMINAL_MM: float = 2.5 # Nominal focal distance in mm (QRE1113)
EDGE_NOISE_STD: float = 0.02                 # Gaussian noise standard deviation for QRE1113 edge sensors
EDGE_GLITCH_PROB: float = 0.001              # Transient electrical noise probability
EDGE_RECOVERY_DURATION_TICKS: int = 8        # 400 ms recovery sequence (2 ticks reverse + 3 ticks pivot + 3 ticks forward center drive)

# Calibrated PWM command constants
PWM_MAX: int = 255
PWM_MIN: int = -255
PWM_SEARCH_FAST: int = 200
PWM_SEARCH_SLOW: int = 140
PWM_ATTACK_FULL: int = 255
PWM_ATTACK_DEFENSIVE: int = 220
PWM_RECOVERY_LEFT: int = -220
PWM_RECOVERY_RIGHT: int = -220
PWM_RECOVERY_REVERSE: int = -220
PWM_RECOVERY_PIVOT: int = 220
PWM_BAIT_REVERSE: int = -180
PWM_JUGGERNAUT: int = 255

# Starting Formations Registry for Tournament Openings
STARTING_FORMATIONS: Tuple[str, ...] = (
    "HEAD_ON",          # Collinear face-to-face (0° / 180°, centered)
    "ANGLED_INWARD",    # Angled 30° inward toward ring center
    "LATERAL_OFFSET",   # Shifted along Shikiri line (facing forward)
    "SIDE_START",       # 90° lateral sweep facing ring perimeter
    "RANDOM_MIX",       # Stochastic uniform sampling across formations per match
)

# ==============================================================================
# 3. SENSOR AND ROBOT DATACLASSES
# ==============================================================================
@dataclass(frozen=True)
class SensorSpec:
    """Specification for an opponent or edge sensor."""
    name: str
    angle_deg: float                         # Angle relative to robot forward heading (+X)
    offset_x_cm: float = 0.0                 # Sensor mounting offset X (forward) from centroid
    offset_y_cm: float = 0.0                 # Sensor mounting offset Y (left) from centroid
    max_range_cm: float = SENSOR_MAX_RANGE_CM
    specular_max_angle_deg: float = SPECULAR_MAX_ANGLE_DEG


@dataclass(frozen=True)
class RobotClassConfig:
    """Official tournament specifications for an autonomous sumo class."""
    class_name: str                          # "KIT_1KG" or "MEGA_3KG"
    dohyo_diameter_cm: float                 # Ring diameter
    border_width_cm: float                   # White border width
    dohyo_height_cm: float                   # Dohyo platform elevation
    shikiri_width_cm: float                  # Start line width
    shikiri_length_cm: float                 # Start line length
    shikiri_separation_cm: float              # Distance between the two start lines
    robot_width_cm: float                    # Robot lateral width (Y-axis)
    robot_length_cm: float                   # Robot longitudinal length (X-axis)
    wheelbase_cm: float                      # Distance between drive wheels (L)
    v_max_cms: float                         # Max linear velocity at PWM 255 (cm/s)
    opponent_sensor_angles_deg: Tuple[float, ...]
    opponent_sensor_names: Tuple[str, ...]
    edge_sensor_offsets_cm: Tuple[Tuple[float, float], ...]  # [(x_FL, y_FL), (x_FR, y_FR)]
    wheel_offset_x_cm: float = 0.0           # Longitudinal drive axle offset (rear-biased)
    edge_sensor_names: Tuple[str, ...] = ("IR_Edge_FL", "IR_Edge_FR")
    edge_threshold: float = EDGE_LINE_THRESHOLD
    max_ticks: int = DEFAULT_MATCH_MAX_TICKS
    dt: float = SIMULATION_DT_S
    coulomb_mu: float = COULOMB_FRICTION_MU

    @property
    def dohyo_radius_cm(self) -> float:
        return self.dohyo_diameter_cm / 2.0

    @property
    def inner_ring_radius_cm(self) -> float:
        """Boundary between black dohyo and white outer ring."""
        return self.dohyo_radius_cm - self.border_width_cm

    @property
    def center_zone_radius_cm(self) -> float:
        """Inner 50% radius defining ring center control."""
        return self.dohyo_radius_cm / 2.0

    @property
    def opponent_sensors(self) -> List[SensorSpec]:
        """Generate full SensorSpec list with realistic chassis mounting offsets.
        All optical sensors are mounted on a shared sensor bracket line set slightly behind the front bumper (80% of half_l)
        to account for wedge/bumper mechanical clearance.
        """
        sensors: List[SensorSpec] = []
        half_l = self.robot_length_cm / 2.0
        half_w = self.robot_width_cm / 2.0
        sensor_x = half_l * 0.80  # Shared X position set behind front bumper edge

        for name, angle in zip(self.opponent_sensor_names, self.opponent_sensor_angles_deg):
            off_x = sensor_x
            if "90" in name:
                # 90-degree flank sensors on the lateral outer sides, sharing the same X line
                off_y = half_w if angle > 0 else -half_w
            elif "22" in name:
                # Outer front angles (85% of half width)
                off_y = (half_w * 0.85) if angle > 0 else (-half_w * 0.85)
            elif "18" in name:
                # Wide front angle for Kit class (75% of half width)
                off_y = (half_w * 0.75) if angle > 0 else (-half_w * 0.75)
            elif "15" in name:
                # Intermediate front angle (45% of half width)
                off_y = (half_w * 0.45) if angle > 0 else (-half_w * 0.45)
            else:
                # Center front sensor F0
                off_y = 0.0

            sensors.append(
                SensorSpec(
                    name=name,
                    angle_deg=angle,
                    offset_x_cm=off_x,
                    offset_y_cm=off_y,
                    max_range_cm=SENSOR_MAX_RANGE_CM,
                    specular_max_angle_deg=SPECULAR_MAX_ANGLE_DEG,
                )
            )
        return sensors


# ==============================================================================
# 4. OFFICIAL TOURNAMENT CLASS CONFIGURATIONS
# ==============================================================================
KIT_1KG_CONFIG = RobotClassConfig(
    class_name="KIT_1KG",
    dohyo_diameter_cm=77.0,
    border_width_cm=2.5,
    dohyo_height_cm=2.5,
    shikiri_width_cm=1.0,
    shikiri_length_cm=10.0,
    shikiri_separation_cm=10.0,
    robot_width_cm=15.0,
    robot_length_cm=15.0,
    wheelbase_cm=13.0,
    wheel_offset_x_cm=-3.5,  # Rear-biased drive axle (x = -3.5 cm)
    v_max_cms=60.0,
    opponent_sensor_angles_deg=(90.0, 18.0, 0.0, -18.0, -90.0),
    opponent_sensor_names=("Opp_L90", "Opp_L18", "Opp_F0", "Opp_R18", "Opp_R90"),
    edge_sensor_offsets_cm=((7.5, 7.5), (7.5, -7.5)),  # FL: (+7.5, +7.5), FR: (+7.5, -7.5) Front Corners
    edge_sensor_names=("IR_Edge_FL", "IR_Edge_FR"),
)

MEGA_3KG_CONFIG = RobotClassConfig(
    class_name="MEGA_3KG",
    dohyo_diameter_cm=154.0,
    border_width_cm=5.0,
    dohyo_height_cm=5.0,
    shikiri_width_cm=2.0,
    shikiri_length_cm=20.0,
    shikiri_separation_cm=20.0,
    robot_width_cm=20.0,
    robot_length_cm=20.0,
    wheelbase_cm=18.0,
    wheel_offset_x_cm=-5.0,  # Rear-biased drive axle (x = -5.0 cm)
    v_max_cms=80.0,
    opponent_sensor_angles_deg=(90.0, 22.5, 15.0, 0.0, -15.0, -22.5, -90.0),
    opponent_sensor_names=("Opp_L90", "Opp_L22_5", "Opp_L15", "Opp_F0", "Opp_R15", "Opp_R22_5", "Opp_R90"),
    edge_sensor_offsets_cm=((10.0, 10.0), (10.0, -10.0)),  # FL: (+10.0, +10.0), FR: (+10.0, -10.0) Front Corners
    edge_sensor_names=("IR_Edge_FL", "IR_Edge_FR"),
)

CONFIG_REGISTRY: Dict[str, RobotClassConfig] = {
    "KIT_1KG": KIT_1KG_CONFIG,
    "MEGA_3KG": MEGA_3KG_CONFIG,
}
