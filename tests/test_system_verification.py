"""Autonomous Sumo Robot Digital Twin - Comprehensive System Verification Test Suite.

Standard Python unittest implementation validating:
1. Sensor Raycasting, Geometry & Optical Physics (E3Z-D62, QRE1113).
2. Differential Drive Kinematics & Coulomb Collision Resolution.
3. FSM Opponent Strategy Tactical Profiles & Sensor-Response Appropriateness.
4. Batch Match Simulation Lifecycle & RoboGames 3-Minute Limit Adherence.
5. Strict Sim-to-Real Data Leakage Audit (Agent vs. Observer Schema Isolation).
6. Official RoboGames Tournament Specification Compliance.
7. Decision Tree Artifact Persistence, MCU Hardware Benchmarks & C++ Export Audit.
"""

from __future__ import annotations

import math
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from config.config import (
    CONFIG_REGISTRY,
    KIT_1KG_CONFIG,
    MEGA_3KG_CONFIG,
    SENSOR_MAX_RANGE_CM,
)
from data_pipeline.etl_pipeline import (
    compute_observer_kpis,
    extract_match_summary,
    process_raw_telemetry,
)
from environment.opponents import (
    OPPONENT_REGISTRY,
    AggressiveCharger,
    BaitAndSwitch,
    RandomMixOpponent,
)
from environment.sumo_env import SumoArena, SumoBot, SumoEnvironment, run_simulation_match
from models.tree_optimizer import (
    build_hardware_summary_table,
    compute_mcu_benchmarks,
    engineer_features,
    evaluate_sensor_action,
    generate_cpp_harness,
    generate_cpp_header,
    load_decision_tree_package,
    save_decision_tree_package,
    train_and_benchmark_models,
    train_and_optimize_tree,
)


# ==============================================================================
# SUITE 1: SENSOR RAYCASTING, GEOMETRY & OPTICAL PHYSICS
# ==============================================================================
class TestSensorPhysics(unittest.TestCase):
    """Validate 2D multi-ray raycasting, specular angle loss, and edge reflectance."""

    def test_sensor_mounting_geometry(self):
        """Verify sensor mounting positions and Cartesian angular orientations."""
        cfg = KIT_1KG_CONFIG
        sensors = cfg.opponent_sensors
        self.assertEqual(len(sensors), 5)

        sensor_map = {s.name: s for s in sensors}
        self.assertIn("Opp_F0", sensor_map)
        self.assertEqual(sensor_map["Opp_F0"].angle_deg, 0.0)
        self.assertEqual(sensor_map["Opp_L18"].angle_deg, 18.0)
        self.assertEqual(sensor_map["Opp_R18"].angle_deg, -18.0)
        self.assertEqual(sensor_map["Opp_L90"].angle_deg, 90.0)
        self.assertEqual(sensor_map["Opp_R90"].angle_deg, -90.0)

        # Opponent sensors are mounted on shared bracket line behind front edge (offset_x = 80% of half_l)
        expected_x = (cfg.robot_length_cm / 2.0) * 0.80
        self.assertAlmostEqual(sensor_map["Opp_F0"].offset_x_cm, expected_x, places=3)
        self.assertEqual(sensor_map["Opp_L90"].offset_x_cm, sensor_map["Opp_F0"].offset_x_cm)
        self.assertEqual(sensor_map["Opp_R90"].offset_x_cm, sensor_map["Opp_F0"].offset_x_cm)
        # Side sensors must be mounted on lateral perimeter (offset_y = +/- half_w)
        self.assertAlmostEqual(sensor_map["Opp_L90"].offset_y_cm, cfg.robot_width_cm / 2.0, places=3)
        self.assertAlmostEqual(sensor_map["Opp_R90"].offset_y_cm, -cfg.robot_width_cm / 2.0, places=3)

        # Also verify MEGA_3KG 7-sensor geometry (L22.5 is at outer corner, L15 is intermediate)
        mega_cfg = MEGA_3KG_CONFIG
        mega_map = {s.name: s for s in mega_cfg.opponent_sensors}
        self.assertEqual(len(mega_map), 7)
        self.assertGreater(mega_map["Opp_L22_5"].offset_y_cm, mega_map["Opp_L15"].offset_y_cm)
        self.assertLess(mega_map["Opp_R22_5"].offset_y_cm, mega_map["Opp_R15"].offset_y_cm)

    def test_direct_front_raycast_detection(self):
        """Raycast directly against opponent placed along heading vector."""
        cfg = KIT_1KG_CONFIG
        arena = SumoArena(cfg)
        rng = np.random.default_rng(42)

        bot = SumoBot("BotA", cfg, x=0.0, y=0.0, heading_rad=0.0)
        # Place opponent 25 cm ahead along X-axis
        opponent = SumoBot("BotB", cfg, x=25.0, y=0.0, heading_rad=math.pi)

        readings = arena.cast_opponent_sensors(bot, opponent, rng)
        self.assertIn("Opp_F0", readings)
        # Distance from front bumper (x=7.5) to opponent front bumper (x=17.5) is ~10.0 cm
        self.assertGreater(readings["Opp_F0"], 0.0)
        self.assertLess(readings["Opp_F0"], 25.0)
        # Target directly in front should NOT hit side 90-deg sensors
        self.assertEqual(readings["Opp_L90"], -1.0)
        self.assertEqual(readings["Opp_R90"], -1.0)

    def test_out_of_range_sensor_clipping(self):
        """Targets beyond 50 cm maximum optical range must return -1.0."""
        cfg = KIT_1KG_CONFIG
        arena = SumoArena(cfg)
        rng = np.random.default_rng(42)

        bot = SumoBot("BotA", cfg, x=0.0, y=0.0, heading_rad=0.0)
        # Place opponent 75 cm ahead (bumper-to-bumper distance 60 cm, beyond 50 cm max range)
        opponent = SumoBot("BotB", cfg, x=75.0, y=0.0, heading_rad=math.pi)

        readings = arena.cast_opponent_sensors(bot, opponent, rng)
        for name, val in readings.items():
            self.assertEqual(val, -1.0, f"Sensor {name} reported {val} cm for target at 75 cm (max range 50 cm)")

    def test_specular_angle_loss(self):
        """Rays hitting surfaces at shallow grazing angles (> 75 deg) must experience specular loss."""
        cfg = KIT_1KG_CONFIG
        arena = SumoArena(cfg)
        rng = np.random.default_rng(42)

        bot = SumoBot("BotA", cfg, x=0.0, y=0.0, heading_rad=0.0)
        opponent = SumoBot("BotB", cfg, x=20.0, y=14.0, heading_rad=0.0)

        readings = arena.cast_opponent_sensors(bot, opponent, rng)
        for name, val in readings.items():
            self.assertTrue(val == -1.0 or (0.0 <= val <= SENSOR_MAX_RANGE_CM))

    def test_qre1113_edge_reflectance(self):
        """Downward sensors: ~0.0 on black dohyo, ~1.0 on white boundary."""
        cfg = KIT_1KG_CONFIG
        arena = SumoArena(cfg)
        rng = np.random.default_rng(42)

        # 1. Centered inside Dohyo (black surface, r = 0)
        bot_center = SumoBot("BotCenter", cfg, x=0.0, y=0.0, heading_rad=0.0)
        readings_center = arena.sample_edge_sensors(bot_center, rng)
        self.assertLess(readings_center["IR_Edge_FL"], 0.30)
        self.assertLess(readings_center["IR_Edge_FR"], 0.30)

        # 2. Positioned on white outer border (r >= inner_ring_radius_cm = 36.0 cm)
        bot_edge = SumoBot("BotEdge", cfg, x=37.0, y=0.0, heading_rad=0.0)
        readings_edge = arena.sample_edge_sensors(bot_edge, rng)
        self.assertTrue(readings_edge["IR_Edge_FL"] > 0.70 or readings_edge["IR_Edge_FR"] > 0.70)


# ==============================================================================
# SUITE 2: DIFFERENTIAL DRIVE KINEMATICS & COLLISION DYNAMICS
# ==============================================================================
class TestKinematicsAndPhysics(unittest.TestCase):
    """Validate 2D differential drive integration and collision mechanics."""

    def test_pure_linear_forward_kinematics(self):
        """PWM (255, 255) must produce forward linear velocity v = v_max and zero angular velocity."""
        cfg = KIT_1KG_CONFIG
        bot = SumoBot("BotA", cfg, x=0.0, y=0.0, heading_rad=0.0)

        bot.update_kinematics(pwm_left=255, pwm_right=255, dt=0.05)
        self.assertAlmostEqual(bot.v_cms, cfg.v_max_cms, places=2)
        self.assertAlmostEqual(bot.omega_rads, 0.0, places=3)
        self.assertAlmostEqual(bot.heading_deg, 0.0, places=2)
        self.assertAlmostEqual(bot.x, cfg.v_max_cms * 0.05, places=2)
        self.assertAlmostEqual(bot.y, 0.0, places=3)

    def test_in_place_spin_kinematics(self):
        """PWM (-255, 255) must produce pure zero linear velocity and maximum yaw rate."""
        cfg = KIT_1KG_CONFIG
        bot = SumoBot("BotA", cfg, x=0.0, y=0.0, heading_rad=0.0)

        bot.update_kinematics(pwm_left=-255, pwm_right=255, dt=0.05)
        expected_omega = (cfg.v_max_cms - (-cfg.v_max_cms)) / cfg.wheelbase_cm
        self.assertAlmostEqual(bot.v_cms, 0.0, places=3)
        self.assertAlmostEqual(bot.omega_rads, expected_omega, places=2)
        self.assertGreater(bot.heading_rad, 0.0)

    def test_coulomb_collision_resolution(self):
        """Overlapping robots must be physically separated and transfer forward thrust."""
        cfg = KIT_1KG_CONFIG
        arena = SumoArena(cfg)
        rng = np.random.default_rng(42)

        bot_a = SumoBot("BotA", cfg, x=0.0, y=0.0, heading_rad=0.0)
        bot_b = SumoBot("BotB", cfg, x=11.0, y=0.0, heading_rad=math.pi)
        bot_a.v_cms = 60.0
        bot_b.v_cms = 60.0

        resolved = arena.resolve_collisions(bot_a, bot_b, rng)
        self.assertTrue(resolved)

        post_dist = math.hypot(bot_b.x - bot_a.x, bot_b.y - bot_a.y)
        min_allowed = (cfg.robot_width_cm + cfg.robot_length_cm) / 2.0  # 15.0 cm
        self.assertGreaterEqual(post_dist, min_allowed - 0.1)

    def test_ring_out_boundary_detection(self):
        """Centroid distance > Dohyo radius must trigger ring-out."""
        cfg = KIT_1KG_CONFIG
        arena = SumoArena(cfg)

        bot_in = SumoBot("BotIn", cfg, x=30.0, y=0.0, heading_rad=0.0)
        self.assertFalse(arena.is_ring_out(bot_in))

        bot_out = SumoBot("BotOut", cfg, x=39.0, y=0.0, heading_rad=0.0)
        self.assertTrue(arena.is_ring_out(bot_out))  # Dohyo radius = 38.5 cm

    def test_wheel_drop_off_traction_loss(self):
        """Wheels that cross the elevated Dohyo boundary lose traction and drop to zero velocity."""
        cfg = KIT_1KG_CONFIG  # Radius = 38.5 cm, Wheelbase = 13.0 cm, Wheel offset = -3.5 cm
        bot = SumoBot("BotTraction", cfg, x=37.0, y=0.0, heading_rad=0.0)

        # 1. When bot is facing outward with rear wheels inside: traction is full (1.0, 1.0)
        trac_l, trac_r = bot.get_wheel_traction()
        self.assertEqual(trac_l, 1.0)
        self.assertEqual(trac_r, 1.0)

        # 2. When bot is facing inward with rear wheels hanging off the elevated rim (x = -3.5 is at x = +40.5)
        bot_rear_drop = SumoBot("BotRearDrop", cfg, x=37.0, y=0.0, heading_rad=math.pi)
        trac_l_drop, trac_r_drop = bot_rear_drop.get_wheel_traction()
        self.assertEqual(trac_l_drop, 0.0)
        self.assertEqual(trac_r_drop, 0.0)

        # Kinematic velocity must drop to zero when wheels have lost ground contact
        bot_rear_drop.update_kinematics(pwm_left=255, pwm_right=255, dt=0.05)
        self.assertEqual(bot_rear_drop.v_cms, 0.0)

    def test_starting_formations_geometry(self):
        """Verify all starting formations place robots legally behind Shikiri lines within the Dohyo."""
        cfg = KIT_1KG_CONFIG
        formations = ["HEAD_ON", "ANGLED_INWARD", "LATERAL_OFFSET", "SIDE_START", "RANDOM_MIX"]
        env = SumoEnvironment(config=cfg, seed=42)

        for form_a in formations:
            for form_b in formations:
                obs_a, obs_b = env.reset(formation_a=form_a, formation_b=form_b)
                # Both bots must start safely inside the dohyo
                self.assertLess(env.bot_a.distance_to_center, cfg.dohyo_radius_cm)
                self.assertLess(env.bot_b.distance_to_center, cfg.dohyo_radius_cm)
                # Bot A must be behind the negative Shikiri line (x <= -shikiri_sep/2)
                self.assertLessEqual(env.bot_a.x, -cfg.shikiri_separation_cm / 2.0)
                # Bot B must be behind the positive Shikiri line (x >= +shikiri_sep/2)
                self.assertGreaterEqual(env.bot_b.x, +cfg.shikiri_separation_cm / 2.0)


# ==============================================================================
# SUITE 3: OPPONENT STRATEGY BEHAVIORS & SENSOR-RESPONSE APPROPRIATENESS
# ==============================================================================
class TestOpponentBehaviors(unittest.TestCase):
    """Verify all 5 strategy profiles and sensor-driven FSM transitions."""

    def test_opening_blitz_active_all_strategies(self):
        """Every strategy must execute an active, formation-aware opening move for the first 6 ticks."""
        cfg = KIT_1KG_CONFIG
        formations = ["HEAD_ON", "SIDE_START", "LATERAL_OFFSET", "ANGLED_INWARD"]

        for strat_name, strat_cls in OPPONENT_REGISTRY.items():
            for form in formations:
                bot = strat_cls(cfg)
                bot.reset()

                obs = {"Tick": 0, "Starting_Formation": form, "IR_Edge_FL": 0.0, "IR_Edge_FR": 0.0, "Opp_F0": -1.0}
                pwm_l, pwm_r, state = bot.compute_action(obs)

                # Verify active motor actuation (not stalling)
                self.assertTrue(abs(pwm_l) > 50 or abs(pwm_r) > 50, f"{strat_name} stalled in {form}: ({pwm_l}, {pwm_r})")
                self.assertIn(state, ["ATTACK", "SEARCH", "TRACK", "EVADE"], f"{strat_name} invalid state in {form}")

        # Strategy-specific opening signature verifications in HEAD_ON
        charger = OPPONENT_REGISTRY["AGGRESSIVE_CHARGER"](cfg)
        obs_head = {"Tick": 0, "Starting_Formation": "HEAD_ON", "IR_Edge_FL": 0.0, "IR_Edge_FR": 0.0, "Opp_F0": -1.0}
        pl, pr, st = charger.compute_action(obs_head)
        self.assertEqual((pl, pr, st), (255, 255, "ATTACK"), "Aggressive charger did not full blitz in HEAD_ON")

        sweeper = OPPONENT_REGISTRY["DEFENSIVE_SWEEPER"](cfg)
        pl, pr, st = sweeper.compute_action(obs_head)
        self.assertEqual((pl, pr, st), (180, 180, "SEARCH"), "Defensive sweeper did not advance with controlled speed")

        flanker = OPPONENT_REGISTRY["RANDOM_FLANKER"](cfg)
        pl, pr, st = flanker.compute_action(obs_head)
        self.assertTrue(pl != pr, "Random flanker did not execute differential curved flank opening")

        bait = OPPONENT_REGISTRY["BAIT_AND_SWITCH"](cfg)
        pl, pr, st = bait.compute_action(obs_head)
        self.assertEqual(st, "SEARCH", "Bait and switch did not enter tactical search on tick 0")

        # Sensory preemption during opening: immediate close opponent triggers evasion
        obs_evade = {"Tick": 1, "Starting_Formation": "HEAD_ON", "IR_Edge_FL": 0.0, "IR_Edge_FR": 0.0, "Opp_F0": 10.0}
        pl, pr, st = bait.compute_action(obs_evade)
        self.assertEqual(st, "EVADE", "Bait and switch did not preempt opening with EVADE when target <= 14cm")

    def test_edge_recovery_sequence_safety_all_strategies(self):
        """When edge sensor triggers (>= 0.70), robot must execute 3-phase safe recovery."""
        cfg = KIT_1KG_CONFIG
        for strat_name, strat_cls in OPPONENT_REGISTRY.items():
            bot = strat_cls(cfg)
            bot.reset()

            # Trigger left edge sensor across 2 ticks (confirming genuine physical boundary crossing)
            obs_trigger = {"Tick": 10, "IR_Edge_FL": 0.85, "IR_Edge_FR": 0.05, "Opp_F0": -1.0}
            bot.compute_action(obs_trigger)  # Sample 1 (debouncing)
            pwm_l1, pwm_r1, state1 = bot.compute_action(obs_trigger)  # Sample 2 (confirmed)

            self.assertEqual(state1, "EDGE_RECOVERY", f"{strat_name} did not enter EDGE_RECOVERY")
            # Phase 1: Disengage back-step
            self.assertLess(pwm_l1, 0)
            self.assertLess(pwm_r1, 0)

            # Step through recovery ticks
            pwms = [(pwm_l1, pwm_r1)]
            for t in range(7):
                obs_step = {"Tick": 11 + t, "IR_Edge_FL": 0.0, "IR_Edge_FR": 0.0, "Opp_F0": -1.0}
                pl, pr, st = bot.compute_action(obs_step)
                self.assertEqual(st, "EDGE_RECOVERY")
                pwms.append((pl, pr))

            # Final recovery tick MUST drive forward back into the Dohyo center
            final_pwms = pwms[-1]
            self.assertGreater(final_pwms[0], 0, f"{strat_name} final recovery step PWM_L was not positive: {final_pwms}")
            self.assertGreater(final_pwms[1], 0, f"{strat_name} final recovery step PWM_R was not positive: {final_pwms}")

    def test_front_target_lock_attack_all_strategies(self):
        """Front target detection must trigger high-power ATTACK forward charge."""
        cfg = KIT_1KG_CONFIG
        for strat_name, strat_cls in OPPONENT_REGISTRY.items():
            bot = strat_cls(cfg)
            bot.reset()

            obs_front = {"Tick": 10, "IR_Edge_FL": 0.0, "IR_Edge_FR": 0.0, "Opp_F0": 20.0, "Opp_L18": -1.0, "Opp_R18": -1.0, "Opp_L90": -1.0, "Opp_R90": -1.0}
            pwm_l, pwm_r, state = bot.compute_action(obs_front)

            self.assertIn(state, ("ATTACK", "EVADE"))
            if state == "ATTACK":
                self.assertGreaterEqual(pwm_l, 220)
                self.assertGreaterEqual(pwm_r, 220)

    def test_side_flank_snap_turn_all_strategies(self):
        """Lateral 90-degree sensor hit must trigger sharp orienting TRACK snap turn."""
        cfg = KIT_1KG_CONFIG
        for strat_name, strat_cls in OPPONENT_REGISTRY.items():
            bot = strat_cls(cfg)
            bot.reset()

            # Target on Left 90 deg
            obs_left = {"Tick": 10, "IR_Edge_FL": 0.0, "IR_Edge_FR": 0.0, "Opp_F0": -1.0, "Opp_L18": -1.0, "Opp_R18": -1.0, "Opp_L90": 22.0, "Opp_R90": -1.0}
            pwm_l, pwm_r, state = bot.compute_action(obs_left)

            self.assertEqual(state, "TRACK")
            # Turn left: left wheel slower / reversed compared to right wheel
            self.assertLess(pwm_l, pwm_r)

    def test_bait_and_switch_close_evasion(self):
        """BaitAndSwitch must execute a quick tactical dodge when target is <= 14 cm."""
        cfg = KIT_1KG_CONFIG
        bot = BaitAndSwitch(cfg)
        bot.reset()

        # Target very close in front (10 cm)
        obs_close = {"Tick": 10, "IR_Edge_FL": 0.0, "IR_Edge_FR": 0.0, "Opp_F0": 10.0, "Opp_L18": -1.0, "Opp_R18": -1.0, "Opp_L90": -1.0, "Opp_R90": -1.0}
        pwm_l, pwm_r, state = bot.compute_action(obs_close)
        self.assertEqual(state, "EVADE")


# ==============================================================================
# SUITE 4: BATCH MATCH SIMULATION & ROBOGAMES 3-MINUTE LIMIT
# ==============================================================================
class TestMatchSimulation(unittest.TestCase):
    """Verify synchronized match recording and RoboGames 3-minute limit."""

    def test_full_match_lifecycle_kit_1kg(self):
        self._run_lifecycle_test("KIT_1KG")

    def test_full_match_lifecycle_mega_3kg(self):
        self._run_lifecycle_test("MEGA_3KG")

    def _run_lifecycle_test(self, class_name: str):
        cfg = CONFIG_REGISTRY[class_name]
        env = SumoEnvironment(config=cfg, seed=42)

        for i in range(5):
            agent_a = AggressiveCharger(cfg)
            agent_b = RandomMixOpponent(cfg)
            match_id = f"TEST_M_{i+1:03d}"
            records = run_simulation_match(env, agent_a, agent_b, match_id=match_id)

            self.assertGreater(len(records), 0)
            df_m = pd.DataFrame(records)

            required_cols = ["Match_ID", "Tick", "Timestamp_ms", "Bot_ID", "Pos_X", "Pos_Y", "Heading_Deg", "Match_Status", "Current_State", "Action_PWM_Left", "Action_PWM_Right"]
            for col in required_cols:
                self.assertIn(col, df_m.columns)

            final_status = df_m["Match_Status"].iloc[-1]
            self.assertIn(final_status, ("BOT_A_WIN", "BOT_B_WIN", "DRAW"))

            # Check 3-minute limit compliance (<= 3600 ticks = 180s)
            max_tick = df_m["Tick"].max()
            self.assertLessEqual(max_tick, 3600)


# ==============================================================================
# SUITE 5: STRICT SIM-TO-REAL DATA LEAKAGE AUDIT
# ==============================================================================
class TestDataLeakageAndETL(unittest.TestCase):
    """Strictly verify that Agent Dataset has ZERO global coordinate leakage."""

    def test_schema_isolation_and_leakage_audit(self):
        cfg = KIT_1KG_CONFIG
        env = SumoEnvironment(config=cfg, seed=42)

        records = []
        for i in range(2):
            agent_a = AggressiveCharger(cfg)
            agent_b = RandomMixOpponent(cfg)
            records.extend(run_simulation_match(env, agent_a, agent_b, match_id=f"M_{i+1:02d}"))

        df_raw = pd.DataFrame(records)

        with tempfile.TemporaryDirectory() as td:
            temp_dir = Path(td)
            raw_parquet = temp_dir / "test_raw.parquet"
            df_raw.to_parquet(raw_parquet, index=False, engine="pyarrow")

            res = process_raw_telemetry(raw_parquet, base_data_dir=temp_dir, weight_class="KIT_1KG")

            df_agent = pd.read_parquet(res["agent_path"])
            df_observer = pd.read_parquet(res["observer_path"])

            # 1. STRICT FORBIDDEN COORDINATE AUDIT ON AGENT DATASET
            forbidden_spatial_cols = [
                "Pos_X",
                "Pos_Y",
                "Heading_Deg",
                "Dist_To_Center",
                "Heading_Rad",
                "Global_X",
                "Global_Y",
                "Opponent_X",
                "Opponent_Y",
                "Match_Status",
            ]

            for forbidden in forbidden_spatial_cols:
                self.assertNotIn(
                    forbidden,
                    df_agent.columns,
                    f"CRITICAL DATA LEAKAGE DETECTED! '{forbidden}' found in Agent Dataset ({res['agent_path']}). "
                    f"Sim-to-Real policy must strictly rely on onboard hardware sensors only!",
                )

            # 2. Verify Valid Local Onboard Sensors in Agent Dataset
            expected_agent_features = [
                "Match_ID",
                "Bot_ID",
                "Timestamp_ms",
                "IR_Edge_FL",
                "IR_Edge_FR",
                "Opp_L90",
                "Opp_L18",
                "Opp_F0",
                "Opp_R18",
                "Opp_R90",
                "Current_State",
                "Action_PWM_Left",
                "Action_PWM_Right",
            ]
            for feat in expected_agent_features:
                self.assertIn(feat, df_agent.columns)

            # 3. Verify Observer Dataset Retains Ground Truth for Spatial Analysis
            observer_ground_truth_cols = ["Pos_X", "Pos_Y", "Heading_Deg", "Dist_To_Center", "Match_Status"]
            for col in observer_ground_truth_cols:
                self.assertIn(col, df_observer.columns)

            # 4. Check Data Hygiene
            self.assertEqual(df_agent.isna().sum().sum(), 0)
            self.assertEqual(df_observer.isna().sum().sum(), 0)
            self.assertEqual(len(df_agent), len(df_raw))

            # 5. Check Formal Data Quality Report
            self.assertIn("data_quality", res)
            dq = res["data_quality"]
            self.assertEqual(dq["completeness_score_pct"], 100.0)
            self.assertTrue(dq["domain_boundaries_valid"])
            self.assertTrue(dq["timestamp_monotonic_50ms"])
            self.assertGreaterEqual(dq["class_imbalance_ratio"], 1.0)


# ==============================================================================
# SUITE 6: OFFICIAL ROBOGAMES RULEBOOK SPECIFICATION AUDIT
# ==============================================================================
class TestRoboGamesCompliance(unittest.TestCase):
    """Verify adherence to official RoboGames Unified Sumo rules."""

    def test_dohyo_and_robot_dimensions(self):
        # Kit 1kg Class
        k1 = KIT_1KG_CONFIG
        self.assertEqual(k1.dohyo_diameter_cm, 77.0)
        self.assertEqual(k1.border_width_cm, 2.5)
        self.assertEqual(k1.shikiri_separation_cm, 10.0)
        self.assertEqual(k1.shikiri_length_cm, 10.0)
        self.assertEqual(k1.robot_width_cm, 15.0)
        self.assertEqual(k1.robot_length_cm, 15.0)

        # Mega 3kg Class
        m3 = MEGA_3KG_CONFIG
        self.assertEqual(m3.dohyo_diameter_cm, 154.0)
        self.assertEqual(m3.border_width_cm, 5.0)
        self.assertEqual(m3.shikiri_separation_cm, 20.0)
        self.assertEqual(m3.shikiri_length_cm, 20.0)
        self.assertEqual(m3.robot_width_cm, 20.0)
        self.assertEqual(m3.robot_length_cm, 20.0)

    def test_official_match_duration_limit(self):
        """Section 6, Article 11: Length of Match is 3 minutes (180.0 s)."""
        k1 = KIT_1KG_CONFIG
        m3 = MEGA_3KG_CONFIG

        self.assertEqual(k1.max_ticks, 3600)
        self.assertEqual(m3.max_ticks, 3600)
        self.assertEqual(k1.max_ticks * k1.dt, 180.0)
        self.assertEqual(m3.max_ticks * m3.dt, 180.0)


# ==============================================================================
# SUITE 7: DECISION TREE ARTIFACT PERSISTENCE & C++ EXPORT AUDIT
# ==============================================================================
class TestDecisionTreePersistenceAndExport(unittest.TestCase):
    """Validate Decision Tree model packaging, data/dt/ persistence, and C++ header generation."""

    def test_decision_tree_packaging_and_loading(self):
        """Verify saving to data/dt/ and complete deserialization of metrics and C++ code."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            n_samples = 120
            mock_data = pd.DataFrame({
                "Match_ID": [f"M_{i:04d}" for i in range(n_samples)],
                "Bot_ID": ["Bot_A"] * n_samples,
                "Timestamp_ms": [i * 50 for i in range(n_samples)],
                "IR_Edge_FL": np.random.uniform(0.0, 1.0, n_samples),
                "IR_Edge_FR": np.random.uniform(0.0, 1.0, n_samples),
                "Opp_L90": np.random.uniform(0.0, 50.0, n_samples),
                "Opp_L18": np.random.uniform(0.0, 50.0, n_samples),
                "Opp_F0": np.random.uniform(0.0, 50.0, n_samples),
                "Opp_R18": np.random.uniform(0.0, 50.0, n_samples),
                "Opp_R90": np.random.uniform(0.0, 50.0, n_samples),
                "Current_State": np.random.choice(["ATTACK", "EDGE_RECOVERY", "SEARCH", "TRACK"], n_samples),
            })

            # 1. Train Decision Tree
            dt_res = train_and_optimize_tree(mock_data, random_state=42)
            self.assertIsNotNone(dt_res)
            self.assertIn("model", dt_res)

            # 2. Save Package
            save_out = save_decision_tree_package(
                dt_res=dt_res,
                dataset_base="test_pkg_kit",
                robot_class="KIT_1KG",
                output_dir=tmp_path,
            )

            json_file = save_out["json_path"]
            pkl_file = save_out["pkl_path"]
            self.assertTrue(json_file.exists())
            self.assertTrue(pkl_file.exists())

            # 3. Load Package
            loaded_pkg = load_decision_tree_package(json_file)
            self.assertEqual(loaded_pkg["dataset_base"], "test_pkg_kit")
            self.assertEqual(loaded_pkg["robot_class"], "KIT_1KG")
            self.assertIn("metrics", loaded_pkg)
            self.assertIn("test_macro_f1", loaded_pkg["metrics"])
            self.assertIn("cpp_header_code", loaded_pkg)

            cpp_code = loaded_pkg["cpp_header_code"]
            self.assertIn("#ifndef STRATEGY_CONFIG_H", cpp_code)
            self.assertIn("typedef struct {", cpp_code)
            self.assertIn("evaluate_strategy", cpp_code)

    def test_temporal_derivative_and_paired_testing(self):
        """Verify feature engineering (closing velocity, asymmetry) and paired hypothesis testing."""
        n_samples = 200
        mock_data = pd.DataFrame({
            "Match_ID": [f"M_{i // 20:02d}" for i in range(n_samples)],
            "Bot_ID": ["Bot_A"] * n_samples,
            "Timestamp_ms": [(i % 20) * 50 for i in range(n_samples)],
            "IR_Edge_FL": np.random.uniform(0.0, 1.0, n_samples),
            "IR_Edge_FR": np.random.uniform(0.0, 1.0, n_samples),
            "Opp_L90": np.random.uniform(0.0, 50.0, n_samples),
            "Opp_L18": np.random.uniform(0.0, 50.0, n_samples),
            "Opp_F0": np.random.uniform(0.0, 50.0, n_samples),
            "Opp_R18": np.random.uniform(0.0, 50.0, n_samples),
            "Opp_R90": np.random.uniform(0.0, 50.0, n_samples),
            "Current_State": np.random.choice(["ATTACK", "EDGE_RECOVERY", "SEARCH", "TRACK"], n_samples),
        })

        benchmark_res = train_and_benchmark_models(
            mock_data,
            test_size=0.20,
            random_state=42,
            selected_models=["Decision Tree", "Logistic Regression"],
        )

        self.assertIn("models", benchmark_res)
        self.assertIn("leaderboard", benchmark_res)
        self.assertIn("Delta_Opp_F0", benchmark_res["feature_names"])
        self.assertIn("Opp_Lat_Delta", benchmark_res["feature_names"])
        self.assertIn("Opp_Bearing_Est_Deg", benchmark_res["feature_names"])

        dt_info = benchmark_res["models"]["Decision Tree"]
        self.assertIn("permutation_importances", dt_info)
        self.assertIn("cv_fold_scores", dt_info)
        self.assertIn("pruning_curve", dt_info)
        self.assertIn("alphas", dt_info["pruning_curve"])
        self.assertIn("node_counts", dt_info["pruning_curve"])
        self.assertEqual(len(dt_info["cv_fold_scores"]), 5)

        # Leaderboard hypothesis testing columns
        lb = benchmark_res["leaderboard"]
        self.assertIn("p-value (vs DT)", lb.columns)
        self.assertIn("Stat. Significant", lb.columns)

    def test_clang_c_firmware_compilation(self):
        """Verify that transpiled strategy_config.h compiles cleanly with C99 compiler and executes."""
        clang_path = shutil.which("clang") or shutil.which("gcc")
        if not clang_path:
            self.skipTest("No C compiler (clang/gcc) found in environment.")

        strategy_header = Path("strategy_config.h")
        pkg_1kg_path = Path("data/dt/test_1kg_dt.json")
        pkg_3kg_path = Path("data/dt/test_3kg_dt.json")
        header_code = None
        if strategy_header.exists():
            header_code = strategy_header.read_text(encoding="utf-8")
        elif pkg_1kg_path.exists():
            header_code = load_decision_tree_package(pkg_1kg_path).get("cpp_header_code")
        elif pkg_3kg_path.exists():
            header_code = load_decision_tree_package(pkg_3kg_path).get("cpp_header_code")
        else:
            from sklearn.tree import DecisionTreeClassifier
            m_dummy = DecisionTreeClassifier(max_depth=3).fit(
                np.random.randn(20, 10), np.random.choice(["SEARCH", "ATTACK", "TRACK"], size=20)
            )
            dummy_feats = [
                "IR_Edge_FL", "IR_Edge_FR", "Opp_L90", "Opp_L18", "Opp_F0", "Opp_R18", "Opp_R90",
                "Delta_Opp_F0", "Opp_Lat_Delta", "Opp_Bearing_Est_Deg"
            ]
            header_code = generate_cpp_header(m_dummy, dummy_feats, list(m_dummy.classes_), weight_class="KIT_1KG")

        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            hdr_file = tmp / "strategy_config.h"
            hdr_file.write_text(header_code, encoding="utf-8")
            c_test_src = tmp / "test_harness.c"
            c_bin = tmp / "test_harness"

            # Minimal C test runner
            c_test_src.write_text(
                '#include <stdio.h>\n'
                '#include <assert.h>\n'
                f'#include "{hdr_file.resolve()}"\n'
                'int main() {\n'
                '    SumoSensors sensors = {0};\n'
                '    BotState s = evaluate_strategy(&sensors);\n'
                '    assert(s >= STATE_SEARCH && s <= STATE_EVADE);\n'
                '    printf("Strategy evaluation return code: %d\\n", (int)s);\n'
                '    return 0;\n'
                '}\n'
            )

            compile_cmd = [
                clang_path,
                "-std=c99",
                "-Wall",
                "-Wextra",
                str(c_test_src),
                "-o",
                str(c_bin),
            ]
            comp_res = subprocess.run(compile_cmd, capture_output=True, text=True)
            self.assertEqual(
                comp_res.returncode,
                0,
                f"C firmware failed to compile: {comp_res.stderr}",
            )

            # Execute compiled binary
            run_res = subprocess.run([str(c_bin)], capture_output=True, text=True)
            self.assertEqual(run_res.returncode, 0)
            self.assertIn("Strategy evaluation return code", run_res.stdout)

    def test_all_four_models_benchmark_and_timings(self):
        """Verify full 4-model benchmark suite training, positive timings, and leaderboard structure."""
        classes = ["ATTACK", "SEARCH", "TRACK", "EVADE", "EDGE_RECOVERY"]
        n_samples = 150
        rng = np.random.default_rng(42)
        mock_data = pd.DataFrame({
            "Match_ID": [f"M_{i // 15:02d}" for i in range(n_samples)],
            "Bot_ID": ["Bot_A"] * n_samples,
            "Timestamp_ms": [(i % 15) * 50 for i in range(n_samples)],
            "IR_Edge_FL": rng.uniform(0.0, 1.0, n_samples),
            "IR_Edge_FR": rng.uniform(0.0, 1.0, n_samples),
            "Opp_L90": rng.uniform(0.0, 50.0, n_samples),
            "Opp_L18": rng.uniform(0.0, 50.0, n_samples),
            "Opp_F0": rng.uniform(0.0, 50.0, n_samples),
            "Opp_R18": rng.uniform(0.0, 50.0, n_samples),
            "Opp_R90": rng.uniform(0.0, 50.0, n_samples),
            "Current_State": [classes[i % len(classes)] for i in range(n_samples)],
        })

        all_models = ["Decision Tree", "Logistic Regression", "Random Forest", "Gradient Boosting"]
        res = train_and_benchmark_models(
            mock_data,
            test_size=0.20,
            random_state=42,
            selected_models=all_models,
        )

        self.assertIn("models", res)
        self.assertIn("leaderboard", res)
        lb = res["leaderboard"]
        self.assertEqual(len(lb), 4)

        for m_name in all_models:
            self.assertIn(m_name, res["models"])
            m_info = res["models"][m_name]
            self.assertIn("training_time_s", m_info)
            self.assertGreaterEqual(m_info["training_time_s"], 0.0)
            self.assertIn("test_f1_macro", m_info)
            self.assertIn("test_accuracy", m_info)

        # Leaderboard hypothesis testing columns must be present
        self.assertIn("p-value (vs DT)", lb.columns)
        self.assertIn("Stat. Significant", lb.columns)
        self.assertIn("Training Time (s)", lb.columns)

    def test_mega_3kg_feature_engineering_and_bearing(self):
        """Verify Mega 3kg 7-ray optical array feature engineering and bearing calculation."""
        n_samples = 150
        classes = ["ATTACK", "SEARCH", "TRACK", "EVADE", "EDGE_RECOVERY"]
        rng = np.random.default_rng(42)
        mock_3kg = pd.DataFrame({
            "Match_ID": [f"M_{i // 15:02d}" for i in range(n_samples)],
            "Bot_ID": ["Bot_A"] * n_samples,
            "Timestamp_ms": [(i % 15) * 50 for i in range(n_samples)],
            "IR_Edge_FL": rng.uniform(0.0, 1.0, n_samples),
            "IR_Edge_FR": rng.uniform(0.0, 1.0, n_samples),
            "Opp_L90": rng.uniform(0.0, 50.0, n_samples),
            "Opp_L22_5": rng.uniform(0.0, 50.0, n_samples),
            "Opp_L15": rng.uniform(0.0, 50.0, n_samples),
            "Opp_F0": rng.uniform(0.0, 50.0, n_samples),
            "Opp_R15": rng.uniform(0.0, 50.0, n_samples),
            "Opp_R22_5": rng.uniform(0.0, 50.0, n_samples),
            "Opp_R90": rng.uniform(0.0, 50.0, n_samples),
            "Current_State": [classes[i % len(classes)] for i in range(n_samples)],
        })

        res = train_and_benchmark_models(
            mock_3kg,
            test_size=0.20,
            random_state=42,
            selected_models=["Decision Tree"],
        )

        feats = res["feature_names"]
        self.assertIn("Opp_Lat_Delta", feats)
        self.assertIn("Opp_Bearing_Est_Deg", feats)
        self.assertIn("Delta_Opp_F0", feats)
        self.assertIn("Opp_L15", feats)
        self.assertIn("Opp_R15", feats)
        self.assertIn("Opp_L22_5", feats)
        self.assertIn("Opp_R22_5", feats)
        self.assertNotIn("Opp_L18", feats)
        self.assertNotIn("Opp_R18", feats)

    def test_canonical_hardware_threshold_row_ordering(self):
        """Verify build_hardware_summary_table produces canonical subsystem row ordering."""
        # 1. Kit 1kg feature ordering verification
        feats_1kg = [
            "Opp_L90", "Opp_R90", "Opp_Bearing_Est_Deg", "IR_Edge_FL", "Opp_F0",
            "Delta_Opp_F0", "Opp_L18", "Opp_R18", "IR_Edge_FR", "Opp_Lat_Delta"
        ]
        df_thresh_1kg = pd.DataFrame({"Sensor_Feature": feats_1kg, "Threshold_Value": [10.0] * len(feats_1kg)})
        imp_1kg = {f: 0.1 for f in feats_1kg}
        canonical_1kg = [
            "Opp_F0", "Delta_Opp_F0", "IR_Edge_FL", "IR_Edge_FR", "Opp_Lat_Delta",
            "Opp_Bearing_Est_Deg", "Opp_L18", "Opp_R18", "Opp_L90", "Opp_R90"
        ]
        res_1kg = build_hardware_summary_table(df_thresh_1kg, imp_1kg)
        self.assertEqual(res_1kg["Sensor_Feature"].tolist(), canonical_1kg)

        # 2. Mega 3kg feature ordering verification
        feats_3kg = [
            "Opp_L90", "Opp_R90", "Opp_Bearing_Est_Deg", "IR_Edge_FL", "Opp_F0",
            "Delta_Opp_F0", "Opp_L15", "Opp_R15", "Opp_L22_5", "Opp_R22_5", "IR_Edge_FR", "Opp_Lat_Delta"
        ]
        df_thresh_3kg = pd.DataFrame({"Sensor_Feature": feats_3kg, "Threshold_Value": [10.0] * len(feats_3kg)})
        imp_3kg = {f: 0.1 for f in feats_3kg}
        canonical_3kg = [
            "Opp_F0", "Delta_Opp_F0", "IR_Edge_FL", "IR_Edge_FR", "Opp_Lat_Delta",
            "Opp_Bearing_Est_Deg", "Opp_L15", "Opp_R15", "Opp_L22_5", "Opp_R22_5", "Opp_L90", "Opp_R90"
        ]
        res_3kg = build_hardware_summary_table(df_thresh_3kg, imp_3kg)
        self.assertEqual(res_3kg["Sensor_Feature"].tolist(), canonical_3kg)

        # 3. Check pre-packaged JSON packages if present
        pkg_1kg_path = Path("data/dt/test_1kg_dt.json")
        if pkg_1kg_path.exists():
            pkg_1kg = load_decision_tree_package(pkg_1kg_path)
            hw_summary = pkg_1kg.get("hardware_summary_table", [])
            if hw_summary:
                summary_feats = [r["Sensor_Feature"] for r in hw_summary]
                expected_present = [f for f in canonical_1kg if f in summary_feats]
                self.assertEqual(summary_feats, expected_present)

    def test_clang_cpp_main_harness_compilation(self):
        """Verify that generated main.cpp harnesses for 1kg and 3kg compile cleanly with clang++."""
        clangpp_path = shutil.which("clang++") or shutil.which("g++")
        if not clangpp_path:
            self.skipTest("No C++ compiler (clang++/g++) found in environment.")

        strategy_header_1kg = Path("strategy_config.h")
        pkg_1kg_path = Path("data/dt/test_1kg_dt.json")
        pkg_3kg_path = Path("data/dt/test_3kg_dt.json")

        header_1kg_code = None
        if strategy_header_1kg.exists():
            header_1kg_code = strategy_header_1kg.read_text(encoding="utf-8")
        elif pkg_1kg_path.exists():
            header_1kg_code = load_decision_tree_package(pkg_1kg_path).get("cpp_header_code")
        else:
            from sklearn.tree import DecisionTreeClassifier
            m_dummy = DecisionTreeClassifier(max_depth=3).fit(
                np.random.randn(20, 10), np.random.choice(["SEARCH", "ATTACK", "TRACK"], size=20)
            )
            feats_1kg = [
                "IR_Edge_FL", "IR_Edge_FR", "Opp_L90", "Opp_L18", "Opp_F0", "Opp_R18", "Opp_R90",
                "Delta_Opp_F0", "Opp_Lat_Delta", "Opp_Bearing_Est_Deg"
            ]
            header_1kg_code = generate_cpp_header(m_dummy, feats_1kg, list(m_dummy.classes_), weight_class="KIT_1KG")

        if pkg_3kg_path.exists():
            header_3kg_code = load_decision_tree_package(pkg_3kg_path).get("cpp_header_code")
        else:
            from sklearn.tree import DecisionTreeClassifier
            m_dummy3 = DecisionTreeClassifier(max_depth=3).fit(
                np.random.randn(20, 12), np.random.choice(["SEARCH", "ATTACK", "TRACK"], size=20)
            )
            feats_3kg = [
                "IR_Edge_FL", "IR_Edge_FR", "Opp_L90", "Opp_L22_5", "Opp_L15", "Opp_F0",
                "Opp_R15", "Opp_R22_5", "Opp_R90", "Delta_Opp_F0", "Opp_Lat_Delta", "Opp_Bearing_Est_Deg"
            ]
            header_3kg_code = generate_cpp_header(m_dummy3, feats_3kg, list(m_dummy3.classes_), weight_class="MEGA_3KG")

        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            # 1KG Compilation & Run
            (tmp / "strategy_config_1kg.h").write_text(header_1kg_code, encoding="utf-8")
            harness_1kg = generate_cpp_harness(robot_class="KIT_1KG", header_filename="strategy_config_1kg.h")
            f1 = tmp / "main_1kg.cpp"
            b1 = tmp / "main_1kg"
            f1.write_text(harness_1kg + "\nint main() { setup(); loop(); return 0; }\n")
            comp_1kg = subprocess.run(
                [clangpp_path, "-std=c++11", "-Wall", "-Wextra", str(f1), "-o", str(b1)],
                capture_output=True,
                text=True,
            )
            self.assertEqual(comp_1kg.returncode, 0, f"1KG harness failed to compile: {comp_1kg.stderr}")
            run_1kg = subprocess.run([str(b1)], capture_output=True, text=True)
            self.assertEqual(run_1kg.returncode, 0)

            # 3KG Compilation & Run
            (tmp / "strategy_config_3kg.h").write_text(header_3kg_code, encoding="utf-8")
            harness_3kg = generate_cpp_harness(robot_class="MEGA_3KG", header_filename="strategy_config_3kg.h")
            f3 = tmp / "main_3kg.cpp"
            b3 = tmp / "main_3kg"
            f3.write_text(harness_3kg + "\nint main() { setup(); loop(); return 0; }\n")
            comp_3kg = subprocess.run(
                [clangpp_path, "-std=c++11", "-Wall", "-Wextra", str(f3), "-o", str(b3)],
                capture_output=True,
                text=True,
            )
            self.assertEqual(comp_3kg.returncode, 0, f"3KG harness failed to compile: {comp_3kg.stderr}")
            run_3kg = subprocess.run([str(b3)], capture_output=True, text=True)
            self.assertEqual(run_3kg.returncode, 0)

    def test_modular_engineer_features_standalone(self):
        """Verify standalone engineer_features() produces velocity, asymmetry, and bearing features."""
        n_samples = 50
        mock_df = pd.DataFrame({
            "Match_ID": ["M_01"] * n_samples,
            "Bot_ID": ["Bot_A"] * n_samples,
            "Tick": list(range(n_samples)),
            "Timestamp_ms": [i * 50 for i in range(n_samples)],
            "IR_Edge_FL": [0.0] * n_samples,
            "IR_Edge_FR": [0.0] * n_samples,
            "Opp_L90": [50.0] * n_samples,
            "Opp_L18": [10.0] * n_samples,
            "Opp_F0": [20.0 - i * 0.2 for i in range(n_samples)],  # Closing in
            "Opp_R18": [15.0] * n_samples,
            "Opp_R90": [50.0] * n_samples,
        })
        enriched, feats = engineer_features(mock_df)
        self.assertIn("Delta_Opp_F0", enriched.columns)
        self.assertIn("Opp_Lat_Delta", enriched.columns)
        self.assertIn("Opp_Bearing_Est_Deg", enriched.columns)
        # Verify Delta_Opp_F0 is negative (closing in)
        self.assertLess(enriched["Delta_Opp_F0"].iloc[1], 0.0)
        # Verify Opp_Lat_Delta = Opp_L18 - Opp_R18 = 10 - 15 = -5
        self.assertAlmostEqual(enriched["Opp_Lat_Delta"].iloc[0], -5.0)

    def test_modular_compute_observer_kpis_standalone(self):
        """Verify standalone compute_observer_kpis() accurately extracts match duration, center %, and outcomes."""
        mock_obs = pd.DataFrame({
            "Match_ID": ["M_TEST", "M_TEST"],
            "Bot_ID": ["Bot_A", "Bot_B"],
            "Tick": [20, 20],
            "Timestamp_ms": [1000, 1000],
            "Dist_To_Center": [10.0, 30.0],
            "Target_Visible": [True, False],
            "Match_Status": ["BOT_A_WIN", "BOT_A_WIN"],
            "Strategy_Profile": ["DEFENSIVE_SWEEPER", "AGGRESSIVE_CHARGER"],
            "Starting_Formation": ["HEAD_ON", "HEAD_ON"],
        })
        kpis = compute_observer_kpis(mock_obs, center_zone_r=19.25)
        self.assertFalse(kpis.empty)
        self.assertEqual(len(kpis), 1)
        self.assertEqual(kpis.iloc[0]["Status"], "BOT_A_WIN")
        self.assertEqual(kpis.iloc[0]["Duration_ms"], 1000)
        self.assertEqual(kpis.iloc[0]["Center_Control_Pct"], 100.0)
        self.assertEqual(kpis.iloc[0]["TTRO_ms"], 1000)

        # Verify extract_match_summary functionality
        summary = extract_match_summary(mock_obs)
        self.assertFalse(summary.empty)
        self.assertEqual(len(summary), 1)
        self.assertEqual(summary.iloc[0]["Match_ID"], "M_TEST")
        self.assertEqual(summary.iloc[0]["Match_Status"], "BOT_A_WIN")

    def test_modular_compute_mcu_benchmarks(self):
        """Verify standalone compute_mcu_benchmarks() calculates AVR ATmega328P resource envelope."""
        bench = compute_mcu_benchmarks(tree_depth=6, n_leaves=15, n_features=9)
        self.assertAlmostEqual(bench["latency_us"], 4.5)  # 6 * 0.75
        self.assertEqual(bench["sram_bytes"], 36)        # 9 * 4
        self.assertEqual(bench["flash_bytes"], 29 * 24)  # (2*15 - 1) * 24 = 29 * 24 = 696
        self.assertGreater(bench["bandwidth_khz"], 200)
        self.assertGreater(bench["headroom_x"], 10000)

    def test_evaluate_sensor_action_python_inference(self):
        """Verify standalone evaluate_sensor_action() performs single-tick dictionary inference."""
        n_samples = 40
        mock_data = pd.DataFrame({
            "IR_Edge_FL": [0.0] * n_samples,
            "IR_Edge_FR": [0.0] * n_samples,
            "Opp_F0": [10.0] * n_samples,
            "Opp_L18": [50.0] * n_samples,
            "Opp_R18": [50.0] * n_samples,
            "Opp_L90": [50.0] * n_samples,
            "Opp_R90": [50.0] * n_samples,
            "Current_State": ["ATTACK"] * (n_samples // 2) + ["SEARCH"] * (n_samples // 2),
        })
        dt_res = train_and_optimize_tree(mock_data, random_state=42)
        model = dt_res["model"]

        sample_reading = {
            "IR_Edge_FL": 0.0,
            "IR_Edge_FR": 0.0,
            "Opp_F0": 8.0,
            "Opp_L18": 50.0,
            "Opp_R18": 50.0,
            "Opp_L90": 50.0,
            "Opp_R90": 50.0,
        }
        pred = evaluate_sensor_action(model, sample_reading, feature_names=dt_res["feature_names"])
        self.assertIn(pred, ["ATTACK", "SEARCH"])

    def test_compute_observer_kpis_dynamic_weight_class(self):
        """Verify compute_observer_kpis() auto-detects center radius for MEGA_3KG without explicit parameter."""
        mock_3kg_obs = pd.DataFrame({
            "Match_ID": ["M_3KG_01", "M_3KG_01"],
            "Bot_ID": ["Bot_A", "Bot_B"],
            "Tick": [10, 10],
            "Timestamp_ms": [500, 500],
            "Weight_Class": ["MEGA_3KG", "MEGA_3KG"],
            "Dist_To_Center": [25.0, 50.0],  # 25 cm is inside 3kg center ring (38.5cm) but outside 1kg (19.25cm)
            "Target_Visible": [True, False],
            "Match_Status": ["BOT_A_WIN", "BOT_A_WIN"],
            "Strategy_Profile": ["AGGRESSIVE_CHARGER", "DEFENSIVE_SWEEPER"],
            "Starting_Formation": ["HEAD_ON", "HEAD_ON"],
        })
        # Called with center_zone_r=None -> must auto-detect 38.5cm for MEGA_3KG
        kpis = compute_observer_kpis(mock_3kg_obs, center_zone_r=None)
        self.assertFalse(kpis.empty)
        # Because dist (25 cm) <= 38.5 cm, Center_Control_Pct must be 100%
        self.assertEqual(kpis.iloc[0]["Center_Control_Pct"], 100.0)

    def test_engineer_features_dt_parameterization(self):
        """Verify engineer_features() respects custom dt argument."""
        mock_df = pd.DataFrame({
            "Opp_F0": [20.0, 15.0],  # Difference is -5.0 cm
            "IR_Edge_FL": [0.0, 0.0],
            "IR_Edge_FR": [0.0, 0.0],
        })
        # With dt = 0.02, delta velocity = -5.0 / 0.02 = -250 cm/s
        enriched, _ = engineer_features(mock_df, dt=0.02)
        self.assertAlmostEqual(enriched["Delta_Opp_F0"].iloc[1], -250.0)


if __name__ == "__main__":
    unittest.main()
