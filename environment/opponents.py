"""Finite State Machine (FSM) Opponent Strategies.

Implements authentic tournament tactical behaviors for Autonomous Sumo Robots (RoboGames rules):
1. AGGRESSIVE_CHARGER (Direct Bull-Rush)
2. DEFENSIVE_SWEEPER (Center Counter-Striker)
3. RANDOM_FLANKER (Evasive Flank Hunter)
4. BAIT_AND_SWITCH (Tactical Retreater with Matador Slip)
5. JUGGERNAUT_PUSH (Torque Creep with Forward Pressure)
6. RANDOM_MIX (Uniform stochastic sampling)

Key Architectural Principles (Official RoboGames Compliance):
- Opening Blitz: Active forward rush on buzzer (no static start line freeze).
- Locked Bull-Rush Charge: Locks forward thrust upon target lock to commit to attacks and smash targets.
- Snap-to-Bearing Flank Intercept: Snaps onto enemy bearing in 1-2 ticks and accelerates into the attack.
- Three-Phase Safe Edge Recovery: Disengage back-step -> sharp inward pivot -> forward center drive.
- Arena-Traversing Search: Active wide-area scanning crossing the Dohyo center.
"""

from __future__ import annotations

import abc
import random
from typing import Any, Dict, Tuple

from config.config import (
    EDGE_RECOVERY_DURATION_TICKS,
    PWM_ATTACK_FULL,
    PWM_JUGGERNAUT,
    RobotClassConfig,
)


# ==============================================================================
# 1. BASE FINITE STATE MACHINE OPPONENT
# ==============================================================================
class BaseFSMOpponent(abc.ABC):
    """Abstract Base Class for all Sumo Robot FSM controllers."""

    def __init__(
        self,
        config: RobotClassConfig,
        name: str = "BaseFSM",
        search_switch_interval: int = 25,
    ) -> None:
        self.config = config
        self.name = name
        self._strategy_name = name.upper()
        self.state = "SEARCH"
        self.edge_recovery_ticks_remaining = 0
        self.recovery_pivot_dir = 1
        self.ticks_in_state = 0
        self.search_switch_interval = search_switch_interval
        self.search_dir = 1
        self.search_timer = 0
        self.charge_lock_ticks = 0
        self.consecutive_search_ticks = 0
        self.edge_consecutive_ticks = 0

    @property
    def strategy_name(self) -> str:
        return self._strategy_name

    def reset(self) -> None:
        self.state = "SEARCH"
        self.edge_recovery_ticks_remaining = 0
        self.recovery_pivot_dir = 1
        self.ticks_in_state = 0
        self.charge_lock_ticks = 0
        self.consecutive_search_ticks = 0
        self.edge_consecutive_ticks = 0
        self.search_dir = random.choice([1, -1])
        self.search_timer = random.randint(0, max(1, self.search_switch_interval // 3))

    def check_edge_trigger(self, obs: Dict[str, Any]) -> bool:
        """Check if any downward-facing QRE1113 edge sensor has triggered."""
        for sensor_name in self.config.edge_sensor_names:
            if obs.get(sensor_name, 0.0) >= self.config.edge_threshold:
                return True
        return False

    def update_search_direction(self) -> int:
        """Alternate search arc direction periodically."""
        self.search_timer += 1
        if self.search_timer >= self.search_switch_interval:
            self.search_dir *= -1
            self.search_timer = 0
        return self.search_dir

    def _opening_step(self, obs: Dict[str, Any]) -> Tuple[int, int, str]:
        """Opening rush for the first 6 ticks (300ms) to claim the center line."""
        self.state = "ATTACK"
        return 255, 255, "ATTACK"

    def compute_action(self, obs: Dict[str, Any]) -> Tuple[int, int, str]:
        """Process observation and return (PWM_Left, PWM_Right, State_String)."""
        self.ticks_in_state += 1

        # Priority 1: Three-Phase Safe Edge Recovery with 2-Sample Digital Noise Debounce
        fl_val = obs.get("IR_Edge_FL", 0.0)
        fr_val = obs.get("IR_Edge_FR", 0.0)
        raw_edge_triggered = (fl_val >= self.config.edge_threshold or fr_val >= self.config.edge_threshold)

        if raw_edge_triggered:
            self.edge_consecutive_ticks += 1
        else:
            self.edge_consecutive_ticks = 0

        # Require 2 consecutive samples to confirm genuine physical boundary crossing (rejects 1-tick EMI noise)
        edge_confirmed = (self.edge_consecutive_ticks >= 2)

        if edge_confirmed and self.edge_recovery_ticks_remaining <= 0:
            self.edge_recovery_ticks_remaining = EDGE_RECOVERY_DURATION_TICKS
            self.charge_lock_ticks = 0
            if fl_val >= self.config.edge_threshold and fr_val < self.config.edge_threshold:
                self.recovery_pivot_dir = 1   # Left sensor hit: boundary on left -> pivot right
            elif fr_val >= self.config.edge_threshold and fl_val < self.config.edge_threshold:
                self.recovery_pivot_dir = -1  # Right sensor hit: boundary on right -> pivot left
            else:
                self.recovery_pivot_dir = random.choice([1, -1])

        if self.edge_recovery_ticks_remaining > 0:
            self.edge_recovery_ticks_remaining -= 1
            self.state = "EDGE_RECOVERY"

            # Phase 1 (Ticks 1-2, 100ms): Brief back-step to disengage wedge from white line
            if self.edge_recovery_ticks_remaining >= 6:
                return -220, -220, "EDGE_RECOVERY"
            # Phase 2 (Ticks 3-5, 150ms): Sharp pivot away from the boundary
            elif self.edge_recovery_ticks_remaining >= 3:
                if self.recovery_pivot_dir > 0:
                    return 255, -140, "EDGE_RECOVERY"
                else:
                    return -140, 255, "EDGE_RECOVERY"
            # Phase 3 (Ticks 6-8, 150ms): Drive forward back into the Dohyo center!
            else:
                return 255, 255, "EDGE_RECOVERY"

        # Priority 2: Opening Match Blitz (first 6 ticks = 300ms from start)
        if obs.get("Tick", 0) < 6:
            return self._opening_step(obs)

        # Priority 3: Strategy-specific tactical state machine
        action = self._strategy_step(obs)
        if action[2] == "SEARCH":
            self.consecutive_search_ticks += 1
        else:
            self.consecutive_search_ticks = 0

        return action

    @abc.abstractmethod
    def _strategy_step(self, obs: Dict[str, Any]) -> Tuple[int, int, str]:
        """Subclass specific FSM logic."""
        pass


# ==============================================================================
# 2. BENCHMARK STRATEGY 1: AGGRESSIVE CHARGER
# ==============================================================================
class AggressiveCharger(BaseFSMOpponent):
    """Direct Bull-Rush: High-speed forward charge with proportional target centering."""

    def __init__(self, config: RobotClassConfig) -> None:
        super().__init__(config, name="AGGRESSIVE_CHARGER", search_switch_interval=25)

    def _opening_step(self, obs: Dict[str, Any]) -> Tuple[int, int, str]:
        start_form = str(obs.get("Starting_Formation", "HEAD_ON")).upper()
        tick = int(obs.get("Tick", 0))

        if "SIDE" in start_form:
            # Side Start: Quick 2-tick inward snap-turn, then full charge
            if tick < 2:
                self.state = "TRACK"
                return (-220, 255, "TRACK") if self.search_dir > 0 else (255, -220, "TRACK")
            else:
                self.state = "ATTACK"
                return 255, 255, "ATTACK"
        elif "OFFSET" in start_form:
            # Lateral Offset: Intercept angle
            self.state = "ATTACK"
            return (255, 190, "ATTACK") if self.search_dir > 0 else (190, 255, "ATTACK")
        else:
            # Head-On or Angled-Inward: Full throttle straight blitz
            self.state = "ATTACK"
            return 255, 255, "ATTACK"

    def _strategy_step(self, obs: Dict[str, Any]) -> Tuple[int, int, str]:
        opp_f0 = obs.get("Opp_F0", -1.0)
        opp_l18 = max(obs.get("Opp_L18", -1.0), obs.get("Opp_L15", -1.0), obs.get("Opp_L22_5", -1.0))
        opp_r18 = max(obs.get("Opp_R18", -1.0), obs.get("Opp_R15", -1.0), obs.get("Opp_R22_5", -1.0))
        opp_l90 = obs.get("Opp_L90", -1.0)
        opp_r90 = obs.get("Opp_R90", -1.0)

        # 1. 90-degree lateral flank -> Sharp snap turn
        if opp_l90 > 0.0 and (opp_r90 < 0 or opp_l90 < opp_r90) and opp_f0 < 0 and opp_r18 < 0:
            self.state = "TRACK"
            return -220, 255, "TRACK"

        if opp_r90 > 0.0 and (opp_l90 < 0 or opp_r90 < opp_l90) and opp_f0 < 0 and opp_l18 < 0:
            self.state = "TRACK"
            return 255, -220, "TRACK"

        # 2. Diagonal tracking (turn continuously to acquire front lock)
        if opp_f0 < 0.0:
            if opp_l18 > 0.0 and (opp_r18 < 0.0 or opp_l18 < opp_r18):
                self.state = "TRACK"
                return -160, 255, "TRACK"
            elif opp_r18 > 0.0:
                self.state = "TRACK"
                return 255, -160, "TRACK"

        # 3. Front Lock (F0 > 0) -> Proportional centering attack
        if opp_f0 > 0.0:
            self.state = "ATTACK"
            if opp_r18 > 0.0 and (opp_l18 < 0.0 or opp_r18 < opp_l18):
                return 255, 160, "ATTACK"
            elif opp_l18 > 0.0 and (opp_r18 < 0.0 or opp_l18 < opp_r18):
                return 160, 255, "ATTACK"
            else:
                return PWM_ATTACK_FULL, PWM_ATTACK_FULL, "ATTACK"

        # 4. Search Sweep
        self.state = "SEARCH"
        s_dir = self.update_search_direction()
        if s_dir > 0:
            return 255, 170, "SEARCH"
        else:
            return 170, 255, "SEARCH"


# ==============================================================================
# 3. BENCHMARK STRATEGY 2: DEFENSIVE SWEEPER
# ==============================================================================
class DefensiveSweeper(BaseFSMOpponent):
    """Center Counter-Striker: Claims ring center; attacks with maximum torque on engagement."""

    def __init__(self, config: RobotClassConfig) -> None:
        super().__init__(config, name="DEFENSIVE_SWEEPER", search_switch_interval=20)

    def _opening_step(self, obs: Dict[str, Any]) -> Tuple[int, int, str]:
        start_form = str(obs.get("Starting_Formation", "HEAD_ON")).upper()
        tick = int(obs.get("Tick", 0))

        if "SIDE" in start_form:
            # Side Start: Ticks 0-1 pivot inward toward center (0,0), then advance
            if tick < 2:
                self.state = "TRACK"
                return (-180, 180, "TRACK") if self.search_dir > 0 else (180, -180, "TRACK")
            else:
                self.state = "SEARCH"
                return 180, 180, "SEARCH"
        elif "OFFSET" in start_form:
            self.state = "SEARCH"
            return (190, 150, "SEARCH") if self.search_dir > 0 else (150, 190, "SEARCH")
        else:
            # Head-On or Angled-Inward: Measured center advance, then hold
            self.state = "SEARCH"
            if tick < 3:
                return 180, 180, "SEARCH"
            else:
                return 120, 120, "SEARCH"

    def _strategy_step(self, obs: Dict[str, Any]) -> Tuple[int, int, str]:
        opp_f0 = obs.get("Opp_F0", -1.0)
        opp_l18 = max(obs.get("Opp_L18", -1.0), obs.get("Opp_L15", -1.0), obs.get("Opp_L22_5", -1.0))
        opp_r18 = max(obs.get("Opp_R18", -1.0), obs.get("Opp_R15", -1.0), obs.get("Opp_R22_5", -1.0))
        opp_l90 = obs.get("Opp_L90", -1.0)
        opp_r90 = obs.get("Opp_R90", -1.0)

        # 1. 90-degree lateral flank -> Sharp snap turn
        if opp_l90 > 0.0 and (opp_r90 < 0 or opp_l90 < opp_r90) and opp_f0 < 0 and opp_r18 < 0:
            self.state = "TRACK"
            return -200, 200, "TRACK"

        if opp_r90 > 0.0 and (opp_l90 < 0 or opp_r90 < opp_l90) and opp_f0 < 0 and opp_l18 < 0:
            self.state = "TRACK"
            return 200, -200, "TRACK"

        # 2. Diagonal tracking
        if opp_f0 < 0.0:
            if opp_l18 > 0.0 and (opp_r18 < 0.0 or opp_l18 < opp_r18):
                self.state = "TRACK"
                return -140, 255, "TRACK"
            elif opp_r18 > 0.0:
                self.state = "TRACK"
                return 255, -140, "TRACK"

        # 3. Front Lock -> Attack with active centering
        if opp_f0 > 0.0:
            self.state = "ATTACK"
            if opp_r18 > 0.0 and (opp_l18 < 0.0 or opp_r18 < opp_l18):
                return 240, 150, "ATTACK"
            elif opp_l18 > 0.0 and (opp_r18 < 0.0 or opp_l18 < opp_r18):
                return 150, 240, "ATTACK"
            else:
                return PWM_ATTACK_FULL, PWM_ATTACK_FULL, "ATTACK"

        self.state = "SEARCH"
        s_dir = self.update_search_direction()
        if s_dir > 0:
            return 240, 160, "SEARCH"
        else:
            return 160, 240, "SEARCH"


# ==============================================================================
# 4. BENCHMARK STRATEGY 3: RANDOM FLANKER
# ==============================================================================
class RandomFlanker(BaseFSMOpponent):
    """Evasive Flank Hunter: Sweeping curved flank attacks with centering."""

    def __init__(self, config: RobotClassConfig) -> None:
        super().__init__(config, name="RANDOM_FLANKER", search_switch_interval=20)

    def _opening_step(self, obs: Dict[str, Any]) -> Tuple[int, int, str]:
        start_form = str(obs.get("Starting_Formation", "HEAD_ON")).upper()
        tick = int(obs.get("Tick", 0))

        if "SIDE" in start_form:
            # Perimeter drive along the ring curve to wrap around
            self.state = "ATTACK"
            return 255, 190, "ATTACK"
        elif "OFFSET" in start_form:
            self.state = "ATTACK"
            return (255, 140, "ATTACK") if self.search_dir > 0 else (140, 255, "ATTACK")
        elif "ANGLED" in start_form:
            self.state = "ATTACK"
            return (255, 160, "ATTACK") if self.search_dir > 0 else (160, 255, "ATTACK")
        else:
            # Head-on: Immediate curved flank arc to dodge the front collision
            self.state = "ATTACK"
            return (255, 140, "ATTACK") if self.search_dir > 0 else (140, 255, "ATTACK")

    def _strategy_step(self, obs: Dict[str, Any]) -> Tuple[int, int, str]:
        opp_f0 = obs.get("Opp_F0", -1.0)
        opp_l18 = max(obs.get("Opp_L18", -1.0), obs.get("Opp_L15", -1.0), obs.get("Opp_L22_5", -1.0))
        opp_r18 = max(obs.get("Opp_R18", -1.0), obs.get("Opp_R15", -1.0), obs.get("Opp_R22_5", -1.0))
        opp_l90 = obs.get("Opp_L90", -1.0)
        opp_r90 = obs.get("Opp_R90", -1.0)

        # 1. 90-degree lateral flank -> Sharp snap turn
        if opp_l90 > 0.0 and (opp_r90 < 0 or opp_l90 < opp_r90) and opp_f0 < 0 and opp_r18 < 0:
            self.state = "TRACK"
            return -220, 255, "TRACK"

        if opp_r90 > 0.0 and (opp_l90 < 0 or opp_r90 < opp_l90) and opp_f0 < 0 and opp_l18 < 0:
            self.state = "TRACK"
            return 255, -220, "TRACK"

        # 2. Diagonal tracking
        if opp_f0 < 0.0:
            if opp_l18 > 0.0 and (opp_r18 < 0.0 or opp_l18 < opp_r18):
                self.state = "TRACK"
                return -160, 255, "TRACK"
            elif opp_r18 > 0.0:
                self.state = "TRACK"
                return 255, -160, "TRACK"

        # 3. Front Lock -> Full Power Attack
        if opp_f0 > 0.0:
            self.state = "ATTACK"
            if opp_r18 > 0.0 and (opp_l18 < 0.0 or opp_r18 < opp_l18):
                return 255, 160, "ATTACK"
            elif opp_l18 > 0.0 and (opp_r18 < 0.0 or opp_l18 < opp_r18):
                return 160, 255, "ATTACK"
            else:
                return PWM_ATTACK_FULL, PWM_ATTACK_FULL, "ATTACK"

        self.state = "SEARCH"
        s_dir = self.update_search_direction()
        if s_dir > 0:
            return 255, 150, "SEARCH"
        else:
            return 150, 255, "SEARCH"


# ==============================================================================
# 5. BENCHMARK STRATEGY 4: BAIT AND SWITCH
# ==============================================================================
class BaitAndSwitch(BaseFSMOpponent):
    """Tactical Retreater: Opening rush; when opponent approaches < 14cm, executes lateral Matador dodge and counter-strike."""

    def __init__(self, config: RobotClassConfig) -> None:
        super().__init__(config, name="BAIT_AND_SWITCH", search_switch_interval=25)
        self.evade_ticks = 0

    def reset(self) -> None:
        super().reset()
        self.evade_ticks = 0

    def _opening_step(self, obs: Dict[str, Any]) -> Tuple[int, int, str]:
        start_form = str(obs.get("Starting_Formation", "HEAD_ON")).upper()
        tick = int(obs.get("Tick", 0))

        # Check if opponent is already within close range (<= 14cm): if so, immediately execute evasion!
        opp_f0 = obs.get("Opp_F0", -1.0)
        if 0.0 < opp_f0 <= 14.0:
            self.state = "EVADE"
            return (240, 40, "EVADE") if self.search_dir > 0 else (40, 240, "EVADE")

        if "SIDE" in start_form:
            self.state = "SEARCH"
            return (140, -140, "SEARCH") if self.search_dir > 0 else (-140, 140, "SEARCH")
        elif "OFFSET" in start_form:
            self.state = "SEARCH"
            return 150, 150, "SEARCH"
        else:
            # Head-On or Angled-Inward: Short bait creep, then slows down to prep evasion
            self.state = "SEARCH"
            if tick < 2:
                return 160, 160, "SEARCH"
            else:
                return 80, 80, "SEARCH"

    def _strategy_step(self, obs: Dict[str, Any]) -> Tuple[int, int, str]:
        opp_f0 = obs.get("Opp_F0", -1.0)
        opp_l18 = max(obs.get("Opp_L18", -1.0), obs.get("Opp_L15", -1.0), obs.get("Opp_L22_5", -1.0))
        opp_r18 = max(obs.get("Opp_R18", -1.0), obs.get("Opp_R15", -1.0), obs.get("Opp_R22_5", -1.0))
        opp_l90 = obs.get("Opp_L90", -1.0)
        opp_r90 = obs.get("Opp_R90", -1.0)

        # Tactical bait: If opponent is within 14cm and charging, do a quick 3-tick lateral dodge
        if 0.0 < opp_f0 <= 14.0:
            self.evade_ticks += 1
            if self.evade_ticks <= 3:
                self.state = "EVADE"
                return (240, 40, "EVADE") if self.search_dir > 0 else (40, 240, "EVADE")
            else:
                self.state = "ATTACK"
                return 255, 255, "ATTACK"

        self.evade_ticks = 0

        # 1. 90-degree lateral flank -> Sharp snap turn
        if opp_l90 > 0.0 and (opp_r90 < 0 or opp_l90 < opp_r90) and opp_f0 < 0 and opp_r18 < 0:
            self.state = "TRACK"
            return -220, 255, "TRACK"

        if opp_r90 > 0.0 and (opp_l90 < 0 or opp_r90 < opp_l90) and opp_f0 < 0 and opp_l18 < 0:
            self.state = "TRACK"
            return 255, -220, "TRACK"

        # 2. Diagonal tracking
        if opp_f0 < 0.0:
            if opp_l18 > 0.0 and (opp_r18 < 0.0 or opp_l18 < opp_r18):
                self.state = "TRACK"
                return -160, 255, "TRACK"
            elif opp_r18 > 0.0:
                self.state = "TRACK"
                return 255, -160, "TRACK"

        # 3. Front Lock -> Full Power Attack
        if opp_f0 > 0.0:
            self.state = "ATTACK"
            if opp_r18 > 0.0 and (opp_l18 < 0.0 or opp_r18 < opp_l18):
                return 255, 160, "ATTACK"
            elif opp_l18 > 0.0 and (opp_r18 < 0.0 or opp_l18 < opp_r18):
                return 160, 255, "ATTACK"
            else:
                return 255, 255, "ATTACK"

        self.state = "SEARCH"
        s_dir = self.update_search_direction()
        if s_dir > 0:
            return 240, 160, "SEARCH"
        else:
            return 160, 240, "SEARCH"


# ==============================================================================
# 6. BENCHMARK STRATEGY 5: JUGGERNAUT PUSH
# ==============================================================================
class JuggernautPush(BaseFSMOpponent):
    """Torque Creep: High-torque continuous forward engagement on detection."""

    def __init__(self, config: RobotClassConfig) -> None:
        super().__init__(config, name="JUGGERNAUT_PUSH", search_switch_interval=20)

    def _opening_step(self, obs: Dict[str, Any]) -> Tuple[int, int, str]:
        start_form = str(obs.get("Starting_Formation", "HEAD_ON")).upper()
        tick = int(obs.get("Tick", 0))

        if "SIDE" in start_form:
            if tick < 2:
                self.state = "TRACK"
                return (-200, 230, "TRACK") if self.search_dir > 0 else (230, -200, "TRACK")
            else:
                self.state = "ATTACK"
                return PWM_JUGGERNAUT, PWM_JUGGERNAUT, "ATTACK"
        elif "OFFSET" in start_form:
            self.state = "ATTACK"
            return (PWM_JUGGERNAUT, 190, "ATTACK") if self.search_dir > 0 else (190, PWM_JUGGERNAUT, "ATTACK")
        else:
            self.state = "ATTACK"
            return PWM_JUGGERNAUT, PWM_JUGGERNAUT, "ATTACK"

    def _strategy_step(self, obs: Dict[str, Any]) -> Tuple[int, int, str]:
        opp_f0 = obs.get("Opp_F0", -1.0)
        opp_l18 = max(obs.get("Opp_L18", -1.0), obs.get("Opp_L15", -1.0), obs.get("Opp_L22_5", -1.0))
        opp_r18 = max(obs.get("Opp_R18", -1.0), obs.get("Opp_R15", -1.0), obs.get("Opp_R22_5", -1.0))
        opp_l90 = obs.get("Opp_L90", -1.0)
        opp_r90 = obs.get("Opp_R90", -1.0)

        # 1. 90-degree lateral flank -> Sharp snap turn
        if opp_l90 > 0.0 and (opp_r90 < 0 or opp_l90 < opp_r90) and opp_f0 < 0 and opp_r18 < 0:
            self.state = "TRACK"
            return -220, 255, "TRACK"

        if opp_r90 > 0.0 and (opp_l90 < 0 or opp_r90 < opp_l90) and opp_f0 < 0 and opp_l18 < 0:
            self.state = "TRACK"
            return 255, -220, "TRACK"

        # 2. Diagonal tracking
        if opp_f0 < 0.0:
            if opp_l18 > 0.0 and (opp_r18 < 0.0 or opp_l18 < opp_r18):
                self.state = "TRACK"
                return -140, 255, "TRACK"
            elif opp_r18 > 0.0:
                self.state = "TRACK"
                return 255, -140, "TRACK"

        # 3. Front Lock -> Juggernaut Attack
        if opp_f0 > 0.0:
            self.state = "ATTACK"
            if opp_r18 > 0.0 and (opp_l18 < 0.0 or opp_r18 < opp_l18):
                return PWM_JUGGERNAUT, 160, "ATTACK"
            elif opp_l18 > 0.0 and (opp_r18 < 0.0 or opp_l18 < opp_r18):
                return 160, PWM_JUGGERNAUT, "ATTACK"
            else:
                return PWM_JUGGERNAUT, PWM_JUGGERNAUT, "ATTACK"

        self.state = "SEARCH"
        s_dir = self.update_search_direction()
        if s_dir > 0:
            return 240, 170, "SEARCH"
        else:
            return 170, 240, "SEARCH"


# ==============================================================================
# 7. RANDOM MIX OPPONENT
# ==============================================================================
class RandomMixOpponent(BaseFSMOpponent):
    """Uniform stochastic sampling across all 5 benchmark strategies per match."""

    def __init__(self, config: RobotClassConfig) -> None:
        super().__init__(config, name="RANDOM_MIX")
        self.strategies = [
            AggressiveCharger(config),
            DefensiveSweeper(config),
            RandomFlanker(config),
            BaitAndSwitch(config),
            JuggernautPush(config),
        ]
        self.current_strategy: BaseFSMOpponent = random.choice(self.strategies)

    def reset(self) -> None:
        super().reset()
        self.current_strategy = random.choice(self.strategies)
        self.current_strategy.reset()

    def compute_action(self, obs: Dict[str, Any]) -> Tuple[int, int, str]:
        action = self.current_strategy.compute_action(obs)
        self.state = self.current_strategy.state
        return action

    def _opening_step(self, obs: Dict[str, Any]) -> Tuple[int, int, str]:
        return self.current_strategy._opening_step(obs)

    def _strategy_step(self, obs: Dict[str, Any]) -> Tuple[int, int, str]:
        return self.current_strategy._strategy_step(obs)

    @property
    def strategy_name(self) -> str:
        return getattr(self.current_strategy, "strategy_name", "RANDOM_MIX")


# ==============================================================================
# 8. STRATEGY REGISTRY
# ==============================================================================
class _StrategyRegistry(dict):
    """Case-insensitive dictionary for opponent strategy lookup."""

    def __getitem__(self, key: str) -> Any:
        if isinstance(key, str) and key not in self:
            ukey = key.upper()
            if ukey in self:
                return super().__getitem__(ukey)
        return super().__getitem__(key)

    def get(self, key: str, default: Any = None) -> Any:
        if isinstance(key, str) and key not in self:
            ukey = key.upper()
            if ukey in self:
                return super().get(ukey, default)
        return super().get(key, default)


CANONICAL_OPPONENT_STRATEGIES: Tuple[str, ...] = (
    "AGGRESSIVE_CHARGER",
    "DEFENSIVE_SWEEPER",
    "RANDOM_FLANKER",
    "BAIT_AND_SWITCH",
    "JUGGERNAUT_PUSH",
    "RANDOM_MIX",
)

OPPONENT_REGISTRY = _StrategyRegistry({
    "AGGRESSIVE_CHARGER": AggressiveCharger,
    "DEFENSIVE_SWEEPER": DefensiveSweeper,
    "RANDOM_FLANKER": RandomFlanker,
    "BAIT_AND_SWITCH": BaitAndSwitch,
    "JUGGERNAUT_PUSH": JuggernautPush,
    "RANDOM_MIX": RandomMixOpponent,
})
