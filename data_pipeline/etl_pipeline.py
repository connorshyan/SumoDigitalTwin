"""Sim-to-Real ETL Pipeline Module.

Enforces strict schema isolation across raw, processed, agent, and observer datasets,
validating data hygiene, type casting, and ensuring zero privileged coordinate leakage.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Union

import numpy as np
import pandas as pd

from config.config import (
    AGENT_DATA_DIR,
    KIT_1KG_CONFIG,
    MEGA_3KG_CONFIG,
    OBSERVER_DATA_DIR,
    PROCESSED_DATA_DIR,
    SIMULATION_DT_S,
)


# ==============================================================================
# 1. SIM-TO-REAL ETL PIPELINE & SCHEMA ENFORCEMENT
# ==============================================================================
def process_raw_telemetry(
    raw_path: Union[str, Path, pd.DataFrame],
    base_data_dir: Optional[Path] = None,
    dataset_name: Optional[str] = None,
    weight_class: str = "KIT_1KG",
) -> Dict[str, Any]:
    """Clean raw simulation logs and strictly partition into Agent and Observer datasets.

    Agent Dataset: Zero spatial coordinate features (Sim-to-Real compliant local sensor pings).
    Observer Dataset: Analytical ground truth spatial coordinates and match referee metrics.
    """
    # 1. Load DataFrame
    if isinstance(raw_path, pd.DataFrame):
        df_raw = raw_path.copy()
        if dataset_name is None:
            dataset_name = f"telemetry_{weight_class.lower()}"
    else:
        raw_file = Path(raw_path)
        if not raw_file.exists():
            raise FileNotFoundError(f"Raw telemetry file not found: {raw_file}")
        df_raw = pd.read_parquet(raw_file)
        if dataset_name is None:
            dataset_name = raw_file.stem
            if dataset_name.endswith("_raw"):
                dataset_name = dataset_name[:-4]

    # Resolve target directory paths
    proc_dir = (base_data_dir / "processed") if base_data_dir else PROCESSED_DATA_DIR
    agent_dir = (base_data_dir / "agent") if base_data_dir else AGENT_DATA_DIR
    obs_dir = (base_data_dir / "observer") if base_data_dir else OBSERVER_DATA_DIR

    for d in [proc_dir, agent_dir, obs_dir]:
        d.mkdir(parents=True, exist_ok=True)

    # 2. Data Hygiene & Validation
    initial_rows = len(df_raw)
    nan_count_initial = int(df_raw.isna().sum().sum())

    # Drop any corrupted records
    df_clean = df_raw.dropna().copy()
    cleaned_rows = len(df_clean)

    # Type casting
    if "Tick" in df_clean.columns:
        df_clean["Tick"] = df_clean["Tick"].astype(int)
    if "Timestamp_ms" in df_clean.columns:
        df_clean["Timestamp_ms"] = df_clean["Timestamp_ms"].astype(int)
    if "Action_PWM_Left" in df_clean.columns:
        df_clean["Action_PWM_Left"] = df_clean["Action_PWM_Left"].astype(int)
    if "Action_PWM_Right" in df_clean.columns:
        df_clean["Action_PWM_Right"] = df_clean["Action_PWM_Right"].astype(int)
    if "Target_Visible" in df_clean.columns:
        df_clean["Target_Visible"] = df_clean["Target_Visible"].astype(bool)

    # Identify opponent sensor columns dynamically
    opp_sensor_cols = [c for c in df_clean.columns if c.startswith("Opp_")]
    edge_sensor_cols = [c for c in df_clean.columns if c.startswith("IR_Edge_")]

    for col in opp_sensor_cols + edge_sensor_cols:
        df_clean[col] = df_clean[col].astype(float)

    # 2.1 Formal Data Quality & Sensor Domain Boundary Audit
    domain_valid = True
    for col in edge_sensor_cols:
        # Edge reflectance sensors normalized between 0.0 and 1.0 (allow minor noise tolerance [-0.05, 1.05])
        if ((df_clean[col] < -0.05) | (df_clean[col] > 1.05)).any():
            domain_valid = False
    for col in opp_sensor_cols:
        # Optical rangefinder: -1.0 (out of range/specular) or [0.0, 50.0 cm] (allow minor noise tolerance up to 55.0)
        invalid_opp = (df_clean[col] < -1.05) | ((df_clean[col] > -0.95) & (df_clean[col] < -0.05)) | (df_clean[col] > 55.0)
        if invalid_opp.any():
            domain_valid = False

    pwm_valid = True
    for pwm_col in ["Action_PWM_Left", "Action_PWM_Right"]:
        if pwm_col in df_clean.columns:
            if ((df_clean[pwm_col] < -255) | (df_clean[pwm_col] > 255)).any():
                pwm_valid = False

    # Timestamp monotonicity audit
    timestamp_uniform = True
    expected_tick_ms = int(round(SIMULATION_DT_S * 1000))
    if "Timestamp_ms" in df_clean.columns and "Match_ID" in df_clean.columns and "Bot_ID" in df_clean.columns:
        diffs = df_clean.groupby(["Match_ID", "Bot_ID"])["Timestamp_ms"].diff().dropna()
        if not diffs.empty and not (diffs == expected_tick_ms).all():
            timestamp_uniform = False

    # 3. Save Processed Master Dataset (Apache Parquet)
    processed_path = proc_dir / f"{dataset_name}_processed.parquet"
    df_clean.to_parquet(processed_path, index=False, engine="pyarrow")

    # 4. Strict Schema Partition: Agent Dataset (Hardware-Constrained Features Only)
    agent_cols = ["Match_ID", "Bot_ID", "Timestamp_ms"] + edge_sensor_cols + opp_sensor_cols + [
        "Current_State",
        "Action_PWM_Left",
        "Action_PWM_Right",
    ]
    valid_agent_cols = [c for c in agent_cols if c in df_clean.columns]
    df_agent = df_clean[valid_agent_cols].copy()
    agent_path = agent_dir / f"{dataset_name}_agent.parquet"
    df_agent.to_parquet(agent_path, index=False, engine="pyarrow")

    # 5. Strict Schema Partition: Observer Dataset (Pure Referee & Spatial Ground-Truth)
    obs_cols = [
        "Match_ID",
        "Tick",
        "Timestamp_ms",
        "Bot_ID",
        "Weight_Class",
        "Strategy_Profile",
        "Starting_Formation",
        "Pos_X",
        "Pos_Y",
        "Heading_Deg",
        "Dist_To_Center",
        "Target_Visible",
        "Current_State",
        "Match_Status",
    ]
    valid_obs_cols = [c for c in obs_cols if c in df_clean.columns]
    df_observer = df_clean[valid_obs_cols].copy()
    observer_path = obs_dir / f"{dataset_name}_observer.parquet"
    df_observer.to_parquet(observer_path, index=False, engine="pyarrow")

    # Class balance and imbalance ratio calculation
    class_distribution = {}
    imbalance_ratio = 1.0
    if "Current_State" in df_clean.columns:
        counts = df_clean["Current_State"].value_counts()
        class_distribution = counts.to_dict()
        if len(counts) > 1 and counts.min() > 0:
            imbalance_ratio = round(float(counts.max() / counts.min()), 2)

    data_quality_report = {
        "completeness_score_pct": round((1.0 - (nan_count_initial / max(1, initial_rows * max(1, len(df_raw.columns))))) * 100.0, 2),
        "domain_boundaries_valid": domain_valid and pwm_valid,
        "timestamp_monotonic_50ms": timestamp_uniform,
        "class_imbalance_ratio": imbalance_ratio,
    }

    return {
        "dataset_name": dataset_name,
        "initial_rows": initial_rows,
        "cleaned_rows": cleaned_rows,
        "nan_count": nan_count_initial,
        "processed_path": str(processed_path),
        "agent_path": str(agent_path),
        "observer_path": str(observer_path),
        "agent_features": valid_agent_cols,
        "observer_features": valid_obs_cols,
        "class_distribution": class_distribution,
        "data_quality": data_quality_report,
    }


# ==============================================================================
# 2. OBSERVER COMBAT KPI AGGREGATION & MATCH SUMMARY
# ==============================================================================
def compute_observer_kpis(
    df_obs: pd.DataFrame,
    center_zone_r: Optional[float] = None,
) -> pd.DataFrame:
    """Compute micro-combat KPIs across the Observer dataset with high-speed vectorized operations.

    Calculates:
    - Center_Control_Pct: Percentage of ticks Bot_A occupied the inner 50% radius arena core
    - TTA_ms: First target acquisition latency in milliseconds
    - TTRO_ms: Time-to-ring-out when Bot_A achieves ring-out victory
    - Duration_s / Duration_ms: Match duration
    - Win / Loss / Draw outcome attribution
    """
    if df_obs is None or df_obs.empty:
        return pd.DataFrame()

    # Dynamic Weight-Class detection if center zone radius is not explicitly provided
    if center_zone_r is None:
        weight_cls = ""
        if "Weight_Class" in df_obs.columns and not df_obs["Weight_Class"].empty:
            weight_cls = str(df_obs["Weight_Class"].iloc[0]).upper()
        if "MEGA" in weight_cls or "3KG" in weight_cls:
            center_zone_r = MEGA_3KG_CONFIG.center_zone_radius_cm
        else:
            center_zone_r = KIT_1KG_CONFIG.center_zone_radius_cm

    df_a = df_obs[df_obs["Bot_ID"] == "Bot_A"].copy()
    if df_a.empty:
        return pd.DataFrame()

    df_b = df_obs[df_obs["Bot_ID"] == "Bot_B"].copy()

    # Pre-aggregate using fast vectorized operations
    last_records_a = df_a.groupby("Match_ID").last().reset_index()
    match_counts_a = df_a.groupby("Match_ID").size().rename("Total_Ticks_A")
    center_counts_a = df_a[df_a["Dist_To_Center"] <= center_zone_r].groupby("Match_ID").size().rename("Center_Ticks_A")

    # First target acquisition
    visible_a = df_a[df_a["Target_Visible"] == True]
    first_tta = visible_a.groupby("Match_ID")["Timestamp_ms"].first().rename("TTA_ms")

    # Merge into summary table
    df_kpi = last_records_a.merge(match_counts_a, on="Match_ID", how="left")
    df_kpi = df_kpi.merge(center_counts_a, on="Match_ID", how="left").fillna({"Center_Ticks_A": 0})
    df_kpi = df_kpi.merge(first_tta, on="Match_ID", how="left")

    df_kpi["Center_Control_Pct"] = (df_kpi["Center_Ticks_A"] / df_kpi["Total_Ticks_A"]) * 100.0
    df_kpi["Duration_ms"] = df_kpi["Timestamp_ms"]
    df_kpi["Duration_s"] = df_kpi["Timestamp_ms"] / 1000.0
    df_kpi["Status"] = df_kpi["Match_Status"]
    df_kpi["TTRO_ms"] = np.where(df_kpi["Status"] == "BOT_A_WIN", df_kpi["Duration_ms"], np.nan)
    df_kpi["Strategy_A"] = df_kpi.get("Strategy_Profile", "UNKNOWN")
    df_kpi["Formation_A"] = df_kpi.get("Starting_Formation", "HEAD_ON")

    if not df_b.empty and "Strategy_Profile" in df_b.columns:
        last_b = df_b.groupby("Match_ID").first().reset_index()
        df_kpi["Strategy_B"] = last_b.get("Strategy_Profile", "UNKNOWN")
        df_kpi["Formation_B"] = last_b.get("Starting_Formation", "HEAD_ON")
    else:
        df_kpi["Strategy_B"] = "UNKNOWN"
        df_kpi["Formation_B"] = "HEAD_ON"

    return df_kpi


def extract_match_summary(df_obs: pd.DataFrame) -> pd.DataFrame:
    """Extract lightweight match-level strategy, formation, and final outcome summary."""
    if df_obs is None or df_obs.empty:
        return pd.DataFrame()

    last_ticks = df_obs.groupby(["Match_ID", "Bot_ID"]).last().reset_index()
    a_summary = last_ticks[last_ticks["Bot_ID"] == "Bot_A"][["Match_ID", "Match_Status", "Timestamp_ms", "Tick"]].rename(
        columns={"Timestamp_ms": "Duration_ms", "Tick": "Total_Ticks"}
    )
    if "Strategy_Profile" in last_ticks.columns:
        a_strat = last_ticks[last_ticks["Bot_ID"] == "Bot_A"][["Match_ID", "Strategy_Profile", "Starting_Formation"]].rename(
            columns={"Strategy_Profile": "Strategy_A", "Starting_Formation": "Formation_A"}
        )
        b_strat = last_ticks[last_ticks["Bot_ID"] == "Bot_B"][["Match_ID", "Strategy_Profile", "Starting_Formation"]].rename(
            columns={"Strategy_Profile": "Strategy_B", "Starting_Formation": "Formation_B"}
        )
        a_summary = a_summary.merge(a_strat, on="Match_ID", how="left").merge(b_strat, on="Match_ID", how="left")

    return a_summary


__all__ = [
    "process_raw_telemetry",
    "compute_observer_kpis",
    "extract_match_summary",
]
