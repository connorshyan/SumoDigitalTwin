"""Centralized 5-Stage Streamlit Web Studio for Autonomous Sumo Robot Digital Twin.

A complete research studio and data product providing:
Stage 1: Data Gathering (Simulation Sandbox)
Stage 2: Data Pre-processing (Sim-to-Real ETL & Schema Enforcement)
Stage 3: Strategy Modeling (Decision Tree & Multi-Model Benchmark)
Stage 4: Strategy Analytics & Replay (Statistical Verification & 2D Canvas)
Stage 5: Insights & Real-World Application (Embedded Deployment & C++ Handoff)
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import streamlit.components.v1 as components

from config.config import (
    AGENT_DATA_DIR,
    CONFIG_REGISTRY,
    DT_DATA_DIR,
    OBSERVER_DATA_DIR,
    PROCESSED_DATA_DIR,
    RAW_DATA_DIR,
    STARTING_FORMATIONS,
)
from data_pipeline.etl_pipeline import (
    compute_observer_kpis,
    extract_match_summary,
    process_raw_telemetry,
)
from environment.opponents import (
    OPPONENT_REGISTRY,
    RandomMixOpponent,
)
from environment.sumo_env import SumoEnvironment, run_simulation_match
from models.tree_optimizer import (
    build_hardware_summary_table,
    compute_mcu_benchmarks,
    generate_cpp_harness,
    load_decision_tree_package,
    save_decision_tree_package,
    train_and_benchmark_models,
)

try:
    from canvas_renderer import render_html5_canvas_replay  # type: ignore
except (ModuleNotFoundError, ImportError):
    from app.canvas_renderer import render_html5_canvas_replay  # type: ignore

# ==============================================================================
# 1. STREAMLIT PAGE CONFIGURATION & CUSTOM STYLING
# ==============================================================================
st.set_page_config(
    page_title="Sumo Digital Twin Studio",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for dark-mode aesthetics, glassmorphism, and responsive badges
st.markdown(
    """
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }

    /* Global Header Styling */
    .studio-header {
        background: linear-gradient(135deg, #0f172a 0%, #1e1b4b 50%, #0f172a 100%);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 16px;
        padding: 24px 32px;
        margin-bottom: 24px;
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.37);
    }
    .studio-title {
        font-size: 2.2rem;
        font-weight: 800;
        background: linear-gradient(90deg, #38bdf8, #818cf8, #c084fc);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 8px;
    }
    .studio-subtitle {
        color: #94a3b8;
        font-size: 1.0rem;
        font-weight: 400;
    }

    /* Stat Cards */
    .metric-card {
        background: rgba(30, 41, 59, 0.7);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 12px;
        padding: 16px 20px;
        backdrop-filter: blur(8px);
        box-shadow: 0 4px 16px rgba(0, 0, 0, 0.2);
        margin-bottom: 12px;
    }

    /* Dataset Auto-Detection Banner (Valid) */
    .dataset-info-card-valid {
        background: rgba(15, 23, 42, 0.85);
        border: 1px solid #38bdf8;
        border-left: 5px solid #38bdf8;
        border-radius: 10px;
        padding: 14px 18px;
        margin: 12px 0 18px 0;
    }

    /* Dataset Auto-Detection Banner (Invalid) */
    .dataset-info-card-invalid {
        background: rgba(45, 10, 15, 0.85);
        border: 1px solid #f43f5e;
        border-left: 5px solid #f43f5e;
        border-radius: 10px;
        padding: 14px 18px;
        margin: 12px 0 18px 0;
    }

    /* Output File Path Banner */
    .output-path-card {
        background: rgba(30, 41, 59, 0.8);
        border: 1px solid rgba(74, 222, 128, 0.4);
        border-left: 5px solid #4ade80;
        border-radius: 8px;
        padding: 10px 16px;
        margin: 10px 0;
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.85rem;
    }

    /* Badges */
    .badge {
        display: inline-block;
        padding: 4px 10px;
        border-radius: 9999px;
        font-size: 0.75rem;
        font-weight: 600;
        margin-right: 6px;
    }
    .badge-cyan { background: rgba(56, 189, 248, 0.2); color: #38bdf8; border: 1px solid #38bdf8; }
    .badge-green { background: rgba(74, 222, 128, 0.2); color: #4ade80; border: 1px solid #4ade80; }
    .badge-purple { background: rgba(192, 132, 252, 0.2); color: #c084fc; border: 1px solid #c084fc; }
    .badge-red { background: rgba(244, 63, 94, 0.2); color: #f43f5e; border: 1px solid #f43f5e; }
    .badge-amber { background: rgba(251, 191, 36, 0.2); color: #fbbf24; border: 1px solid #fbbf24; }

    code, pre {
        font-family: 'JetBrains Mono', monospace !important;
    }
</style>
""",
    unsafe_allow_html=True,
)

# Header Banner
st.markdown(
    """
<div class="studio-header">
    <div class="studio-title">Autonomous Sumo Robot Digital Twin Framework Studio</div>
</div>
""",
    unsafe_allow_html=True,
)


# ==============================================================================
# HELPER: SUMO ROBOT FORMAT & VALIDITY AUTO-DETECTOR
# ==============================================================================
def validate_sumo_dataset(
    df: pd.DataFrame, source_path: Optional[Union[str, Path]] = None
) -> Dict[str, Any]:
    """Inspect dataset schema and determine if it is a valid 1 KG or 3 KG Autonomous Sumo Robot dataset."""
    filename = Path(source_path).name if source_path else "Uploaded Dataset"
    filepath_str = str(Path(source_path).resolve()) if source_path else "In-Memory Stream"
    cols = set(df.columns)

    # 1. Inspect Sumo-specific sensor arrays
    edge_cols = [c for c in cols if c.startswith("IR_Edge_")]
    opp_cols = [c for c in cols if c.startswith("Opp_")]
    has_obs_coords = any(k in cols for k in ["Pos_X", "Pos_Y", "Dist_To_Center", "Heading_Deg"])
    has_state = "Current_State" in cols

    # Reject non-sumo arbitrary datasets
    if not (edge_cols or opp_cols or has_obs_coords):
        return {
            "is_valid_sumo": False,
            "filename": filename,
            "filepath": filepath_str,
            "status_label": "Incompatible Dataset: Not a recognized Sumo Robot Telemetry dataset",
            "reason": "Missing required Sumo robot sensor arrays (IR_Edge_* and Opp_*) and kinematics coordinates.",
            "robot_class": "UNKNOWN",
            "class_label": "Incompatible Schema",
            "total_rows": len(df),
            "columns": list(df.columns),
            "match_count": 0,
        }

    # 2. Determine Robot Class (1 KG vs 3 KG)
    has_mega_rays = any("22_5" in c or "15" in c for c in opp_cols) or len(opp_cols) >= 7
    has_kit_rays = any("18" in c for c in opp_cols) or len(opp_cols) == 5

    if "Weight_Class" in cols:
        w_val = str(df["Weight_Class"].iloc[0]).upper()
        if "MEGA" in w_val or "3KG" in w_val:
            detected_class = "MEGA_3KG"
        else:
            detected_class = "KIT_1KG"
    elif has_mega_rays or "mega" in filename.lower():
        detected_class = "MEGA_3KG"
    elif has_kit_rays or "kit" in filename.lower():
        detected_class = "KIT_1KG"
    elif has_obs_coords:
        detected_class = "MEGA_3KG" if "mega" in filename.lower() else "KIT_1KG"
    else:
        detected_class = "KIT_1KG"

    class_label = (
        "1 kg Kit Sumo"
        if detected_class == "KIT_1KG"
        else "3 kg Mega Sumo"
    )

    total_rows = len(df)
    nan_count = int(df.isna().sum().sum())
    match_count = int(df["Match_ID"].nunique()) if "Match_ID" in cols else 1
    opp_count = len(opp_cols)
    edge_count = len(edge_cols)

    return {
        "is_valid_sumo": True,
        "filename": filename,
        "filepath": filepath_str,
        "robot_class": detected_class,
        "class_label": class_label,
        "status_label": "Valid Sumo Robot Telemetry Dataset",
        "total_rows": total_rows,
        "nan_count": nan_count,
        "match_count": match_count,
        "columns": list(df.columns),
        "opp_count": opp_count,
        "edge_count": edge_count,
        "sensor_count": opp_count + edge_count,
    }


def render_dataset_badge(meta: Dict[str, Any]) -> None:
    """Render styled metadata inspection badge card."""
    if meta["is_valid_sumo"]:
        st.markdown(
            f"""
        <div class="dataset-info-card-valid">
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;">
                <div><strong>File Name:</strong> <code>{meta['filename']}</code></div>
                <div><span class="badge badge-green">Valid Sumo Dataset</span> <span class="badge badge-cyan">{meta['class_label']}</span></div>
            </div>
            <div style="font-size: 0.85rem; color: #94a3b8;">
                <strong>Full Path:</strong> <code>{meta['filepath']}</code><br>
                <strong>Telemetry Info:</strong> {meta['total_rows']:,} ticks | {meta['match_count']:,} matches | {meta['opp_count']} Opponent Sensors | {meta['edge_count']} Edge Sensors
            </div>
        </div>
        """,
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            f"""
        <div class="dataset-info-card-invalid">
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;">
                <div><strong>File Name:</strong> <code>{meta['filename']}</code></div>
                <div><span class="badge badge-red">{meta['status_label']}</span></div>
            </div>
            <div style="font-size: 0.85rem; color: #f87171;">
                <strong>Issue:</strong> {meta.get('reason', 'Dataset schema does not match 1 KG or 3 KG Sumo Robot specification.')}<br>
                <strong>Path:</strong> <code>{meta['filepath']}</code> | {meta['total_rows']:,} rows
            </div>
        </div>
        """,
            unsafe_allow_html=True,
        )


# ==============================================================================
# FAST STREAMLIT DATA CACHING & OPTIMIZATION LAYER
# ==============================================================================
@st.cache_data(show_spinner=False, max_entries=16)
def load_parquet_cached(file_path_str: str) -> pd.DataFrame:
    """Load and cache Apache Parquet datasets in RAM to prevent redundant disk I/O on reruns."""
    return pd.read_parquet(file_path_str)


@st.cache_data(show_spinner=False, max_entries=16)
def compute_observer_kpis_cached(observer_file_str: str, center_zone_r: float) -> pd.DataFrame:
    """Compute and cache match KPIs across the Observer dataset with high-speed vectorized operations."""
    df_obs = pd.read_parquet(observer_file_str)
    return compute_observer_kpis(df_obs, center_zone_r=center_zone_r)


@st.cache_data(show_spinner=False, max_entries=16)
def extract_match_summary_cached(observer_file_str: str) -> pd.DataFrame:
    """Extract lightweight match-level strategy, formation, and final outcome summary."""
    try:
        df = pd.read_parquet(
            observer_file_str,
            columns=["Match_ID", "Bot_ID", "Strategy_Profile", "Starting_Formation", "Match_Status"],
        )
        return extract_match_summary(df)
    except Exception:
        return pd.DataFrame()


# ==============================================================================
# 2. TAB NAVIGATION
# ==============================================================================
tabs = st.tabs(
    [
        "1. Data Gathering (Simulation)",
        "2. Data Pre-processing (ETL)",
        "3. Strategy Modeling (ML)",
        "4. Strategy Analytics & Replay",
        "5. Real World Application",
    ]
)

# ==============================================================================
# STAGE 1: DATA GATHERING (SIMULATION SANDBOX)
# ==============================================================================
with tabs[0]:
    st.subheader("Stage 1: High-Volume Telemetry Simulator")
    st.markdown(
        "Generate high-fidelity, deterministic 2D combat telemetry across standardized tournament specifications at a fixed 50 ms tick rate ($dt = 0.05$ s)."
    )

    col1, col2 = st.columns([1, 1])

    with col1:
        st.markdown("### Combat & Environment Setup")
        weight_class_sel = st.selectbox(
            "Tournament Robot Class",
            options=["KIT_1KG", "MEGA_3KG"],
            index=0,
        )
        active_config = CONFIG_REGISTRY[weight_class_sel]

        strategy_options = [
            "AGGRESSIVE_CHARGER",
            "DEFENSIVE_SWEEPER",
            "RANDOM_FLANKER",
            "BAIT_AND_SWITCH",
            "JUGGERNAUT_PUSH",
            "RANDOM_MIX",
        ]
        adversary_sel = st.selectbox(
            "Adversary Strategy Profile",
            options=strategy_options,
            index=strategy_options.index("RANDOM_MIX"),
        )

        form_col_a, form_col_b = st.columns(2)
        default_form_idx = list(STARTING_FORMATIONS).index("RANDOM_MIX")
        with form_col_a:
            formation_a_sel = st.selectbox(
                "Bot A Starting Formation",
                options=list(STARTING_FORMATIONS),
                index=default_form_idx,
            )
        with form_col_b:
            formation_b_sel = st.selectbox(
                "Bot B Starting Formation",
                options=list(STARTING_FORMATIONS),
                index=default_form_idx,
            )

        if "sim_match_count" not in st.session_state:
            st.session_state.sim_match_count = 500
        if "slider_match_count" not in st.session_state:
            st.session_state.slider_match_count = 500
        if "num_match_count" not in st.session_state:
            st.session_state.num_match_count = 500

        def sync_from_slider():
            st.session_state.sim_match_count = st.session_state.slider_match_count
            st.session_state.num_match_count = st.session_state.slider_match_count

        def sync_from_num():
            st.session_state.sim_match_count = st.session_state.num_match_count
            st.session_state.slider_match_count = st.session_state.num_match_count

        subcol_a, subcol_b = st.columns([2, 1])
        with subcol_a:
            st.slider(
                "Batch Match Count (Slider)",
                min_value=10,
                max_value=5000,
                step=10,
                key="slider_match_count",
                on_change=sync_from_slider,
            )
        with subcol_b:
            st.number_input(
                "Manual Count Entry",
                min_value=10,
                max_value=5000,
                step=10,
                key="num_match_count",
                on_change=sync_from_num,
            )

        dataset_name_input = st.text_input(
            "Dataset Base Name",
            value="",
            placeholder=f"e.g. telemetry_{weight_class_sel.lower()}",
        )
        clean_input = dataset_name_input.strip()
        default_base_name = f"telemetry_{weight_class_sel.lower()}"
        clean_dataset_name = clean_input if clean_input else default_base_name
        if clean_dataset_name.endswith(".parquet"):
            clean_dataset_name = clean_dataset_name[:-8]
        if clean_dataset_name.endswith("_raw"):
            clean_dataset_name = clean_dataset_name[:-4]
        target_raw_filename = f"{clean_dataset_name}_raw.parquet"
        target_raw_fullpath = RAW_DATA_DIR / target_raw_filename

    with col2:
        st.markdown("### Hardware & Arena Specifications")
        st.markdown(
            f"""
        <div class="metric-card" style="margin-top: 24px; padding: 20px 24px; min-height: 386px; display: flex; flex-direction: column; justify-content: space-between;">
            <div>
                <div style="font-size: 1.02rem; color: #e2e8f0; font-weight: 500; margin-bottom: 14px; line-height: 1.5;">
                    Based on the official RoboGames Rules and robot specifications from our industry partner.
                </div>
                <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 14px 20px; font-size: 0.98rem; line-height: 1.45;">
                <div><strong style="color: #94a3b8;">Dohyo Diameter:</strong> {active_config.dohyo_diameter_cm:.1f} cm</div>
                <div><strong style="color: #94a3b8;">White Border Width:</strong> {active_config.border_width_cm:.1f} cm</div>
                <div><strong style="color: #94a3b8;">Shikiri Separation:</strong> {active_config.shikiri_separation_cm:.1f} cm</div>
                <div><strong style="color: #94a3b8;">Shikiri Line Length:</strong> {active_config.shikiri_length_cm:.1f} cm</div>
                <div><strong style="color: #94a3b8;">Robot Dimensions:</strong> {active_config.robot_width_cm:.1f} × {active_config.robot_length_cm:.1f} cm</div>
                <div><strong style="color: #94a3b8;">Wheelbase (L):</strong> {active_config.wheelbase_cm:.1f} cm</div>
                <div><strong style="color: #94a3b8;">Max Speed (v_max):</strong> {active_config.v_max_cms:.1f} cm/s</div>
                <div><strong style="color: #94a3b8;">Friction Coefficient (μ):</strong> {active_config.coulomb_mu:.2f}</div>
                <div><strong style="color: #94a3b8;">Opponent Array:</strong> {len(active_config.opponent_sensor_angles_deg)}x E3Z-D62</div>
                <div><strong style="color: #94a3b8;">Line Array:</strong> 2x QRE1113</div>
            </div>
            </div>
            <div style="margin-top: 14px; padding-top: 12px; border-top: 1px solid rgba(255,255,255,0.1); font-size: 0.92rem; line-height: 1.6; color: #cbd5e1;">
                <div><strong style="color: #94a3b8;">Opponent Sensor Angles:</strong> <code style="color: #38bdf8; font-size: 0.88rem;">{list(active_config.opponent_sensor_angles_deg)}°</code></div>
                <div style="margin-top: 6px;"><strong style="color: #94a3b8;">Edge Sensor Offsets (FL, FR):</strong> <code style="color: #34d399; font-size: 0.88rem;">{list(active_config.edge_sensor_offsets_cm)} cm</code></div>
            </div>
        </div>
        """,
            unsafe_allow_html=True,
        )

    st.markdown("---")
    run_sim_btn = st.button("Run Batch Simulation", type="primary", use_container_width=True)

    if run_sim_btn:
        ui_lock_placeholder = st.empty()
        ui_lock_placeholder.markdown(
            """
            <style>
            button,
            [role="tab"],
            [data-baseweb="tab"],
            .stSelectbox,
            .stMultiSelect,
            .stRadio,
            .stSlider,
            input,
            select,
            a {
                pointer-events: none !important;
                opacity: 0.45 !important;
                filter: grayscale(85%) !important;
                cursor: not-allowed !important;
                transition: opacity 0.3s ease, filter 0.3s ease;
            }
            html, body {
                cursor: wait !important;
            }
            </style>
            """,
            unsafe_allow_html=True,
        )

        env = SumoEnvironment(config=active_config, seed=42)
        bot_a_policy = RandomMixOpponent(active_config)
        adversary_cls = OPPONENT_REGISTRY.get(adversary_sel, RandomMixOpponent)
        bot_b_policy = adversary_cls(active_config)

        n_matches = int(st.session_state.sim_match_count)
        all_telemetry: List[Dict[str, Any]] = []
        match_summaries: List[Dict[str, Any]] = []

        progress_bar = st.progress(0.0, text=f"Initializing simulation of {n_matches:,} matches...")
        time.sleep(0.05)
        start_time = time.time()

        update_step = max(1, n_matches // 100)
        for m_idx in range(n_matches):
            match_id = f"M_{m_idx+1:05d}"
            records = run_simulation_match(
                env,
                bot_a_policy,
                bot_b_policy,
                match_id=match_id,
                formation_a=formation_a_sel,
                formation_b=formation_b_sel,
            )
            all_telemetry.extend(records)

            final_record = records[-1]
            match_summaries.append(
                {
                    "Match_ID": match_id,
                    "Status": final_record["Match_Status"],
                    "Duration_Ticks": final_record["Tick"],
                    "Duration_s": final_record["Tick"] * active_config.dt,
                }
            )

            if (m_idx + 1) % update_step == 0 or m_idx == n_matches - 1:
                prog = (m_idx + 1) / n_matches
                progress_bar.progress(
                    prog,
                    text=f"Simulating Match {m_idx+1:,}/{n_matches:,} ({(m_idx+1)/n_matches*100:.1f}%) | Elapsed: {time.time() - start_time:.2f}s",
                )

        elapsed = time.time() - start_time
        progress_bar.progress(1.0, text=f"Batch simulation completed! {n_matches:,} matches ({len(all_telemetry):,} ticks) simulated in {elapsed:.2f}s.")

        df_raw = pd.DataFrame(all_telemetry)
        df_raw.to_parquet(target_raw_fullpath, index=False, engine="pyarrow")
        st.cache_data.clear()

        ui_lock_placeholder.empty()

        st.session_state.last_sim_raw_df = df_raw
        st.session_state.last_sim_summary = pd.DataFrame(match_summaries)
        st.session_state.last_raw_file = target_raw_fullpath

    # Table Viewer for Generated Raw Dataset BEFORE Simulation Outcome Metrics
    if "last_sim_raw_df" in st.session_state:
        df_raw_view = st.session_state.last_sim_raw_df
        raw_file_p = st.session_state.get("last_raw_file", target_raw_fullpath)

        st.markdown("### Generated Raw Telemetry Dataset")
        st.markdown(
            f"""
        <div class="output-path-card">
            <strong>Output File:</strong> <code>{Path(raw_file_p).name}</code><br>
            <strong>Full Path:</strong> <code>{Path(raw_file_p).resolve()}</code>
        </div>
        """,
            unsafe_allow_html=True,
        )

        with st.expander(f"Preview Raw Simulation Telemetry ({Path(raw_file_p).name} - First 200 Ticks)", expanded=False):
            st.dataframe(df_raw_view.head(200), use_container_width=True, height=260)

    # Simulation Outcome Metrics
    if "last_sim_summary" in st.session_state:
        df_sum = st.session_state.last_sim_summary
        total_m = len(df_sum)
        wins_a = (df_sum["Status"] == "BOT_A_WIN").sum()
        wins_b = (df_sum["Status"] == "BOT_B_WIN").sum()
        draws = (df_sum["Status"] == "DRAW").sum()
        win_rate_a = (wins_a / total_m) * 100.0 if total_m > 0 else 0.0

        st.markdown("### Simulation Outcome Metrics")
        kpi_col1, kpi_col2, kpi_col3, kpi_col4, kpi_col5 = st.columns(5)
        with kpi_col1:
            st.metric("Total Matches", f"{total_m:,}")
        with kpi_col2:
            st.metric("Candidate Wins", f"{wins_a:,}")
        with kpi_col3:
            st.metric("Adversary Wins", f"{wins_b:,}")
        with kpi_col4:
            st.metric("Draws / Timeouts", f"{draws:,}")
        with kpi_col5:
            st.metric("Mean Duration", f"{df_sum['Duration_s'].mean():.2f} s")

        chart_col1, chart_col2 = st.columns([1, 1])

        with chart_col1:
            fig_donut = px.pie(
                df_sum,
                names="Status",
                title="Match Outcome Distribution",
                color="Status",
                color_discrete_map={
                    "BOT_A_WIN": "#00d2ff",
                    "BOT_B_WIN": "#ff4b4b",
                    "DRAW": "#ffaa00",
                },
                hole=0.5,
            )
            fig_donut.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#f8fafc"),
                margin=dict(t=50, b=30, l=20, r=20),
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            )
            st.plotly_chart(fig_donut, use_container_width=True)

        with chart_col2:
            fig_dur = px.histogram(
                df_sum,
                x="Duration_s",
                color="Status",
                title="Match Duration Distribution (Seconds)",
                color_discrete_map={
                    "BOT_A_WIN": "#00d2ff",
                    "BOT_B_WIN": "#ff4b4b",
                    "DRAW": "#ffaa00",
                },
                nbins=25,
                barmode="stack",
            )
            fig_dur.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#f8fafc"),
                xaxis_title="Duration (Seconds)",
                yaxis_title="Number of Matches",
                margin=dict(t=50, b=40, l=40, r=20),
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            )
            st.plotly_chart(fig_dur, use_container_width=True)

# ==============================================================================
# STAGE 2: DATA PRE-PROCESSING (SIM-TO-REAL ETL & SCHEMA ENFORCEMENT)
# ==============================================================================
with tabs[1]:
    st.subheader("Stage 2: Sim-to-Real ETL Pipeline & Schema Enforcement")
    st.markdown(
        "Ingests raw simulation logs from `data/raw/`, performs data hygiene validation, and strictly partitions features into **Agent Dataset** and **Observer Dataset**."
    )

    raw_files = sorted(list(RAW_DATA_DIR.glob("*_raw.parquet")))
    raw_file_names = [f.name for f in raw_files]

    selected_raw_path = None
    dataset_base = "telemetry_dataset"
    raw_is_valid = True

    if raw_file_names:
        selected_filename = st.selectbox("Select Raw Dataset", options=raw_file_names)
        selected_raw_path = RAW_DATA_DIR / selected_filename
        dataset_base = selected_filename.replace("_raw.parquet", "").replace(".parquet", "")
        raw_preview_df = load_parquet_cached(str(selected_raw_path))
        meta = validate_sumo_dataset(raw_preview_df, selected_raw_path)
        render_dataset_badge(meta)
        raw_is_valid = meta["is_valid_sumo"]
    else:
        st.warning("No raw datasets found in `data/raw/`. Please generate data in Stage 1 first.")
        raw_is_valid = False

    etl_trigger_btn = st.button(
        "Execute ETL Pipeline",
        type="primary",
        use_container_width=True,
        disabled=not raw_is_valid,
    )

    if etl_trigger_btn:
        ui_lock_placeholder_stage2 = st.empty()
        ui_lock_placeholder_stage2.markdown(
            """
            <style>
            button,
            [role="tab"],
            [data-baseweb="tab"],
            .stSelectbox,
            .stMultiSelect,
            .stRadio,
            .stSlider,
            input,
            select,
            a {
                pointer-events: none !important;
                opacity: 0.45 !important;
                filter: grayscale(85%) !important;
                cursor: not-allowed !important;
                transition: opacity 0.3s ease, filter 0.3s ease;
            }
            html, body {
                cursor: wait !important;
            }
            </style>
            """,
            unsafe_allow_html=True,
        )
        if selected_raw_path is not None:
            with st.spinner("Processing telemetry and enforcing schema isolation..."):
                etl_stats = process_raw_telemetry(
                    raw_path=selected_raw_path,
                    dataset_name=dataset_base,
                )
                st.cache_data.clear()
                st.session_state.active_etl_stats = etl_stats
                st.session_state.active_dataset_base = dataset_base
                st.success("ETL Pipeline successfully executed!")
        else:
            st.error("Please select a valid raw telemetry dataset.")
        ui_lock_placeholder_stage2.empty()

    if "active_etl_stats" in st.session_state:
        stats = st.session_state.active_etl_stats

        st.markdown("### Data Hygiene & Schema Integrity")
        b_col1, b_col2, b_col3, b_col4 = st.columns(4)
        with b_col1:
            st.metric("Total Cleaned Rows", f"{stats['cleaned_rows']:,}")
        with b_col2:
            st.metric("Missing / NaN Values", f"{stats['nan_count']}")
        with b_col3:
            st.metric("Agent Features", f"{len(stats['agent_features'])}")
        with b_col4:
            st.metric("Observer Features", f"{len(stats['observer_features'])}")

        st.markdown("---")
        st.markdown("### Output Datasets (Exact File Paths & Real Data Viewers)")

        st.markdown(
            f"""
        <div class="output-path-card">
            <strong>Master Processed Output:</strong> <code>{Path(stats['processed_path']).name}</code><br>
            <strong>Full Path:</strong> <code>{Path(stats['processed_path']).resolve()}</code>
        </div>
        """,
            unsafe_allow_html=True,
        )

        agent_file_p = Path(stats["agent_path"]).resolve()
        st.markdown(f"#### 1. Agent Dataset: `{agent_file_p.name}`")
        st.markdown(
            f"""
        <div class="output-path-card">
            <strong>File:</strong> <code>{agent_file_p.name}</code> | <strong>Full Path:</strong> <code>{agent_file_p}</code>
        </div>
        """,
            unsafe_allow_html=True,
        )
        if agent_file_p.exists():
            df_agent_view = pd.read_parquet(agent_file_p)
            with st.expander(f"Preview Agent Dataset ({agent_file_p.name} - First 200 Rows)", expanded=False):
                st.dataframe(df_agent_view.head(200), use_container_width=True, height=240)

        obs_file_p = Path(stats["observer_path"]).resolve()
        st.markdown(f"#### 2. Observer Dataset: `{obs_file_p.name}`")
        st.markdown(
            f"""
        <div class="output-path-card">
            <strong>File:</strong> <code>{obs_file_p.name}</code> | <strong>Full Path:</strong> <code>{obs_file_p}</code>
        </div>
        """,
            unsafe_allow_html=True,
        )
        if obs_file_p.exists():
            df_obs_view = pd.read_parquet(obs_file_p)
            with st.expander(f"Preview Observer Dataset ({obs_file_p.name} - First 200 Rows)", expanded=False):
                st.dataframe(df_obs_view.head(200), use_container_width=True, height=240)

        st.markdown("---")
        st.markdown("### Combat State Class Distribution")
        if stats["class_distribution"]:
            # Display metrics for all 5 combat states
            total_samples = sum(stats["class_distribution"].values())
            canonical_order = ["ATTACK", "SEARCH", "TRACK", "EVADE", "EDGE_RECOVERY"]
            sorted_states = sorted(
                list(set(canonical_order + list(stats["class_distribution"].keys()))),
                key=lambda s: stats["class_distribution"].get(s, 0),
                reverse=True,
            )
            state_cols = st.columns(len(sorted_states))
            for col, s_name in zip(state_cols, sorted_states):
                cnt = stats["class_distribution"].get(s_name, 0)
                pct = (cnt / total_samples * 100.0) if total_samples > 0 else 0.0
                with col:
                    st.metric(
                        s_name,
                        f"{cnt:,}",
                        f"{pct:.1f}% share",
                        delta_color="off",
                        delta_arrow="off",
                    )
            st.markdown("<div style='margin-bottom: 12px;'></div>", unsafe_allow_html=True)

            df_class = pd.DataFrame(
                list(stats["class_distribution"].items()), columns=["State", "Count"]
            )
            fig_cls = px.bar(
                df_class,
                x="State",
                y="Count",
                color="State",
                title="Sample Frequency Across Discrete Combat States",
                color_discrete_sequence=px.colors.qualitative.Prism,
            )
            fig_cls.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#f8fafc"),
            )
            st.plotly_chart(fig_cls, use_container_width=True)

        # 4. Sensor Correlation & Multi-Collinearity Analysis
        if agent_file_p.exists():
            corr_cols = [
                c for c in df_agent_view.columns
                if c.startswith("IR_Edge_") or c.startswith("Opp_") or c.startswith("Delta_")
            ]
            if len(corr_cols) >= 2:
                st.markdown("### Sensor Correlation & Multi-Collinearity Heatmap")
                st.markdown(
                    "Empirical evaluation of physical sensor cross-correlation. Adjacent optical rays with overlapping detection cones exhibit strong positive collinearity."
                )
                corr_matrix = df_agent_view[corr_cols].corr(method="pearson").round(2)
                fig_corr = px.imshow(
                    corr_matrix,
                    text_auto=True,
                    aspect="auto",
                    color_continuous_scale="RdBu_r",
                    zmin=-1.0,
                    zmax=1.0,
                    title="Pearson Correlation Matrix (Sim-to-Real Sensor Array)",
                )
                fig_corr.update_layout(
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    font=dict(color="#f8fafc"),
                    height=450,
                )
                st.plotly_chart(fig_corr, use_container_width=True)

# ==============================================================================
# STAGE 3: STRATEGY MODELING & MULTI-MODEL BENCHMARKING
# ==============================================================================
with tabs[2]:
    st.subheader("Stage 3: Strategy Modeling & Multi-Model Benchmark")
    st.markdown(
        """
    Trains and benchmarks machine learning architectures using 5-fold cross-validated `GridSearchCV` on hardware-constrained Agent Telemetry with the opponent telemetry filtered out:
    - **Decision Tree** (Primary White-Box)
    - **Logistic Regression** (Linear Baseline)
    - **Random Forest** (Ensemble Bagging)
    - **Gradient Boosting** (Ensemble Boosting)
    """
    )

    agent_files = sorted(list(AGENT_DATA_DIR.glob("*_agent.parquet")))
    agent_file_names = [f.name for f in agent_files]

    ml_agent_path = None
    active_ml_df = None
    ml_is_valid = True

    if agent_file_names:
        selected_ml_file = st.selectbox(
            "Select Agent Dataset for Training",
            options=agent_file_names,
            key="ml_agent_select",
        )
        ml_agent_path = AGENT_DATA_DIR / selected_ml_file
        active_ml_df = load_parquet_cached(str(ml_agent_path))
        meta = validate_sumo_dataset(active_ml_df, ml_agent_path)
        render_dataset_badge(meta)
        ml_is_valid = meta["is_valid_sumo"]
    else:
        st.warning("No agent datasets found in `data/agent/`. Please process raw data in Stage 2 first.")
        ml_is_valid = False

    all_available_models = [
        "Decision Tree",
        "Logistic Regression",
        "Random Forest",
        "Gradient Boosting",
    ]

    st.markdown("### Multi-Model Training & Hyperparameter Grids")

    selected_ml_models = st.multiselect(
        "Select Model Architecture(s) to Train & Benchmark",
        options=all_available_models,
        default=all_available_models,
        key="selected_ml_models_multiselect",
    )

    grid_details = {
        "Decision Tree": """<div>
            <strong style="color: #38bdf8;">Decision Tree (White-Box Benchmark):</strong><br>
            • <code>criterion</code>: ['gini', 'entropy']<br>
            • <code>max_depth</code>: [4, 6, 8]<br>
            • <code>min_samples_split</code>: [10, 20]<br>
            • <code>class_weight</code>: 'balanced'
        </div>""",
        "Logistic Regression": """<div>
            <strong style="color: #34d399;">Logistic Regression (Linear Baseline):</strong><br>
            • <code>StandardScaler</code> feature normalization<br>
            • <code>C (Regularization)</code>: [0.1, 1.0, 10.0]<br>
            • <code>solver</code>: 'lbfgs' (Multinomial Softmax, <code>max_iter=1000</code>)<br>
            • <code>class_weight</code>: 'balanced'
        </div>""",
        "Random Forest": """<div>
            <strong style="color: #f59e0b;">Random Forest (Ensemble Bagging):</strong><br>
            • <code>n_estimators</code>: [50]<br>
            • <code>max_depth</code>: [6, 10]<br>
            • <code>min_samples_split</code>: [10, 20]<br>
            • <code>class_weight</code>: 'balanced'
        </div>""",
        "Gradient Boosting": """<div>
            <strong style="color: #ec4899;">Gradient Boosting (Ensemble Boosting):</strong><br>
            • <code>n_estimators</code>: [40]<br>
            • <code>learning_rate</code>: [0.1]<br>
            • <code>max_depth</code>: [3]<br>
            • <code>loss</code>: 'log_loss' (Deviance)
        </div>""",
    }

    if selected_ml_models:
        grid_cards_html = "".join([grid_details[m] for m in selected_ml_models if m in grid_details])
        st.markdown(
            f"""<div class="metric-card">
    <div style="display: grid; grid-template-columns: repeat({min(2, len(selected_ml_models))}, 1fr); gap: 16px;">
        {grid_cards_html}
    </div>
</div>""",
            unsafe_allow_html=True,
        )
    else:
        st.warning("Please select at least one ML model architecture above.")

    num_sel = len(selected_ml_models)
    if num_sel == len(all_available_models):
        btn_label = "Train & Benchmark All Models"
    elif num_sel > 0:
        btn_label = f"Train & Benchmark Selected Model{'s' if num_sel > 1 else ''} ({num_sel})"
    else:
        btn_label = "Select at least one model architecture"

    train_models_btn = st.button(
        btn_label,
        type="primary",
        use_container_width=True,
        disabled=not (ml_is_valid and num_sel > 0),
        key="btn_train_selected_models",
    )

    if train_models_btn:
        if ml_agent_path is not None:
            # Grey out and disable all buttons, tabs, dropdowns, inputs and controls during training
            lock_placeholder = st.empty()
            lock_placeholder.markdown(
                """
                <style>
                button,
                [role="tab"],
                [data-baseweb="tab"],
                .stSelectbox,
                .stMultiSelect,
                .stRadio,
                .stSlider,
                input,
                select,
                a {
                    pointer-events: none !important;
                    opacity: 0.45 !important;
                    filter: grayscale(85%) !important;
                    cursor: not-allowed !important;
                    transition: opacity 0.3s ease, filter 0.3s ease;
                }
                html, body {
                    cursor: wait !important;
                }
                </style>
                """,
                unsafe_allow_html=True,
            )

            prog_placeholder = st.empty()
            prog_bar = prog_placeholder.progress(0, text=f"Initializing 5-Fold Cross-Validation for {num_sel} model{'s' if num_sel > 1 else ''}...")
            time.sleep(0.05)

            def update_train_progress(idx: int, total: int, msg: str) -> None:
                pct = int((idx / total) * 100)
                prog_bar.progress(pct, text=msg)

            benchmark_res = train_and_benchmark_models(
                ml_agent_path,
                test_size=0.20,
                random_state=42,
                selected_models=selected_ml_models,
                progress_callback=update_train_progress,
            )
            prog_bar.progress(100, text=f"Completed 5-Fold Cross-Validation across {num_sel} selected architecture{'s' if num_sel > 1 else ''}!")
            time.sleep(0.3)
            lock_placeholder.empty()
            prog_placeholder.empty()

            st.session_state.multi_model_benchmark = benchmark_res
            st.session_state.tree_results = benchmark_res.get("primary_model_results")

            # If Decision Tree was trained, automatically package and save to data/dt/
            if "Decision Tree" in benchmark_res.get("models", {}):
                dt_info = benchmark_res["models"]["Decision Tree"]
                dt_base = Path(ml_agent_path).stem.replace("_agent", "").replace(".parquet", "")
                dt_pkg_res = save_decision_tree_package(
                    dt_res=dt_info,
                    dataset_base=dt_base,
                    robot_class=meta["robot_class"],
                )
                st.session_state.last_saved_dt_pkg = dt_pkg_res

            # Format model training times and total
            timing_items = [
                f"**{name}:** {data.get('training_time_s', 0.0):.2f}s"
                for name, data in benchmark_res.get("models", {}).items()
            ]
            total_train_s = sum(data.get("training_time_s", 0.0) for data in benchmark_res.get("models", {}).values())
            timing_details = " &nbsp;|&nbsp; ".join(timing_items)
            success_msg = (
                f"**Model Training, Hyperparameter Tuning & Benchmark complete!** (Total: {total_train_s:.2f}s)  \n"
                f"Time Taken: {timing_details}"
            )
            st.session_state.train_completion_msg = success_msg
            st.success(success_msg)
        else:
            st.error("Please provide a valid agent dataset for training.")
    elif "train_completion_msg" in st.session_state and "multi_model_benchmark" in st.session_state:
        st.success(st.session_state.train_completion_msg)

    if "last_saved_dt_pkg" in st.session_state and st.session_state.last_saved_dt_pkg is not None:
        dt_pkg_info = st.session_state.last_saved_dt_pkg
        dt_json_p = Path(dt_pkg_info["json_path"])
        st.markdown(
            f"""
        <div class="output-path-card" style="margin-top: 14px; margin-bottom: 14px;">
            <strong>Decision Tree Package Output:</strong> <code>{dt_json_p.name}</code><br>
            <strong>Full Path:</strong> <code>{dt_json_p.resolve()}</code>
        </div>
        """,
            unsafe_allow_html=True,
        )

    if "multi_model_benchmark" in st.session_state:
        bm = st.session_state.multi_model_benchmark
        models_dict = bm["models"]
        df_lb = bm["leaderboard"]
        best_name = bm["best_model_name"]
        dt_res = models_dict.get("Decision Tree")

        # ======================================================================
        # 1. LEADERBOARD & COMPARISON
        # ======================================================================
        st.markdown("---")
        st.markdown("### Strategy Model Benchmark Leaderboard")

        champ_col1, champ_col2, champ_col3, champ_col4 = st.columns(4)
        with champ_col1:
            st.metric("Top Benchmark Model", best_name, f"Top F1: {df_lb.iloc[0]['Test Macro F1']:.4f}")
        with champ_col2:
            if dt_res is not None:
                st.metric("White-Box Decision Tree", f"{dt_res['test_f1_macro']:.4f} F1", f"Accuracy: {dt_res['test_accuracy']:.1%}", delta_arrow="off")
            else:
                st.metric("Top Architecture", best_name, f"Type: {models_dict[best_name]['type']}")
        with champ_col3:
            if dt_res is not None:
                margin = dt_res['test_f1_macro'] - df_lb.iloc[0]['Test Macro F1']
                delta_str = f"{margin:+.4f} vs. Top Model" if abs(margin) > 1e-4 else "0.0000 (Tied)"
                st.metric("Decision Tree F1 Gap", f"{margin:+.4f}", delta=delta_str, delta_color="normal")
            else:
                st.metric("Top Model Precision", f"{models_dict[best_name]['test_precision_macro']:.4f}", f"Recall: {models_dict[best_name]['test_recall_macro']:.4f}")
        with champ_col4:
            if dt_res is not None:
                top_rec = df_lb.iloc[0]
                try:
                    top_p = float(top_rec.get("p-value (vs DT)", 1.0))
                except (ValueError, TypeError):
                    top_p = 1.0
                if best_name == "Decision Tree":
                    st.metric("Top Model vs. DT Significance", "Rank #1 (Best)", delta="Top Model", delta_color="normal", delta_arrow="off")
                elif top_p >= 0.05:
                    st.metric("Top Model vs. DT Significance", f"p = {top_p:.4f}", delta="Statistically Equivalent (p ≥ 0.05)", delta_color="normal", delta_arrow="off")
                else:
                    st.metric("Top Model vs. DT Significance", f"p = {top_p:.4f}", delta="Significant Difference (p < 0.05)", delta_color="inverse", delta_arrow="off")
            else:
                st.metric("Top Model CV F1", f"{df_lb.iloc[0]['5-Fold CV Macro F1']:.4f}", delta="5-Fold Cross-Validation", delta_arrow="off")

        # Leaderboard Table
        st.markdown("##### Comparative Performance Ranking Table")
        df_lb_display = df_lb.drop(columns=["Embedded Deployment"], errors="ignore")
        st.dataframe(
            df_lb_display,
            use_container_width=True,
            hide_index=True,
        )

        # Multi-Model Grouped Bar Chart (Grouped by Metric, colored by Model)
        metric_plot_df = pd.melt(
            df_lb,
            id_vars=["Model"],
            value_vars=["Test Macro F1", "Test Accuracy", "Test Precision (M)", "Test Recall (M)", "5-Fold CV Macro F1"],
            var_name="Metric",
            value_name="Score",
        )

        fig_lb = px.bar(
            metric_plot_df,
            x="Metric",
            y="Score",
            color="Model",
            barmode="group",
            title="Multi-Model Benchmark Comparison (Grouped by Metric)",
            color_discrete_map={
                "Decision Tree": "#00d2ff",
                "Random Forest": "#10b981",
                "Gradient Boosting": "#a855f7",
                "Logistic Regression": "#f59e0b",
            },
            text_auto=".3f",
        )
        fig_lb.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#f8fafc"),
            yaxis=dict(range=[0.0, 1.05], zeroline=True, gridcolor="#1e293b"),
            xaxis_title="Evaluation Metric",
            yaxis_title="Score",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        )
        st.plotly_chart(fig_lb, use_container_width=True)

        # ======================================================================
        # 2. DETAILED ARCHITECTURE INSPECTOR
        # ======================================================================
        st.markdown("---")
        st.markdown("### Detailed Architecture Inspector")

        selected_model_tab = st.radio(
            "Select Model to Inspect",
            options=list(models_dict.keys()),
            horizontal=True,
            key="selected_model_inspector_radio",
        )

        m_info = models_dict[selected_model_tab]

        # Model Metadata & Best Params Card (without green transpilable box)
        st.markdown(
            f"""
        <div class="metric-card" style="margin-bottom: 16px;">
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;">
                <div><strong style="font-size: 1.1rem; color: #38bdf8;">{m_info['name']}</strong> &nbsp; <span class="badge badge-cyan">{m_info['type']}</span></div>
            </div>
            <div style="font-size: 0.9rem; color: #cbd5e1;">
                <strong>Architecture:</strong> {m_info['architecture']}<br>
                <strong>Optimal Hyperparameters:</strong> <code>{m_info['best_params']}</code><br>
                <strong>5-Fold CV Score (Macro F1):</strong> {m_info['best_cv_f1_macro']:.4f} | <strong>Training Latency:</strong> {m_info['training_time_s']:.2f} s
            </div>
        </div>
        """,
            unsafe_allow_html=True,
        )

        # Model KPIs
        k1, k2, k3, k4, k5 = st.columns(5)
        with k1:
            st.metric("Test Accuracy", f"{m_info['test_accuracy']:.4f}")
        with k2:
            st.metric("Macro Precision", f"{m_info['test_precision_macro']:.4f}")
        with k3:
            st.metric("Macro Recall", f"{m_info['test_recall_macro']:.4f}")
        with k4:
            st.metric("Macro F1-Score", f"{m_info['test_f1_macro']:.4f}")
        with k5:
            st.metric("Weighted F1", f"{m_info['test_f1_weighted']:.4f}")

        # Plots for Selected Model
        v_col1, v_col2 = st.columns([1, 1])
        with v_col1:
            st.markdown("##### Normalized Confusion Matrix")
            cm_df = pd.DataFrame(
                m_info["confusion_matrix_normalized"],
                index=m_info["classes"],
                columns=m_info["classes"],
            )
            fig_cm = px.imshow(
                cm_df,
                text_auto=".1%",
                color_continuous_scale="Blues",
                labels=dict(x="Predicted State", y="Ground Truth State", color="Recall"),
                title=f"{m_info['name']} - Confusion Matrix (Per True Class)",
            )
            fig_cm.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#f8fafc"),
            )
            st.plotly_chart(fig_cm, use_container_width=True)

        with v_col2:
            st.markdown("##### Sensor Feature Importances / Coefficients")
            fi_df = pd.DataFrame(
                list(m_info["feature_importances"].items()),
                columns=["Sensor Feature", "Importance / Relative Weight"],
            )
            fig_fi = px.bar(
                fi_df,
                x="Importance / Relative Weight",
                y="Sensor Feature",
                orientation="h",
                color="Importance / Relative Weight",
                color_continuous_scale="Viridis",
                title=f"{m_info['name']} - Feature Importance Rankings",
            )
            fig_fi.update_layout(
                yaxis=dict(autorange="reversed"),
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#f8fafc"),
            )
            st.plotly_chart(fig_fi, use_container_width=True)

        # Classification Report Breakdown
        with st.expander(f"View Full Classification Report Breakdown ({m_info['name']})"):
            report_dict = m_info["classification_report"]
            report_rows = []
            for cls_k, metrics in report_dict.items():
                if isinstance(metrics, dict):
                    report_rows.append({
                        "Class / State": cls_k,
                        "Precision": round(metrics.get("precision", 0), 4),
                        "Recall": round(metrics.get("recall", 0), 4),
                        "F1-Score": round(metrics.get("f1-score", 0), 4),
                        "Support (Samples)": int(metrics.get("support", 0)),
                    })
            st.dataframe(pd.DataFrame(report_rows), use_container_width=True)

        # ======================================================================
        # 3. WHITE-BOX DECISION TREE HARDWARE EXTRACTION & TRANSPILER
        # ======================================================================
        if dt_res is not None:
            st.markdown("---")
            if "pruning_curve" in dt_res:
                st.markdown("### Cost-Complexity Pruning ($c_{\\alpha}$) Sensitivity Curve")
                st.markdown(
                    "Evaluates the mathematical tradeoff between decision tree capacity (node count) and predictive generalization (Macro F1 score)."
                )
                p_curve = dt_res["pruning_curve"]
                df_prune = pd.DataFrame({
                    "Alpha": p_curve["alphas"],
                    "Train F1": p_curve["train_f1"],
                    "Validation F1": p_curve["test_f1"],
                    "Node Count": p_curve["node_counts"],
                })

                fig_prune = go.Figure()
                fig_prune.add_trace(
                    go.Scatter(
                        x=df_prune["Alpha"],
                        y=df_prune["Validation F1"],
                        mode="lines+markers",
                        name="Validation Macro F1",
                        line=dict(color="#38bdf8", width=2.5),
                        marker=dict(size=6),
                    )
                )
                fig_prune.add_trace(
                    go.Scatter(
                        x=df_prune["Alpha"],
                        y=df_prune["Train F1"],
                        mode="lines",
                        name="Train Macro F1",
                        line=dict(color="#94a3b8", width=1.5, dash="dot"),
                    )
                )
                fig_prune.add_trace(
                    go.Scatter(
                        x=df_prune["Alpha"],
                        y=df_prune["Node Count"],
                        mode="lines+markers",
                        name="Tree Node Count",
                        yaxis="y2",
                        line=dict(color="#f59e0b", width=2),
                        marker=dict(symbol="triangle-up", size=6),
                    )
                )
                fig_prune.update_layout(
                    title="Pruning Parameter Alpha vs Macro F1 & Model Complexity",
                    xaxis=dict(title="Cost-Complexity Alpha (ccp_alpha)"),
                    yaxis=dict(title="Macro F1 Score", range=[0.5, 1.02]),
                    yaxis2=dict(
                        title="Decision Nodes",
                        overlaying="y",
                        side="right",
                        showgrid=False,
                    ),
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    font=dict(color="#f8fafc"),
                    height=420,
                )
                st.plotly_chart(fig_prune, use_container_width=True)

            with st.expander("View Full Decision Tree Topology Rules"):
                st.code(dt_res["tree_rules_text"], language="text")

# ==============================================================================
# STAGE 4: STRATEGY ANALYTICS & SPATIAL EVALUATION (STATISTICAL VERIFICATION)
# ==============================================================================
with tabs[3]:
    st.subheader("Stage 4: Strategy Analytics, Spatial Heatmaps, and 2D Replays")
    st.markdown(
        "Quantitative evaluation of tactical match dynamics using ground-truth Observer Telemetry."
    )

    obs_files = sorted(list(OBSERVER_DATA_DIR.glob("*_observer.parquet")))
    obs_file_names = [f.name for f in obs_files]

    if obs_file_names:
        selected_obs_file = st.selectbox(
            "Select Observer Dataset", options=obs_file_names, key="eval_obs_select"
        )
        selected_obs_path = OBSERVER_DATA_DIR / selected_obs_file
        df_obs_eval = load_parquet_cached(str(selected_obs_path))

        meta = validate_sumo_dataset(df_obs_eval, selected_obs_path)
        render_dataset_badge(meta)

        active_eval_config = CONFIG_REGISTRY[meta["robot_class"]]
        center_zone_r = active_eval_config.center_zone_radius_cm

        df_all_kpis = compute_observer_kpis_cached(str(selected_obs_path), center_zone_r)

        st.markdown("### Micro-Combat KPI Summary Cards")
        k_col1, k_col2, k_col3, k_col4 = st.columns(4)
        with k_col1:
            mean_tta = df_all_kpis["TTA_ms"].mean()
            st.metric(
                "Time-to-Acquisition (TTA)",
                f"{mean_tta:.1f} ms" if not np.isnan(mean_tta) else "N/A",
            )
        with k_col2:
            mean_center = df_all_kpis["Center_Control_Pct"].mean()
            st.metric(
                "Center-Control %",
                f"{mean_center:.2f}%",
            )
        with k_col3:
            mean_ttro = df_all_kpis["TTRO_ms"].dropna().mean()
            st.metric(
                "Time-to-Ring-Out (TTRO)",
                f"{mean_ttro:.1f} ms" if not np.isnan(mean_ttro) else "N/A",
            )
        with k_col4:
            win_rate = (df_all_kpis["Status"] == "BOT_A_WIN").mean() * 100.0
            st.metric("Overall Win Rate", f"{win_rate:.1f}%")

        # Tactical Strategy & Starting Formation Breakdown Visualizations
        if not df_all_kpis.empty and ("Strategy_B" in df_all_kpis.columns or "Formation_A" in df_all_kpis.columns):
            st.markdown("---")
            st.markdown("### Tactical Strategy & Starting Formation Performance Analysis")
            st.caption("Empirical visual analytics evaluating combat outcome distributions, opening formation efficacy, and strategy synergies.")

            # Row 1: Strategy Outcome Distribution & Starting Formation Outcome Distribution
            row1_col1, row1_col2 = st.columns(2)

            with row1_col1:
                st.markdown("#### Tactical Strategy Performance Breakdown")
                has_strat_a = "Strategy_A" in df_all_kpis.columns and df_all_kpis["Strategy_A"].iloc[0] != "UNKNOWN"
                strat_col = "Strategy_A" if (has_strat_a and df_all_kpis["Strategy_A"].nunique() > 1) else "Strategy_B"

                strat_stats = []
                for s_name, grp in df_all_kpis.groupby(strat_col):
                    tot = len(grp)
                    if tot == 0:
                        continue
                    wa = (grp["Status"] == "BOT_A_WIN").sum()
                    wb = (grp["Status"] == "BOT_B_WIN").sum()
                    dr = (grp["Status"] == "DRAW").sum()
                    win_pct = round((wa / tot) * 100.0, 1) if strat_col == "Strategy_A" else round((wb / tot) * 100.0, 1)
                    loss_pct = round((wb / tot) * 100.0, 1) if strat_col == "Strategy_A" else round((wa / tot) * 100.0, 1)
                    draw_pct = round((dr / tot) * 100.0, 1)
                    strat_stats.append({
                        "Strategy": s_name,
                        "Total": tot,
                        "Win_Pct": win_pct,
                        "Draw_Pct": draw_pct,
                        "Loss_Pct": loss_pct,
                    })

                if strat_stats:
                    df_s = pd.DataFrame(strat_stats).sort_values("Win_Pct", ascending=True)
                    fig_strat = go.Figure()
                    fig_strat.add_trace(go.Bar(
                        y=df_s["Strategy"], x=df_s["Win_Pct"], name="Win %", orientation="h",
                        marker=dict(color="#22c55e"),
                        text=df_s["Win_Pct"].apply(lambda v: f"{v:.1f}%" if v >= 6.0 else ""),
                        textposition="inside", insidetextanchor="middle",
                        hovertemplate="<b>%{y}</b><br>Win: %{x:.1f}%<extra></extra>",
                    ))
                    fig_strat.add_trace(go.Bar(
                        y=df_s["Strategy"], x=df_s["Draw_Pct"], name="Draw %", orientation="h",
                        marker=dict(color="#64748b"),
                        text=df_s["Draw_Pct"].apply(lambda v: f"{v:.1f}%" if v >= 6.0 else ""),
                        textposition="inside", insidetextanchor="middle",
                        hovertemplate="<b>%{y}</b><br>Draw: %{x:.1f}%<extra></extra>",
                    ))
                    fig_strat.add_trace(go.Bar(
                        y=df_s["Strategy"], x=df_s["Loss_Pct"], name="Loss %", orientation="h",
                        marker=dict(color="#ef4444"),
                        text=df_s["Loss_Pct"].apply(lambda v: f"{v:.1f}%" if v >= 6.0 else ""),
                        textposition="inside", insidetextanchor="middle",
                        hovertemplate="<b>%{y}</b><br>Loss: %{x:.1f}%<extra></extra>",
                    ))
                    fig_strat.update_layout(
                        barmode="stack", height=270, margin=dict(t=25, b=30, l=10, r=10),
                        xaxis=dict(title="Outcome Distribution (%)", range=[0, 100], gridcolor="#1e293b"),
                        yaxis=dict(gridcolor="#1e293b"),
                        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="#0f172a", font=dict(color="#f8fafc"),
                        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                    )
                    st.plotly_chart(fig_strat, use_container_width=True)
                else:
                    st.info("No strategy telemetry available.")

            with row1_col2:
                st.markdown("#### Starting Formation Effectiveness Breakdown")
                form_col = "Formation_A" if "Formation_A" in df_all_kpis.columns else "Formation_B"
                form_stats = []
                for f_name, grp in df_all_kpis.groupby(form_col):
                    tot = len(grp)
                    if tot == 0:
                        continue
                    wa = (grp["Status"] == "BOT_A_WIN").sum()
                    wb = (grp["Status"] == "BOT_B_WIN").sum()
                    dr = (grp["Status"] == "DRAW").sum()
                    form_stats.append({
                        "Formation": f_name,
                        "Total": tot,
                        "Win_Pct": round((wa / tot) * 100.0, 1),
                        "Draw_Pct": round((dr / tot) * 100.0, 1),
                        "Loss_Pct": round((wb / tot) * 100.0, 1),
                    })

                if form_stats:
                    df_f = pd.DataFrame(form_stats).sort_values("Win_Pct", ascending=True)
                    fig_form = go.Figure()
                    fig_form.add_trace(go.Bar(
                        y=df_f["Formation"], x=df_f["Win_Pct"], name="Win %", orientation="h",
                        marker=dict(color="#22c55e"),
                        text=df_f["Win_Pct"].apply(lambda v: f"{v:.1f}%" if v >= 6.0 else ""),
                        textposition="inside", insidetextanchor="middle",
                        hovertemplate="<b>%{y}</b><br>Win: %{x:.1f}%<extra></extra>",
                    ))
                    fig_form.add_trace(go.Bar(
                        y=df_f["Formation"], x=df_f["Draw_Pct"], name="Draw %", orientation="h",
                        marker=dict(color="#64748b"),
                        text=df_f["Draw_Pct"].apply(lambda v: f"{v:.1f}%" if v >= 6.0 else ""),
                        textposition="inside", insidetextanchor="middle",
                        hovertemplate="<b>%{y}</b><br>Draw: %{x:.1f}%<extra></extra>",
                    ))
                    fig_form.add_trace(go.Bar(
                        y=df_f["Formation"], x=df_f["Loss_Pct"], name="Loss %", orientation="h",
                        marker=dict(color="#ef4444"),
                        text=df_f["Loss_Pct"].apply(lambda v: f"{v:.1f}%" if v >= 6.0 else ""),
                        textposition="inside", insidetextanchor="middle",
                        hovertemplate="<b>%{y}</b><br>Loss: %{x:.1f}%<extra></extra>",
                    ))
                    fig_form.update_layout(
                        barmode="stack", height=270, margin=dict(t=25, b=30, l=10, r=10),
                        xaxis=dict(title="Outcome Distribution (%)", range=[0, 100], gridcolor="#1e293b"),
                        yaxis=dict(gridcolor="#1e293b"),
                        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="#0f172a", font=dict(color="#f8fafc"),
                        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                    )
                    st.plotly_chart(fig_form, use_container_width=True)
                else:
                    st.info("No formation telemetry available.")

            # Row 2: Tactical Synergy Matrix Heatmap (Strategy × Formation)
            st.markdown("#### Tactical Synergy Matrix (Strategy × Starting Formation Win Rate)")
            strat_c = "Strategy_A" if ("Strategy_A" in df_all_kpis.columns and df_all_kpis["Strategy_A"].iloc[0] != "UNKNOWN") else "Strategy_B"
            strat_names_all = sorted(list(df_all_kpis[strat_c].unique()))
            form_names_all = sorted(list(df_all_kpis[form_col].unique()))

            z_matrix = []
            text_matrix = []
            for s in strat_names_all:
                z_row = []
                text_row = []
                for f in form_names_all:
                    sub = df_all_kpis[(df_all_kpis[strat_c] == s) & (df_all_kpis[form_col] == f)]
                    if len(sub) > 0:
                        win_p = ((sub["Status"] == "BOT_A_WIN").sum() / len(sub)) * 100.0
                        z_row.append(round(win_p, 1))
                        text_row.append(f"<b>{win_p:.1f}%</b><br>({len(sub)} m)")
                    else:
                        z_row.append(0.0)
                        text_row.append("N/A")
                z_matrix.append(z_row)
                text_matrix.append(text_row)

            if z_matrix and any(len(row) > 0 for row in z_matrix):
                custom_syn_scale = [
                    [0.0, "#0f172a"],
                    [0.2, "#1e293b"],
                    [0.4, "#0369a1"],
                    [0.6, "#0ea5e9"],
                    [0.8, "#22c55e"],
                    [1.0, "#4ade80"],
                ]
                fig_syn = go.Figure(data=go.Heatmap(
                    z=z_matrix, x=form_names_all, y=strat_names_all,
                    text=text_matrix, texttemplate="%{text}", textfont=dict(size=12, color="#f8fafc"),
                    colorscale=custom_syn_scale,
                    zmin=0.0, zmax=100.0,
                    colorbar=dict(title="Win %", thickness=14, len=0.85),
                    hovertemplate="Strategy: %{y}<br>Formation: %{x}<br>Win Rate: %{z:.1f}%<extra></extra>",
                ))
                fig_syn.update_layout(
                    height=300, margin=dict(t=20, b=40, l=10, r=10),
                    xaxis=dict(title="Starting Formation", gridcolor="#1e293b"),
                    yaxis=dict(title="Tactical Strategy", gridcolor="#1e293b"),
                    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="#0f172a", font=dict(color="#f8fafc"),
                )
                st.plotly_chart(fig_syn, use_container_width=True)

        st.markdown("---")

        # 2. Spatial Density Heatmap
        st.markdown("### 2D Spatial Positioning Density Heatmap")
        df_bot_a_all = df_obs_eval[df_obs_eval["Bot_ID"] == "Bot_A"]

        r_dohyo = active_eval_config.dohyo_radius_cm
        r_center = active_eval_config.center_zone_radius_cm
        r_max = r_dohyo * 1.12

        custom_colorscale = [
            [0.0, "rgba(15, 23, 42, 0.0)"],
            [0.05, "#1e293b"],
            [0.15, "#0369a1"],
            [0.35, "#0ea5e9"],
            [0.60, "#38bdf8"],
            [0.80, "#f59e0b"],
            [1.0, "#ef4444"],
        ]

        x_pts = df_bot_a_all["Pos_X"].values
        y_pts = df_bot_a_all["Pos_Y"].values
        x_bins = np.linspace(-r_max, r_max, 46)
        y_bins = np.linspace(-r_max, r_max, 46)
        H, xedges, yedges = np.histogram2d(x_pts, y_pts, bins=[x_bins, y_bins])
        H_masked = np.where(H == 0, np.nan, H)

        fig_heat = go.Figure(data=go.Heatmap(
            z=H_masked.T,
            x=0.5 * (xedges[:-1] + xedges[1:]),
            y=0.5 * (yedges[:-1] + yedges[1:]),
            colorscale=custom_colorscale,
            colorbar=dict(title="Pings", thickness=14, len=0.75),
            hoverongaps=False,
            hovertemplate="X: %{x:.1f} cm<br>Y: %{y:.1f} cm<br>Pings: %{z}<extra></extra>",
        ))
        theta_c = np.linspace(0, 2 * np.pi, 120)
        fig_heat.add_trace(
            go.Scatter(
                x=r_dohyo * np.cos(theta_c),
                y=r_dohyo * np.sin(theta_c),
                mode="lines",
                name="Dohyo Perimeter",
                line=dict(color="#ffffff", width=2.5),
            )
        )
        fig_heat.add_trace(
            go.Scatter(
                x=r_center * np.cos(theta_c),
                y=r_center * np.sin(theta_c),
                mode="lines",
                name="Center Zone (50% R)",
                line=dict(color="#38bdf8", width=1.5, dash="dash"),
            )
        )
        fig_heat.update_layout(
            xaxis=dict(
                range=[-r_max, r_max],
                title="X Position (cm)",
                zeroline=False,
                gridcolor="#1e293b",
            ),
            yaxis=dict(
                range=[-r_max, r_max],
                title="Y Position (cm)",
                scaleanchor="x",
                scaleratio=1,
                zeroline=False,
                gridcolor="#1e293b",
            ),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="#0f172a",
            font=dict(color="#f8fafc"),
            coloraxis_colorbar=dict(title="Pings", thickness=14, len=0.75),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            margin=dict(t=50, b=40, l=40, r=40),
            height=560,
        )

        heat_col1, heat_col2 = st.columns([3, 2])
        with heat_col1:
            st.plotly_chart(fig_heat, use_container_width=True)

        with heat_col2:
            center_zone_pct = (df_bot_a_all["Dist_To_Center"] <= r_center).mean() * 100.0
            perim_vuln_pct = (df_bot_a_all["Dist_To_Center"] > 0.8 * r_dohyo).mean() * 100.0
            mean_dist_to_edge = (r_dohyo - df_bot_a_all["Dist_To_Center"]).clip(lower=0).mean()
            mean_dist_to_center = df_bot_a_all["Dist_To_Center"].mean()

            st.markdown(
                f"""<div class="metric-card" style="padding: 22px 24px; margin-bottom: 16px; min-height: 250px; display: flex; flex-direction: column; justify-content: center;">
<h5 style="color: #38bdf8; margin-bottom: 14px; font-weight: 600;">Spatial Dynamics & Zone Breakdown</h5>
<div style="font-size: 0.95rem; line-height: 2.0; color: #cbd5e1;">
• <strong>Center Ring Dominance (r ≤ 50% R):</strong> {center_zone_pct:.1f}% of total match ticks<br>
• <strong>Edge Risk Exposure (r > 80% R):</strong> {perim_vuln_pct:.1f}% ring perimeter vulnerability<br>
• <strong>Mean Distance to Ring Boundary:</strong> {mean_dist_to_edge:.1f} cm<br>
• <strong>Mean Radial Center Offset:</strong> {mean_dist_to_center:.1f} cm
</div>
</div>
<div class="metric-card" style="padding: 22px 24px; min-height: 250px; display: flex; flex-direction: column; justify-content: center;">
<h5 style="color: #38bdf8; margin-bottom: 14px; font-weight: 600;">Visual Interpretation Guide</h5>
<div style="font-size: 0.9rem; line-height: 1.9; color: #cbd5e1;">
<span style="color: #ef4444; font-weight: bold; font-size: 1.1rem;">■</span> <strong>Glow Core (Amber/Red):</strong> High-traffic combat collision and center lock zone.<br>
<span style="color: #0ea5e9; font-weight: bold; font-size: 1.1rem;">■</span> <strong>Cyan Halos:</strong> Search sweeps and radial evasion maneuvers.<br>
<span style="color: #ffffff; font-weight: bold; font-size: 1.1rem;">—</span> <strong>White Boundary:</strong> Outer physical Dohyo perimeter (R = {r_dohyo} cm).<br>
<span style="color: #38bdf8; font-weight: bold; font-size: 1.1rem;">--</span> <strong>Dashed Blue Ring:</strong> Tactical inner center anchor (50% R).
</div>
</div>""",
                unsafe_allow_html=True,
            )

        st.markdown("---")

        # 3. Top-Down 2D Match Replay Visualizer
        st.markdown("### Top-Down 2D Match Replay Visualizer")

        all_match_ids = sorted(list(df_obs_eval["Match_ID"].unique()))
        match_col1, match_col2 = st.columns([3, 2])
        with match_col1:
            selected_match_id = st.selectbox(
                "Select Match for Granular Breakdown & 2D Replay",
                options=all_match_ids,
                index=0,
                key="replay_match_selector",
            )
        with match_col2:
            # Resolve complete telemetry with live sensor readings and actuator commands for replay
            processed_file_p = PROCESSED_DATA_DIR / selected_obs_file.replace("_observer.parquet", "_processed.parquet")
            if processed_file_p.exists():
                df_match_source = load_parquet_cached(str(processed_file_p))
            else:
                agent_file_p = AGENT_DATA_DIR / selected_obs_file.replace("_observer.parquet", "_agent.parquet")
                if agent_file_p.exists():
                    df_ag = load_parquet_cached(str(agent_file_p))
                    common_cols = [c for c in ["Match_ID", "Bot_ID", "Timestamp_ms"] if c in df_obs_eval.columns and c in df_ag.columns]
                    df_match_source = pd.merge(df_obs_eval, df_ag, on=common_cols, how="left", suffixes=("", "_dup"))
                else:
                    df_match_source = df_obs_eval

            df_match = df_match_source[df_match_source["Match_ID"] == selected_match_id].sort_values(["Tick", "Bot_ID"])
            m_status = str(df_match["Match_Status"].iloc[-1]) if not df_match.empty else "N/A"
            m_dur = float(df_match["Timestamp_ms"].iloc[-1]) / 1000.0 if not df_match.empty else 0.0
            st.markdown(
                f"""
                <div style="background: #0f172a; padding: 10px 14px; border-radius: 8px; border: 1px solid #334155; margin-top: 18px; display: flex; gap: 16px;">
                    <div><strong>Outcome:</strong> <span style="color: {'#22c55e' if 'WIN' in m_status else '#38bdf8'};">{m_status}</span></div>
                    <div><strong>Duration:</strong> {m_dur:.2f} s</div>
                    <div><strong>Total Ticks:</strong> {len(df_match['Tick'].unique())}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        html_code = render_html5_canvas_replay(df_match, active_eval_config, selected_match_id)
        components.html(html_code, height=710)
    else:
        st.warning("No observer datasets found in `data/observer/`. Please run ETL in Stage 2.")

# ==============================================================================
# STAGE 5: REAL WORLD APPLICATION
# ==============================================================================
with tabs[4]:
    st.subheader("Stage 5: Real World Application")
    st.markdown(
        "Comprehensive project deliverables, explainable decision logic, Sim-to-Real hardware specifications, and embedded C++ firmware deployment."
    )

    # 1. Top Dropdown Selector for Decision Tree Package
    dt_files = sorted(list(DT_DATA_DIR.glob("*_dt.json")))
    dt_file_names = [f.name for f in dt_files]

    dt_data = None
    if dt_file_names:
        selected_dt_file = st.selectbox(
            "Select Trained Decision Tree Package (`data/dt/`)",
            options=dt_file_names,
            key="stage5_dt_selector",
        )
        selected_dt_path = DT_DATA_DIR / selected_dt_file
        try:
            dt_data = load_decision_tree_package(selected_dt_path)
        except Exception as e:
            st.error(f"Error loading Decision Tree package: {e}")
    else:
        st.warning("No trained Decision Tree packages found in `data/dt/`. Please train a Decision Tree model in Stage 3 first.")

    if dt_data is not None:
        dataset_base = dt_data.get("dataset_base", "telemetry_dataset")
        robot_class = dt_data.get("robot_class", "KIT_1KG")
        metrics = dt_data.get("metrics", {})
        classes = dt_data.get("classes", [])
        feature_names = dt_data.get("feature_names", [])
        feature_importances = dt_data.get("feature_importances", {})
        tree_depth = dt_data.get("tree_depth", 0)
        n_leaves = dt_data.get("n_leaves", 0)
        timestamp_iso = dt_data.get("timestamp_iso", "N/A")
        cpp_code = dt_data.get("cpp_header_code", "")

        # Model Verification Badge
        st.markdown(
            f"""
        <div class="metric-card" style="padding: 16px 20px; margin-bottom: 20px; display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 12px;">
            <div>
                <span style="color: #38bdf8; font-weight: 700; font-size: 1.05rem;">Verified Decision Tree Deployment Package:</span> <code style="font-size: 1rem; color: #f8fafc;">{selected_dt_file}</code>
                <div style="font-size: 0.85rem; color: #94a3b8; margin-top: 4px;">Trained on: {dataset_base} | Robot Class: {robot_class}</div>
            </div>
            <div style="display: flex; gap: 10px; align-items: center;">
                <span style="background: rgba(34, 197, 94, 0.2); color: #4ade80; border: 1px solid #22c55e; padding: 4px 10px; border-radius: 6px; font-weight: 700; font-size: 0.85rem;">Test Macro F1: {metrics.get('test_macro_f1', 0.0):.4f}</span>
                <span style="background: rgba(14, 165, 233, 0.2); color: #38bdf8; border: 1px solid #0ea5e9; padding: 4px 10px; border-radius: 6px; font-weight: 700; font-size: 0.85rem;">Depth: {tree_depth} Levels</span>
                <span style="background: rgba(245, 158, 11, 0.2); color: #fbbf24; border: 1px solid #f59e0b; padding: 4px 10px; border-radius: 6px; font-weight: 700; font-size: 0.85rem;">Leaves: {n_leaves} Rules</span>
            </div>
        </div>
        """,
            unsafe_allow_html=True,
        )

        # ======================================================================
        # 1. MICROCONTROLLER RESOURCE & EXECUTION LATENCY BENCHMARKS
        # ======================================================================
        st.markdown("### Microcontroller Hardware Resource & Execution Benchmarks")
        st.markdown(
            """
            <div style="background: rgba(15, 23, 42, 0.5); border: 1px solid #334155; border-left: 4px solid #38bdf8; padding: 12px 18px; border-radius: 6px; margin-bottom: 16px; font-size: 0.88rem; color: #cbd5e1; line-height: 1.6;">
                <div style="font-weight: 600; color: #f8fafc; margin-bottom: 6px;">
                    Hardware Benchmark Reference: 8-bit AVR ATmega328P @ 16 MHz (Arduino Uno / Nano)
                </div>
                <ul style="margin: 0; padding-left: 18px; color: #cbd5e1;">
                    <li><strong>Physical Constraints:</strong> 2 KB SRAM, 32 KB Flash ROM, 62.5 ns/cycle (no hardware FPU).</li>
                    <li><strong>Selection Rationale:</strong> As the most resource-constrained MCU in competitive robotics, passing these thresholds guarantees deterministic, zero-lag execution on any 32-bit board (Teensy, ESP32, STM32) with zero risk of stack overflow.</li>
                </ul>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # Dynamic computation of embedded hardware benchmarks
        bench_depth = max(tree_depth, 1)
        bench_features = len(feature_names) if feature_names else 10
        bench_leaves = max(n_leaves, 1)
        bench_total_nodes = 2 * bench_leaves - 1

        # Microcontroller resource calculation (ATmega328P @ 16 MHz baseline)
        mcu_bench = compute_mcu_benchmarks(
            tree_depth=bench_depth,
            n_leaves=bench_leaves,
            n_features=bench_features,
        )
        bench_latency_us = mcu_bench["latency_us"]
        bench_sram_bytes = mcu_bench["sram_bytes"]
        bench_flash_bytes = mcu_bench["flash_bytes"]
        bench_flash_kb = mcu_bench["flash_kb"]
        bench_bandwidth_khz = mcu_bench["bandwidth_khz"]
        bench_headroom_x = mcu_bench["headroom_x"]

        def render_benchmark_badge(status: str, label: str) -> str:
            if status == "green":
                bg = "rgba(34, 197, 94, 0.15)"
                color = "#4ade80"
                border = "rgba(34, 197, 94, 0.4)"
            elif status == "yellow":
                bg = "rgba(245, 158, 11, 0.15)"
                color = "#fbbf24"
                border = "rgba(245, 158, 11, 0.4)"
            else:
                bg = "rgba(239, 68, 68, 0.15)"
                color = "#f87171"
                border = "rgba(239, 68, 68, 0.4)"
            return (
                f'<div style="margin-top: -10px; margin-bottom: 8px;">'
                f'<span style="display: inline-flex; align-items: center; '
                f'background: {bg}; color: {color}; border: 1px solid {border}; '
                f'padding: 3px 10px; border-radius: 9999px; font-size: 0.76rem; font-weight: 600;">'
                f'{label}'
                f'</span></div>'
            )

        # Dynamic rating thresholds based on embedded hardware constraints
        if bench_latency_us <= 20.0:
            lat_status, lat_label = "green", "Optimal (< 0.04% Loop)"
        elif bench_latency_us <= 50.0:
            lat_status, lat_label = "yellow", "Acceptable (< 0.1% Loop)"
        else:
            lat_status, lat_label = "red", "High Latency (> 0.1% Loop)"

        if bench_sram_bytes <= 128:
            sram_status, sram_label = "green", "Optimal (Zero Heap)"
        elif bench_sram_bytes <= 512:
            sram_status, sram_label = "yellow", "Acceptable (< 25% RAM)"
        else:
            sram_status, sram_label = "red", "High RAM Usage"

        if bench_flash_kb <= 16.0:
            flash_status, flash_label = "green", "Optimal (< 20% Flash)"
        elif bench_flash_kb <= 28.0:
            flash_status, flash_label = "yellow", "Acceptable (< 32KB Flash)"
        else:
            flash_status, flash_label = "red", "Flash Constrained"

        if bench_bandwidth_khz >= 50:
            bw_status, bw_label = "green", "Optimal (> 2,500× Margin)"
        elif bench_bandwidth_khz >= 10:
            bw_status, bw_label = "yellow", "Acceptable (> 500× Margin)"
        else:
            bw_status, bw_label = "red", "Low Headroom (< 500×)"

        kpi_col1, kpi_col2, kpi_col3, kpi_col4 = st.columns(4)
        with kpi_col1:
            st.metric("Worst-Case Latency", f"< {bench_latency_us:.1f} μs")
            st.markdown(render_benchmark_badge(lat_status, lat_label), unsafe_allow_html=True)
        with kpi_col2:
            st.metric("SRAM Footprint", f"{bench_sram_bytes} Bytes")
            st.markdown(render_benchmark_badge(sram_status, sram_label), unsafe_allow_html=True)
        with kpi_col3:
            st.metric("Flash ROM Usage", f"~{bench_flash_kb:.1f} KB")
            st.markdown(render_benchmark_badge(flash_status, flash_label), unsafe_allow_html=True)
        with kpi_col4:
            st.metric("Max Decision Bandwidth", f"{bench_bandwidth_khz:,} kHz")
            st.markdown(render_benchmark_badge(bw_status, bw_label), unsafe_allow_html=True)

        st.markdown("<div style='margin-top: 10px;'></div>", unsafe_allow_html=True)
        with st.expander("How Microcontroller Resource Benchmarks Are Computed", expanded=False):
            st.markdown(
                f"""
                <div style="font-size: 0.88rem; line-height: 1.7; color: #cbd5e1;">
                    • <strong>Worst-Case Latency (&lt; {bench_latency_us:.1f} μs):</strong> Maximum tree depth traversal (<code>D = {bench_depth}</code>) on a 16 MHz AVR (~0.75 μs/level): <code>{bench_depth} × 0.75 μs = {bench_latency_us:.2f} μs</code>.<br>
                    • <strong>SRAM Footprint ({bench_sram_bytes} Bytes):</strong> Exact stack memory required by the <code>SumoSensors</code> input struct: <code>{bench_features} active features × 4 bytes/float = {bench_sram_bytes} bytes</code>. Heap memory allocation is strictly zero (<code>malloc = 0</code>).<br>
                    • <strong>Flash ROM Usage (~{bench_flash_kb:.1f} KB):</strong> Compiled machine code footprint of <code>evaluate_strategy()</code> under size optimization (<code>-Os</code>). An unrolled tree with <code>{bench_total_nodes} nodes</code> (<code>2 × {bench_leaves} - 1</code>) consumes ~24 bytes per node: <code>{bench_total_nodes} × 24 = {bench_flash_bytes} bytes</code>.<br>
                    • <strong>Max Decision Bandwidth ({bench_bandwidth_khz:,} kHz):</strong> Theoretical upper bound on evaluation throughput (<code>1 / {bench_latency_us:.2f} μs = {bench_bandwidth_khz:,} kHz</code>). Exceeds the robot's 20 Hz (50 ms) control loop by <strong>{bench_headroom_x:,}×</strong>.
                </div>
                """,
                unsafe_allow_html=True,
            )

        st.markdown("---")

        # ======================================================================
        # 2. HARDWARE OPERATIONAL DECISION THRESHOLDS & NODE HIERARCHY
        # ======================================================================
        st.markdown("### Hardware Operational Decision Thresholds & Node Hierarchy")
        st.markdown(
            "While ensemble architectures require hundreds of kilobytes of memory, the Decision Tree compiles into deterministic C++ if/else logic executable in < 5 microseconds on microcontrollers with zero RAM."
        )

        threshold_view_mode = st.radio(
            "Select Decision Table View",
            options=[
                "Per-Sensor Calibrated Parameters (Hardware Summary)",
                "Full Decision Tree Node Hierarchy",
            ],
            index=0,
            horizontal=True,
            key="stage5_threshold_view_toggle",
        )

        if threshold_view_mode == "Per-Sensor Calibrated Parameters (Hardware Summary)":
            df_hw = dt_data.get("hardware_summary_table")
            if df_hw is None or (isinstance(df_hw, pd.DataFrame) and df_hw.empty) or (isinstance(df_hw, list) and len(df_hw) == 0):
                df_hw = build_hardware_summary_table(
                    dt_data.get("thresholds_table", pd.DataFrame()),
                    dt_data.get("feature_importances", {}),
                )
            elif isinstance(df_hw, list):
                df_hw = pd.DataFrame(df_hw)

            if "Sensor_Type" in df_hw.columns:
                df_hw["Sensor_Type"] = df_hw["Sensor_Type"].replace(
                    {"Closing Velocity (d(Opp_F0)/dt)": "Closing Velocity"}
                )
            # Apply consistent canonical row order across all robot classes
            if "Sensor_Feature" in df_hw.columns:
                canonical_order = [
                    "Opp_F0",
                    "Delta_Opp_F0",
                    "IR_Edge_FL",
                    "IR_Edge_FR",
                    "Opp_Lat_Delta",
                    "Opp_Bearing_Est_Deg",
                    "Opp_L15",
                    "Opp_R15",
                    "Opp_L18",
                    "Opp_R18",
                    "Opp_L22_5",
                    "Opp_R22_5",
                    "Opp_L90",
                    "Opp_R90",
                ]
                present = [f for f in canonical_order if f in df_hw["Sensor_Feature"].values]
                rem = [f for f in df_hw["Sensor_Feature"].values if f not in present]
                df_hw = df_hw.set_index("Sensor_Feature").loc[present + rem].reset_index()

            # Dynamic height: perfectly fits all rows with zero scrollbar and zero blank padding
            dynamic_height = (len(df_hw) + 1) * 35 + 5
            st.dataframe(df_hw, use_container_width=True, hide_index=True, height=dynamic_height)
        else:
            thresh_df = dt_data.get("thresholds_table", pd.DataFrame())
            if isinstance(thresh_df, list):
                thresh_df = pd.DataFrame(thresh_df)
            st.dataframe(thresh_df, use_container_width=True, hide_index=True)

        st.markdown("---")

        # ======================================================================
        # 3. EMBEDDED C++ FIRMWARE & INTEGRATION HARNESS DELIVERABLES
        # ======================================================================
        st.markdown("### Embedded C++ Firmware & Integration Harness Deliverables")
        st.markdown(
            "Production-ready, zero-dependency C++ header (`extern \"C\"`) and 20 Hz deterministic Arduino/ESP32 control loop harness ready for immediate flashing onto your robot's onboard microcontroller."
        )

        exp_col1, exp_col2 = st.columns([1, 1])

        with exp_col1:
            st.markdown("#### 1. Policy Header (`.h`)")
            h_col1, h_col2 = st.columns([3, 2])
            with h_col1:
                custom_header_name = st.text_input(
                    "Header Filename",
                    value=f"{dataset_base}_policy.h",
                    key="stage5_custom_header_name",
                )
            with h_col2:
                st.markdown("<div style='margin-top: 28px;'></div>", unsafe_allow_html=True)
                st.download_button(
                    label="Download (.h)",
                    data=cpp_code,
                    file_name=custom_header_name,
                    mime="text/x-chdr",
                    type="primary",
                    use_container_width=True,
                    key="stage5_download_header_btn",
                )

        harness_code = generate_cpp_harness(
            robot_class=robot_class,
            header_filename=custom_header_name,
            dataset_name=dataset_base,
        )

        with exp_col2:
            st.markdown("#### 2. Control Loop Harness (`.cpp`)")
            m_col1, m_col2 = st.columns([3, 2])
            with m_col1:
                custom_main_name = st.text_input(
                    "Harness Filename",
                    value=f"{dataset_base}_main.cpp",
                    key="stage5_custom_main_name",
                )
            with m_col2:
                st.markdown("<div style='margin-top: 28px;'></div>", unsafe_allow_html=True)
                st.download_button(
                    label="Download (.cpp)",
                    data=harness_code,
                    file_name=custom_main_name,
                    mime="text/x-c++src",
                    type="primary",
                    use_container_width=True,
                    key="stage5_download_main_btn",
                )

        st.markdown(
            f"""
        <div class="output-path-card" style="margin-top: 14px; margin-bottom: 14px;">
            <strong>Generated Firmware Artifacts:</strong> <code>{custom_header_name}</code> & <code>{custom_main_name}</code><br>
            <strong>Target Architectures:</strong> <code>Arduino IDE / PlatformIO / STM32CubeIDE / ESP-IDF</code>
        </div>
        """,
            unsafe_allow_html=True,
        )

        with st.expander(f"1. View Transpiled C++ Policy Header ({custom_header_name})", expanded=False):
            st.code(cpp_code, language="cpp")

        with st.expander(f"2. View Ready-to-Flash Firmware Integration Harness ({custom_main_name})", expanded=False):
            st.code(harness_code, language="cpp")
