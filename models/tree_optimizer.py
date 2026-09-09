"""Machine Learning Strategy Modeling, Multi-Model Benchmark & C++ Firmware Transpiler.

Trains and tunes 4 machine learning architectures:
1. Decision Tree (Primary White-Box Firmware Candidate)
2. Logistic Regression (Linear Baseline)
3. Random Forest (Ensemble Bagging)
4. Gradient Boosting (Ensemble Boosting)

Extracts interpretable hardware decision thresholds and transpiles the pruned
decision tree into an Arduino/ESP32-compliant C++ header (strategy_config.h).
"""

from __future__ import annotations

import json
import pickle
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from scipy.stats import ttest_rel
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier, export_text

from config.config import (
    DT_DATA_DIR,
    EDGE_LINE_THRESHOLD,
    PWM_ATTACK_FULL,
    PWM_RECOVERY_LEFT,
    PWM_RECOVERY_RIGHT,
    PWM_SEARCH_FAST,
    SIMULATION_DT_S,
)

__all__ = [
    "engineer_features",
    "evaluate_sensor_action",
    "compute_mcu_benchmarks",
    "train_and_benchmark_models",
    "train_and_optimize_tree",
    "build_hardware_summary_table",
    "generate_cpp_header",
    "generate_cpp_harness",
    "save_decision_tree_package",
    "load_decision_tree_package",
]


# ==============================================================================
# 0. SENSOR FEATURE ENGINEERING & MCU HARDWARE BENCHMARKS
# ==============================================================================
def engineer_features(agent_df: pd.DataFrame, dt: float = SIMULATION_DT_S) -> Tuple[pd.DataFrame, List[str]]:
    """Engineer dynamic temporal velocity, lateral differential, and harmonic target bearing features.

    Features created:
    - Delta_Opp_F0: Rate of approach d(Opp_F0)/dt in cm/s (negative = closing fast)
    - Opp_Lat_Delta: Lateral sensor difference (left - right in cm)
    - Opp_Bearing_Est_Deg: Proximity-weighted angular target centroid in degrees (-90 to +90)

    Returns:
        Tuple of (enriched_dataframe, list_of_sensor_feature_column_names)
    """
    df = agent_df.copy()

    # 1. First-Order Temporal Velocity: d(Opp_F0)/dt
    if "Opp_F0" in df.columns and "Match_ID" in df.columns and "Bot_ID" in df.columns:
        df["Delta_Opp_F0"] = df.groupby(["Match_ID", "Bot_ID"])["Opp_F0"].diff().fillna(0.0) / dt
    elif "Opp_F0" in df.columns:
        df["Delta_Opp_F0"] = df["Opp_F0"].diff().fillna(0.0) / dt

    # 2. Lateral Spatial Asymmetry
    if "Opp_L18" in df.columns and "Opp_R18" in df.columns:
        df["Opp_Lat_Delta"] = df["Opp_L18"] - df["Opp_R18"]
    elif "Opp_L15" in df.columns and "Opp_R15" in df.columns:
        if "Opp_L22_5" in df.columns and "Opp_R22_5" in df.columns:
            df["Opp_Lat_Delta"] = (df["Opp_L15"] + df["Opp_L22_5"]) / 2.0 - (df["Opp_R15"] + df["Opp_R22_5"]) / 2.0
        else:
            df["Opp_Lat_Delta"] = df["Opp_L15"] - df["Opp_R15"]

    # 3. Estimated Target Bearing (Weighted Angular Centroid in Degrees)
    sensor_angle_map = {
        "Opp_L90": 90.0,
        "Opp_L22_5": 22.5,
        "Opp_L18": 18.0,
        "Opp_L15": 15.0,
        "Opp_F0": 0.0,
        "Opp_R15": -15.0,
        "Opp_R18": -18.0,
        "Opp_R22_5": -22.5,
        "Opp_R90": -90.0,
    }
    avail_bearing_sensors = [col for col in sensor_angle_map if col in df.columns]
    if avail_bearing_sensors:
        weights_sum = np.zeros(len(df), dtype=float)
        weighted_angle_sum = np.zeros(len(df), dtype=float)
        for col in avail_bearing_sensors:
            vals = df[col].to_numpy()
            valid_mask = vals > 0.0
            weights = np.where(valid_mask, 1.0 / np.maximum(vals, 1.0), 0.0)
            weights_sum += weights
            weighted_angle_sum += weights * sensor_angle_map[col]
        df["Opp_Bearing_Est_Deg"] = np.where(weights_sum > 0.0, weighted_angle_sum / np.maximum(weights_sum, 1e-6), 0.0)

    feature_cols = [c for c in df.columns if c.startswith("IR_Edge_") or c.startswith("Opp_") or c.startswith("Delta_")]
    return df, feature_cols


def compute_mcu_benchmarks(
    tree_depth: int,
    n_leaves: int,
    n_features: int = 10,
) -> Dict[str, Any]:
    """Compute physical resource consumption and execution timing for 16 MHz 8-bit AVR baseline (ATmega328P).

    Returns:
        latency_us: float (0.75 μs per tree depth level)
        sram_bytes: int (4 bytes per 32-bit float sensor feature)
        flash_bytes: int (24 bytes per decision node under -Os optimization)
        flash_kb: float
        bandwidth_khz: int (1000 / latency_us)
        headroom_x: int (50000 μs control loop / latency_us)
    """
    depth = max(int(tree_depth), 1)
    leaves = max(int(n_leaves), 1)
    features = max(int(n_features), 1)
    total_nodes = 2 * leaves - 1

    latency_us = depth * 0.75
    sram_bytes = features * 4
    flash_bytes = total_nodes * 24
    flash_kb = flash_bytes / 1024.0
    bandwidth_khz = int(1000.0 / latency_us) if latency_us > 0 else 0
    headroom_x = int(50000.0 / latency_us) if latency_us > 0 else 0

    return {
        "latency_us": latency_us,
        "sram_bytes": sram_bytes,
        "flash_bytes": flash_bytes,
        "flash_kb": flash_kb,
        "bandwidth_khz": bandwidth_khz,
        "headroom_x": headroom_x,
    }


def evaluate_sensor_action(
    model: Any,
    sensor_readings: Dict[str, float],
    feature_names: Optional[List[str]] = None,
    dt: float = SIMULATION_DT_S,
) -> str:
    """Evaluate a single dictionary of raw sensor readings and return the predicted combat state string.

    Automatically calculates derived features (Delta_Opp_F0, Opp_Lat_Delta, Opp_Bearing_Est_Deg)
    if required by the model.
    """
    df_single = pd.DataFrame([sensor_readings])
    enriched_df, available_feats = engineer_features(df_single, dt=dt)

    if feature_names is not None:
        target_cols = feature_names
    elif hasattr(model, "feature_names_in_"):
        target_cols = list(model.feature_names_in_)
    else:
        target_cols = available_feats

    # Ensure all required features are present (defaulting to 0.0 if omitted)
    for col in target_cols:
        if col not in enriched_df.columns:
            enriched_df[col] = 0.0

    X_single = enriched_df[target_cols]
    pred = model.predict(X_single)[0]
    return str(pred)


# ==============================================================================
# 1. MULTI-MODEL BENCHMARKING & HYPERPARAMETER TUNING ENGINE
# ==============================================================================
def train_and_benchmark_models(
    agent_data: Union[str, Path, pd.DataFrame],
    test_size: float = 0.20,
    random_state: int = 42,
    selected_models: Optional[List[str]] = None,
    progress_callback: Optional[Callable[[int, int, str], None]] = None,
) -> Dict[str, Any]:
    """Train and optimize Decision Tree, Logistic Regression, Random Forest, and Gradient Boosting.

    Executes 5-fold cross-validated GridSearchCV for each model architecture and compiles
    an empirical evaluation leaderboard.
    """
    if isinstance(agent_data, (str, Path)):
        df = pd.read_parquet(agent_data)
    else:
        df = agent_data.copy()

    # 1. Validate Feature Isolation (Strictly zero spatial coordinates)
    forbidden_cols = ["Pos_X", "Pos_Y", "Dist_To_Center", "Heading_Deg"]
    for col in forbidden_cols:
        if col in df.columns:
            raise ValueError(f"Target leakage detected! {col} found in Agent Dataset.")

    # 1.1 Isolate Candidate Bot (Bot_A) to prevent contradictory policy label contamination
    if "Bot_ID" in df.columns:
        df_candidate = df[df["Bot_ID"] == "Bot_A"]
        if len(df_candidate) >= 50:
            df = df_candidate

    # 2. Feature Engineering: First-Order Temporal Velocity & Lateral Spatial Asymmetry
    df, feature_cols = engineer_features(df)

    if not feature_cols:
        raise ValueError("No valid sensor features found (IR_Edge_*, Opp_*, or Delta_*).")

    if "Current_State" not in df.columns:
        raise ValueError("Target column 'Current_State' not found in dataset.")

    X = df[feature_cols].copy()
    y = df["Current_State"].copy()

    # Handle rare classes if present (requires at least 5 samples for 5-fold CV)
    class_counts = y.value_counts()
    valid_classes = class_counts[class_counts >= 5].index
    mask = y.isin(valid_classes)
    X = X[mask]
    y = y[mask]

    # 3. Stratified Train/Test Split (Utilizing 100% of simulation data)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)
    classes = sorted(list(y.unique()))

    # 4. Model Architectures & Search Spaces
    model_configs: Dict[str, Dict[str, Any]] = {
        "Decision Tree": {
            "type": "White-Box Decision Tree",
            "architecture": "Single Pruned CART Tree",
            "estimator": DecisionTreeClassifier(random_state=random_state, class_weight="balanced"),
            "param_grid": {
                "max_depth": [4, 6, 8],
                "min_samples_split": [10, 20],
                "criterion": ["gini", "entropy"],
            },
            "is_primary": True,
            "transpilable": "Yes (strategy_config.h)",
        },
        "Logistic Regression": {
            "type": "Linear Baseline",
            "architecture": "Multinomial Softmax with L2 Regularization",
            "estimator": Pipeline([
                ("scaler", StandardScaler()),
                ("clf", LogisticRegression(class_weight="balanced", random_state=random_state, max_iter=1000)),
            ]),
            "param_grid": {
                "clf__C": [0.1, 1.0, 10.0],
            },
            "is_primary": False,
            "transpilable": "Matrix Multiplication Required",
        },
        "Random Forest": {
            "type": "Ensemble (Bagging)",
            "architecture": "Parallel Ensemble of Randomized Decision Trees",
            "estimator": RandomForestClassifier(random_state=random_state, class_weight="balanced", n_jobs=-1),
            "param_grid": {
                "n_estimators": [50],
                "max_depth": [6, 10],
                "min_samples_split": [10, 20],
            },
            "is_primary": False,
            "transpilable": "High Flash Memory / RAM Footprint",
        },
        "Gradient Boosting": {
            "type": "Ensemble (Boosting)",
            "architecture": "Sequential Residual Gradient Boosting Trees",
            "estimator": GradientBoostingClassifier(random_state=random_state),
            "param_grid": {
                "n_estimators": [40],
                "learning_rate": [0.1],
                "max_depth": [3],
            },
            "is_primary": False,
            "transpilable": "High Flash Memory / Floating-Point Overhead",
        },
    }

    if selected_models:
        model_configs = {k: v for k, v in model_configs.items() if k in selected_models}
    if not model_configs:
        raise ValueError("At least one model architecture must be selected for training.")

    model_results: Dict[str, Dict[str, Any]] = {}
    leaderboard_records: List[Dict[str, Any]] = []
    total_models = len(model_configs)

    for idx, (name, cfg) in enumerate(model_configs.items()):
        if progress_callback:
            progress_callback(idx, total_models, f"Tuning & Evaluating {name} ({idx + 1}/{total_models})...")
        t0 = time.time()
        gs = GridSearchCV(
            estimator=cfg["estimator"],
            param_grid=cfg["param_grid"],
            scoring="f1_macro",
            cv=cv,
            n_jobs=-1,
        )
        gs.fit(X_train, y_train)
        t_dur = time.time() - t0

        best_m = gs.best_estimator_
        y_pred = best_m.predict(X_test)

        test_acc = float(accuracy_score(y_test, y_pred))
        test_prec = float(precision_score(y_test, y_pred, average="macro", zero_division=0))
        test_rec = float(recall_score(y_test, y_pred, average="macro", zero_division=0))
        test_f1 = float(f1_score(y_test, y_pred, average="macro", zero_division=0))
        test_f1_w = float(f1_score(y_test, y_pred, average="weighted", zero_division=0))

        cm = confusion_matrix(y_test, y_pred, labels=classes)
        cm_norm = confusion_matrix(y_test, y_pred, labels=classes, normalize="true")
        report = classification_report(y_test, y_pred, labels=classes, output_dict=True, zero_division=0)

        # Feature importances / coefficients extraction
        feat_importances: Dict[str, float] = {}
        if hasattr(best_m, "feature_importances_"):
            raw_imp = best_m.feature_importances_
            feat_importances = {feat: float(imp) for feat, imp in zip(feature_cols, raw_imp)}
        elif hasattr(best_m, "named_steps") and hasattr(best_m.named_steps["clf"], "coef_"):
            coef = best_m.named_steps["clf"].coef_
            mean_abs_coef = np.mean(np.abs(coef), axis=0)
            total = float(np.sum(mean_abs_coef)) if np.sum(mean_abs_coef) > 0 else 1.0
            feat_importances = {feat: float(val / total) for feat, val in zip(feature_cols, mean_abs_coef)}
        else:
            feat_importances = {feat: 1.0 / len(feature_cols) for feat in feature_cols}

        feat_importances = dict(sorted(feat_importances.items(), key=lambda item: item[1], reverse=True))

        # Permutation feature importance on test partition
        try:
            perm = permutation_importance(best_m, X_test, y_test, n_repeats=5, random_state=random_state, scoring="f1_macro", n_jobs=-1)
            perm_importances = {feat: round(float(imp), 4) for feat, imp in zip(feature_cols, perm.importances_mean)}
            perm_importances = dict(sorted(perm_importances.items(), key=lambda item: item[1], reverse=True))
        except Exception:
            perm_importances = feat_importances

        # Extract 5-Fold cross-validation split scores for hypothesis testing
        best_idx = gs.best_index_
        fold_scores = [float(gs.cv_results_[f"split{k}_test_score"][best_idx]) for k in range(5)]

        m_dict: Dict[str, Any] = {
            "name": name,
            "model": best_m,
            "type": cfg["type"],
            "architecture": cfg["architecture"],
            "best_params": gs.best_params_,
            "best_cv_f1_macro": float(gs.best_score_),
            "cv_fold_scores": fold_scores,
            "test_accuracy": test_acc,
            "test_precision_macro": test_prec,
            "test_recall_macro": test_rec,
            "test_f1_macro": test_f1,
            "test_f1_weighted": test_f1_w,
            "training_time_s": float(t_dur),
            "classes": classes,
            "confusion_matrix": cm.tolist(),
            "confusion_matrix_normalized": cm_norm.tolist(),
            "classification_report": report,
            "feature_names": feature_cols,
            "feature_importances": feat_importances,
            "permutation_importances": perm_importances,
            "transpilable": cfg["transpilable"],
            "y_pred": y_pred,
        }

        # Specific extraction for Decision Tree
        if name == "Decision Tree":
            dt_model: DecisionTreeClassifier = best_m
            threshold_records = []
            tree = dt_model.tree_
            for node_id in range(tree.node_count):
                if tree.children_left[node_id] != tree.children_right[node_id]:  # Not a leaf
                    feat_idx = tree.feature[node_id]
                    feat_name = feature_cols[feat_idx]
                    thresh_val = tree.threshold[node_id]
                    threshold_records.append({
                        "Node_ID": node_id,
                        "Sensor_Feature": feat_name,
                        "Comparison": "<=",
                        "Threshold_Value": round(float(thresh_val), 4),
                        "Samples": int(tree.n_node_samples[node_id]),
                        "Impurity": round(float(tree.impurity[node_id]), 4),
                    })
            df_thresholds = pd.DataFrame(threshold_records)
            df_hardware_summary = build_hardware_summary_table(df_thresholds, feat_importances)
            tree_rules_text = export_text(dt_model, feature_names=feature_cols)

            # Minimal Cost-Complexity Pruning Sensitivity Analysis (Occam's Razor)
            pruning_path = dt_model.cost_complexity_pruning_path(X_train, y_train)
            ccp_alphas = pruning_path.ccp_alphas
            ccp_alphas = ccp_alphas[ccp_alphas >= 0.0]
            if len(ccp_alphas) > 20:
                indices = np.linspace(0, len(ccp_alphas) - 1, 20, dtype=int)
                eval_alphas = np.unique(ccp_alphas[indices])
            else:
                eval_alphas = ccp_alphas

            pruning_alphas = []
            pruning_train_f1 = []
            pruning_test_f1 = []
            pruning_node_counts = []

            for alpha in eval_alphas:
                pruned_dt = DecisionTreeClassifier(
                    criterion=dt_model.criterion,
                    max_depth=dt_model.max_depth,
                    min_samples_split=dt_model.min_samples_split,
                    min_samples_leaf=dt_model.min_samples_leaf,
                    class_weight=dt_model.class_weight,
                    ccp_alpha=float(alpha),
                    random_state=random_state,
                )
                pruned_dt.fit(X_train, y_train)
                y_tr_pred = pruned_dt.predict(X_train)
                y_te_pred = pruned_dt.predict(X_test)
                tr_f1 = float(f1_score(y_train, y_tr_pred, average="macro", zero_division=0))
                te_f1 = float(f1_score(y_test, y_te_pred, average="macro", zero_division=0))

                pruning_alphas.append(float(alpha))
                pruning_train_f1.append(round(tr_f1, 4))
                pruning_test_f1.append(round(te_f1, 4))
                pruning_node_counts.append(int(pruned_dt.tree_.node_count))

            m_dict["thresholds_table"] = df_thresholds
            m_dict["hardware_summary_table"] = df_hardware_summary
            m_dict["tree_rules_text"] = tree_rules_text
            m_dict["pruning_curve"] = {
                "alphas": pruning_alphas,
                "train_f1": pruning_train_f1,
                "test_f1": pruning_test_f1,
                "node_counts": pruning_node_counts,
                "optimal_alpha": float(getattr(dt_model, "ccp_alpha", 0.0)),
                "unpruned_nodes": int(dt_model.tree_.node_count),
            }

        model_results[name] = m_dict

        leaderboard_records.append({
            "Rank": 0,
            "Model": name,
            "Model Type": cfg["type"],
            "5-Fold CV Macro F1": round(float(gs.best_score_), 4),
            "Test Macro F1": round(test_f1, 4),
            "Test Accuracy": round(test_acc, 4),
            "Test Precision (M)": round(test_prec, 4),
            "Test Recall (M)": round(test_rec, 4),
            "Weighted F1": round(test_f1_w, 4),
            "Training Time (s)": round(t_dur, 2),
        })

    # Paired K-Fold Hypothesis Testing against primary Decision Tree (Baseline)
    dt_fold_scores = model_results.get("Decision Tree", {}).get("cv_fold_scores", [])
    for rec in leaderboard_records:
        m_name = rec["Model"]
        if m_name in model_results and dt_fold_scores and len(dt_fold_scores) == 5:
            other_folds = model_results[m_name].get("cv_fold_scores", [])
            if m_name == "Decision Tree":
                rec["p-value (vs DT)"] = 1.0000
                rec["Stat. Significant"] = "Reference"
            else:
                diffs = np.array(other_folds) - np.array(dt_fold_scores)
                if np.all(diffs == 0):
                    p_val = 1.0
                else:
                    try:
                        stat, p_val = ttest_rel(other_folds, dt_fold_scores)
                        p_val = float(p_val) if not np.isnan(p_val) else 1.0
                    except Exception:
                        p_val = 1.0
                rec["p-value (vs DT)"] = round(p_val, 4)
                rec["Stat. Significant"] = "Yes (p<0.05)" if p_val < 0.05 else "No (p>=0.05)"
                model_results[m_name]["p_value_vs_dt"] = p_val

    # Compile Leaderboard DataFrame and rank by Test Macro F1
    df_leaderboard = pd.DataFrame(leaderboard_records).sort_values(by="Test Macro F1", ascending=False).reset_index(drop=True)
    df_leaderboard["Rank"] = df_leaderboard.index + 1

    best_model_name = df_leaderboard.iloc[0]["Model"]
    dt_results = model_results.get("Decision Tree")

    return {
        "models": model_results,
        "leaderboard": df_leaderboard,
        "primary_model_results": dt_results,
        "best_model_name": best_model_name,
        "classes": classes,
        "feature_names": feature_cols,
        "X_test": X_test,
        "y_test": y_test,
    }


def train_and_optimize_tree(
    agent_data: Union[str, Path, pd.DataFrame],
    test_size: float = 0.20,
    random_state: int = 42,
) -> Dict[str, Any]:
    """Train and optimize DecisionTreeClassifier strictly on Agent Data using 5-fold CV (Convenience Wrapper)."""
    full_benchmark = train_and_benchmark_models(
        agent_data,
        test_size=test_size,
        random_state=random_state,
        selected_models=["Decision Tree"],
    )
    dt_res = full_benchmark["primary_model_results"]
    dt_res["full_benchmark"] = full_benchmark
    return dt_res


# ==============================================================================
# 2. HARDWARE SUMMARY AGGREGATION UTILITY
# ==============================================================================
def build_hardware_summary_table(
    df_thresholds: pd.DataFrame,
    feature_importances: Dict[str, float],
) -> pd.DataFrame:
    """Aggregate node-level decision tree splits into an intuitive per-sensor hardware summary."""
    sensor_metadata = {
        "Opp_F0": ("Front Optical Ray (0°)", "Front Target Lock (> 0.0 cm)", "Front Bull-Rush / Direct Attack"),
        "Opp_L15": ("Inner-Left Optical Ray (15°)", "Inner-Left Target Lock (> 0.0 cm)", "Tight Left Arc / Alignment"),
        "Opp_R15": ("Inner-Right Optical Ray (15°)", "Inner-Right Target Lock (> 0.0 cm)", "Tight Right Arc / Alignment"),
        "Opp_L18": ("Inner-Left Optical Ray (18°)", "Inner-Left Target Lock (> 0.0 cm)", "Left Orienting Pivot"),
        "Opp_R18": ("Inner-Right Optical Ray (18°)", "Inner-Right Target Lock (> 0.0 cm)", "Right Orienting Pivot"),
        "Opp_L22_5": ("Mid-Left Optical Ray (22.5°)", "Mid-Left Target Lock (> 0.0 cm)", "Left Flank Intercept"),
        "Opp_R22_5": ("Mid-Right Optical Ray (22.5°)", "Mid-Right Target Lock (> 0.0 cm)", "Right Flank Intercept"),
        "Opp_L90": ("Flank Optical Ray (90° Left)", "90° Left Target Lock (> 0.0 cm)", "Perpendicular Left Spin Lock"),
        "Opp_R90": ("Flank Optical Ray (90° Right)", "90° Right Target Lock (> 0.0 cm)", "Perpendicular Right Spin Lock"),
        "IR_Edge_FL": ("Front-Left Line Reflectance", "White Line Trigger (>= 0.70)", "Immediate Reverse Escape Turn"),
        "IR_Edge_FR": ("Front-Right Line Reflectance", "White Line Trigger (>= 0.70)", "Immediate Reverse Escape Turn"),
        "Delta_Opp_F0": ("Closing Velocity", "Rate of Approach (cm/s)", "Charge Intercept / Evasion Trigger"),
        "Opp_Lat_Delta": ("Lateral Angular Offset (L - R)", "Asymmetric Target Bearing", "Differential Turn Bias"),
        "Opp_Bearing_Est_Deg": ("Estimated Target Bearing Angle", "Array Weighted Centroid (-90° to +90°)", "Target Orientation / Steering Bias"),
    }

    summary_records = []
    for feat, imp in feature_importances.items():
        feat_nodes = (
            df_thresholds[df_thresholds["Sensor_Feature"] == feat]
            if (df_thresholds is not None and not df_thresholds.empty)
            else pd.DataFrame()
        )
        type_desc, gate_desc, role_desc = sensor_metadata.get(
            feat, ("Sumo Sensor", "> 0.0", "Tactical Trigger")
        )

        if not feat_nodes.empty:
            vals = sorted(feat_nodes["Threshold_Value"].unique())
            if "IR_Edge" in feat:
                edge_vals = [v for v in vals if v >= 0.50]
                op_thresh = (
                    f"{edge_vals[0]:.4f} (Reflectance)"
                    if edge_vals
                    else f"{vals[-1]:.4f} (Reflectance)"
                )
            else:
                dist_vals = [v for v in vals if v > 0.0]
                if dist_vals:
                    op_thresh = ", ".join([f"{v:.1f} cm" for v in dist_vals[:3]])
                else:
                    op_thresh = f"{vals[0]:.2f} cm (Presence Gate)"
            n_nodes = len(feat_nodes)
        else:
            op_thresh = "N/A (No direct split)"
            n_nodes = 0

        summary_records.append({
            "Sensor_Feature": feat,
            "Sensor_Type": type_desc,
            "Detection_Gate": gate_desc,
            "Operational_Threshold": op_thresh,
            "Tactical_Role": role_desc,
            "Gini_Importance": f"{imp:.4f} ({imp*100:.1f}%)",
            "Decision_Nodes": n_nodes,
        })

    df_res = pd.DataFrame(summary_records)
    if "Sensor_Feature" in df_res.columns:
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
        present = [f for f in canonical_order if f in df_res["Sensor_Feature"].values]
        remaining = [f for f in df_res["Sensor_Feature"].values if f not in present]
        df_res = df_res.set_index("Sensor_Feature").loc[present + remaining].reset_index()
    return df_res


# ==============================================================================
# 3. EMBEDDED C++ STRATEGY HEADER TRANSPILER
# ==============================================================================
def generate_cpp_header(
    model: DecisionTreeClassifier,
    feature_names: List[str],
    classes: List[str],
    weight_class: str = "KIT_1KG",
    output_path: Optional[Union[str, Path]] = None,
) -> str:
    """Transpile trained Decision Tree rules into an Arduino/ESP32 C++ header (strategy_config.h)."""
    tree_ = model.tree_
    feature_name = [
        feature_names[i] if i != -2 else "undefined!"
        for i in tree_.feature
    ]

    header_lines = [
        "/**",
        " * ==============================================================================",
        " * AUTONOMOUS SUMO ROBOT DIGITAL TWIN - FIRMWARE STRATEGY CONFIGURATION",
        " * Generated automatically by Decision Tree Transpiler",
        f" * Target Weight Class: {weight_class}",
        " * Architecture: Arduino / ESP32 C++ Header",
        " * ==============================================================================",
        " */",
        "",
        "#ifndef STRATEGY_CONFIG_H",
        "#define STRATEGY_CONFIG_H",
        "",
        "#include <stdint.h>",
        "",
        "#ifdef __cplusplus",
        'extern "C" {',
        "#endif",
        "",
        "// ==============================================================================",
        "// 1. STATE ENUM DECLARATION",
        "// ==============================================================================",
        "typedef enum {",
        "    STATE_SEARCH = 0,",
        "    STATE_TRACK = 1,",
        "    STATE_ATTACK = 2,",
        "    STATE_EDGE_RECOVERY = 3,",
        "    STATE_EVADE = 4",
        "} BotState;",
        "",
        "// ==============================================================================",
        "// 2. CALIBRATED SPEED CONSTANTS (PWM 0-255)",
        "// ==============================================================================",
        f"#define PWM_SEARCH_SPEED       {PWM_SEARCH_FAST}",
        f"#define PWM_ATTACK_SPEED       {PWM_ATTACK_FULL}",
        f"#define PWM_REVERSE_SPEED_L    {PWM_RECOVERY_LEFT}",
        f"#define PWM_REVERSE_SPEED_R    {PWM_RECOVERY_RIGHT}",
        f"#define EDGE_THRESHOLD_VAL     {EDGE_LINE_THRESHOLD}f",
        "// ==============================================================================",
        "// 3. SENSOR INPUT STRUCT",
        "// ==============================================================================",
        "typedef struct {",
    ]

    sensor_field_comments = {
        "IR_Edge_FL": "Downward line reflectance front-left [0.0, 1.0] (white border >= 0.70)",
        "IR_Edge_FR": "Downward line reflectance front-right [0.0, 1.0] (white border >= 0.70)",
        "Opp_F0": "Center front optical distance in cm (-1.0 = no target detected)",
        "Opp_L15": "Inner-left (15 deg) optical distance in cm (-1.0 = no target)",
        "Opp_R15": "Inner-right (15 deg) optical distance in cm (-1.0 = no target)",
        "Opp_L18": "Inner-left (18 deg) optical distance in cm (-1.0 = no target)",
        "Opp_R18": "Inner-right (18 deg) optical distance in cm (-1.0 = no target)",
        "Opp_L22_5": "Mid-left (22.5 deg) optical distance in cm (-1.0 = no target)",
        "Opp_R22_5": "Mid-right (22.5 deg) optical distance in cm (-1.0 = no target)",
        "Opp_L90": "Flank 90 deg left optical distance in cm (-1.0 = no target)",
        "Opp_R90": "Flank 90 deg right optical distance in cm (-1.0 = no target)",
        "Delta_Opp_F0": "Closing velocity d(Opp_F0)/dt in cm/s (negative = closing fast)",
        "Opp_Lat_Delta": "Lateral angular difference (Opp_Left - Opp_Right in cm)",
        "Opp_Bearing_Est_Deg": "Proximity-weighted target bearing angle in degrees (-90 to +90)",
    }

    for feat in feature_names:
        comment = sensor_field_comments.get(feat, "Calibrated sensor feature")
        header_lines.append(f"    float {feat};  // {comment}")

    header_lines.extend([
        "} SumoSensors;",
        "",
        "// ==============================================================================",
        "// 4. EMBEDDED DECISION TREE INFERENCE ENGINE",
        "// ==============================================================================",
        "/**",
        " * Evaluates sensor readings and returns optimal combat state.",
        " * Execution latency: < 5 microseconds (deterministic O(depth)).",
        " */",
        "static inline BotState evaluate_strategy(const SumoSensors* s) {",
    ])

    def recurse(node: int, depth: int) -> List[str]:
        indent = "    " * (depth + 1)
        lines = []
        if tree_.children_left[node] == tree_.children_right[node]:  # Leaf
            class_idx = int(np.argmax(tree_.value[node]))
            class_name = classes[class_idx]
            enum_val = f"STATE_{class_name}"
            lines.append(f"{indent}return {enum_val};")
        else:
            name = feature_name[node]
            threshold = tree_.threshold[node]
            lines.append(f"{indent}if (s->{name} <= {threshold:.4f}f) {{")
            lines.extend(recurse(tree_.children_left[node], depth + 1))
            lines.append(f"{indent}}} else {{")
            lines.extend(recurse(tree_.children_right[node], depth + 1))
            lines.append(f"{indent}}}")
        return lines

    header_lines.extend(recurse(0, 1))

    header_lines.extend([
        "}",
        "",
        "#ifdef __cplusplus",
        "}",
        "#endif",
        "",
        "#endif // STRATEGY_CONFIG_H",
        "",
    ])

    cpp_code = "\n".join(header_lines)

    if output_path is not None:
        p = Path(output_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "w") as f:
            f.write(cpp_code)

    return cpp_code


def generate_cpp_harness(
    robot_class: str = "KIT_1KG",
    header_filename: str = "strategy_config.h",
    dataset_name: str = "telemetry_dataset",
    output_path: Optional[Union[str, Path]] = None,
) -> str:
    """Generate production-ready Arduino/ESP32 C++ main control loop firmware harness."""
    is_mega = "3KG" in robot_class.upper() or "MEGA" in robot_class.upper()
    if is_mega:
        pin_defs = """// Downward Line Reflectance Sensors (Fairchild QRE1113 Analog Voltage Dividers)
#define PIN_EDGE_FL     A0  // Front-Left Line Sensor (ADC count 0-1023)
#define PIN_EDGE_FR     A1  // Front-Right Line Sensor (ADC count 0-1023)

// Opponent Optical Proximity Sensors (Omron E3Z-D62 Triangulation)
#define PIN_OPP_F0      A2  // Center Front Optical Ray (0°)
#define PIN_OPP_L15     A3  // Inner-Left Optical Ray (15°)
#define PIN_OPP_R15     A4  // Inner-Right Optical Ray (15°)
#define PIN_OPP_L22_5   A5  // Outer-Left Optical Ray (22.5°)
#define PIN_OPP_R22_5   A6  // Outer-Right Optical Ray (22.5°)
#define PIN_OPP_L90     A7  // Lateral Flank Optical Ray (90° Left)
#define PIN_OPP_R90     A8  // Lateral Flank Optical Ray (90° Right)"""

        sensor_reads = """    // E3Z-D62 Optical Range: Calibrated distance in cm (-1.0 = out of range / cone)
    sensors.Opp_F0     = read_optical_cm(PIN_OPP_F0);
    sensors.Opp_L15    = read_optical_cm(PIN_OPP_L15);
    sensors.Opp_R15    = read_optical_cm(PIN_OPP_R15);
    sensors.Opp_L22_5  = read_optical_cm(PIN_OPP_L22_5);
    sensors.Opp_R22_5  = read_optical_cm(PIN_OPP_R22_5);
    sensors.Opp_L90    = read_optical_cm(PIN_OPP_L90);
    sensors.Opp_R90    = read_optical_cm(PIN_OPP_R90);"""

        bearing_func = """/**
 * Calculates proximity-weighted angular target centroid across active optical rays.
 * Transfer Function: Harmonic centroid theta = sum(w_i * theta_i) / sum(w_i), where w_i = 1 / dist_i.
 */
static inline float compute_bearing(const SumoSensors* s) {
    float w_sum = 0.0f;
    float weighted_angle = 0.0f;
    if (s->Opp_F0 > 0.0f)     { float w = 1.0f / s->Opp_F0;    w_sum += w; weighted_angle += w * 0.0f; }
    if (s->Opp_L15 > 0.0f)    { float w = 1.0f / s->Opp_L15;   w_sum += w; weighted_angle += w * 15.0f; }
    if (s->Opp_R15 > 0.0f)    { float w = 1.0f / s->Opp_R15;   w_sum += w; weighted_angle += w * -15.0f; }
    if (s->Opp_L22_5 > 0.0f)  { float w = 1.0f / s->Opp_L22_5; w_sum += w; weighted_angle += w * 22.5f; }
    if (s->Opp_R22_5 > 0.0f)  { float w = 1.0f / s->Opp_R22_5; w_sum += w; weighted_angle += w * -22.5f; }
    if (s->Opp_L90 > 0.0f)    { float w = 1.0f / s->Opp_L90;   w_sum += w; weighted_angle += w * 90.0f; }
    if (s->Opp_R90 > 0.0f)    { float w = 1.0f / s->Opp_R90;   w_sum += w; weighted_angle += w * -90.0f; }
    return (w_sum > 0.001f) ? (weighted_angle / w_sum) : 0.0f;
}"""
        lat_delta_calc = "sensors.Opp_Lat_Delta = (sensors.Opp_L15 + sensors.Opp_L22_5) / 2.0f - (sensors.Opp_R15 + sensors.Opp_R22_5) / 2.0f;"
    else:
        pin_defs = """// Downward Line Reflectance Sensors (Fairchild QRE1113 Analog Voltage Dividers)
#define PIN_EDGE_FL     A0  // Front-Left Line Sensor (ADC count 0-1023)
#define PIN_EDGE_FR     A1  // Front-Right Line Sensor (ADC count 0-1023)

// Opponent Optical Proximity Sensors (Omron E3Z-D62 Triangulation)
#define PIN_OPP_F0      A2  // Center Front Optical Ray (0°)
#define PIN_OPP_L18     A3  // Inner-Left Optical Ray (18°)
#define PIN_OPP_R18     A4  // Inner-Right Optical Ray (18°)
#define PIN_OPP_L90     A5  // Lateral Flank Optical Ray (90° Left)
#define PIN_OPP_R90     A6  // Lateral Flank Optical Ray (90° Right)"""

        sensor_reads = """    // E3Z-D62 Optical Range: Calibrated distance in cm (-1.0 = out of range / cone)
    sensors.Opp_F0     = read_optical_cm(PIN_OPP_F0);
    sensors.Opp_L18    = read_optical_cm(PIN_OPP_L18);
    sensors.Opp_R18    = read_optical_cm(PIN_OPP_R18);
    sensors.Opp_L90    = read_optical_cm(PIN_OPP_L90);
    sensors.Opp_R90    = read_optical_cm(PIN_OPP_R90);"""

        bearing_func = """/**
 * Calculates proximity-weighted angular target centroid across active optical rays.
 * Transfer Function: Harmonic centroid theta = sum(w_i * theta_i) / sum(w_i), where w_i = 1 / dist_i.
 */
static inline float compute_bearing(const SumoSensors* s) {
    float w_sum = 0.0f;
    float weighted_angle = 0.0f;
    if (s->Opp_F0 > 0.0f)   { float w = 1.0f / s->Opp_F0;  w_sum += w; weighted_angle += w * 0.0f; }
    if (s->Opp_L18 > 0.0f)  { float w = 1.0f / s->Opp_L18; w_sum += w; weighted_angle += w * 18.0f; }
    if (s->Opp_R18 > 0.0f)  { float w = 1.0f / s->Opp_R18; w_sum += w; weighted_angle += w * -18.0f; }
    if (s->Opp_L90 > 0.0f)  { float w = 1.0f / s->Opp_L90; w_sum += w; weighted_angle += w * 90.0f; }
    if (s->Opp_R90 > 0.0f)  { float w = 1.0f / s->Opp_R90; w_sum += w; weighted_angle += w * -90.0f; }
    return (w_sum > 0.001f) ? (weighted_angle / w_sum) : 0.0f;
}"""
        lat_delta_calc = "sensors.Opp_Lat_Delta = sensors.Opp_L18 - sensors.Opp_R18;"

    harness_code = f"""/**
 * ==============================================================================
 * Autonomous Sumo Robot - Real-World Firmware Integration Harness (main.cpp)
 * Target Microcontrollers: 8-bit AVR (ATmega328P/Mega), ARM Cortex-M (Teensy), ESP32
 * Target Class: {robot_class.upper()}
 * Included Policy Header: "{header_filename}"
 * ==============================================================================
 *
 * SIM-TO-REAL ELECTRICAL PINOUT REFERENCE:
 * - Line Sensors (QRE1113):      Arduino A0, A1      / ESP32 GPIO 36, 39 (Analog ADC)
 * - Optical Sensors (E3Z-D62):   Arduino A2-A8      / ESP32 GPIO 34, 35, 32, 33, 25
 * - Motor H-Bridge (PWM/Dir):    Arduino D5, D6, D9, D10 / ESP32 GPIO 18, 19, 21, 22
 * ==============================================================================
 */
#if defined(ARDUINO)
    #include <Arduino.h>
#else
    // Desktop simulation & unit test mock harness shims
    #include <stdint.h>
    #include <stdlib.h>
    #include <math.h>
    #ifndef HIGH
        #define HIGH 1
        #define LOW 0
        #define OUTPUT 1
        #define INPUT 0
        #define A0 14
        #define A1 15
        #define A2 16
        #define A3 17
        #define A4 18
        #define A5 19
        #define A6 20
        #define A7 21
        #define A8 22
    #endif
    static inline void pinMode(int pin, int mode) {{ (void)pin; (void)mode; }}
    static inline int analogRead(int pin) {{ (void)pin; return 512; }}
    static inline void analogWrite(int pin, int val) {{ (void)pin; (void)val; }}
    static inline void digitalWrite(int pin, int val) {{ (void)pin; (void)val; }}
    static inline void delay(unsigned long ms) {{ (void)ms; }}
    static inline unsigned long millis(void) {{ return 1000; }}
    static inline void delayMicroseconds(unsigned int us) {{ (void)us; }}
#endif
#include "{header_filename}"

// ==============================================================================
// 1. HARDWARE PIN DEFINITIONS
// ==============================================================================
// Dual DC Motor Driver (TB6612FNG / L298N H-Bridge Direction & PWM)
#define PIN_PWM_LEFT    5   // Left Motor Hardware PWM Speed (0-255)
#define PIN_DIR_LEFT    6   // Left Motor Direction (HIGH = Fwd, LOW = Rev)
#define PIN_PWM_RIGHT   9   // Right Motor Hardware PWM Speed (0-255)
#define PIN_DIR_RIGHT   10  // Right Motor Direction (HIGH = Fwd, LOW = Rev)

{pin_defs}

// ==============================================================================
// 2. SENSOR ACQUISITION & CALIBRATION TRANSFER FUNCTIONS
// ==============================================================================
/**
 * Reads analog reflectance voltage from QRE1113 line sensor.
 * Returns normalized reflectance in range [0.0, 1.0] (white border >= 0.70).
 */
static inline float read_line_reflectance(int pin) {{
    int raw_adc = analogRead(pin);  // 10-bit ADC: 0 (black surface) to 1023 (white border)
    return (float)raw_adc / 1023.0f;
}}

/**
 * Reads Omron E3Z-D62 analog optical triangulation output.
 * Converts raw ADC voltage counts to physical distance in centimeters.
 * Transfer function: dist_cm = 1023 / ADC * 2.5 cm (valid from 0 to 50 cm).
 * Returns -1.0 if out of range or beyond optical cone.
 */
static inline float read_optical_cm(int pin) {{
    int raw_adc = analogRead(pin);
    // Raw ADC threshold gate: values below noise floor indicate no reflection
    if (raw_adc <= 40) return -1.0f;
    // Calibrated inverse distance transfer function (ADC counts -> physical cm)
    float dist_cm = 1023.0f / (float)raw_adc * 2.5f;
    return (dist_cm > 0.0f && dist_cm <= 50.0f) ? dist_cm : -1.0f;
}}

{bearing_func}

static float last_opp_f0 = -1.0f;
static unsigned long last_tick_ms = 0;

// ==============================================================================
// 3. MOTOR ACTUATION CONTROLLER (H-BRIDGE PWM & DIRECTION)
// ==============================================================================
void execute_motor_command(BotState state) {{
    switch (state) {{
        case STATE_ATTACK:
            // High-power forward bull-rush: Maximum forward torque
            analogWrite(PIN_PWM_LEFT, PWM_ATTACK_SPEED);
            analogWrite(PIN_PWM_RIGHT, PWM_ATTACK_SPEED);
            digitalWrite(PIN_DIR_LEFT, HIGH);
            digitalWrite(PIN_DIR_RIGHT, HIGH);
            break;
        case STATE_EDGE_RECOVERY:
            // Immediate boundary escape: High-speed reverse retreat away from white border
            analogWrite(PIN_PWM_LEFT, abs(PWM_REVERSE_SPEED_L));
            analogWrite(PIN_PWM_RIGHT, abs(PWM_REVERSE_SPEED_R));
            digitalWrite(PIN_DIR_LEFT, LOW);
            digitalWrite(PIN_DIR_RIGHT, LOW);
            break;
        case STATE_TRACK:
            // Orienting snap turn towards detected opponent bearing
            analogWrite(PIN_PWM_LEFT, PWM_SEARCH_SPEED);
            analogWrite(PIN_PWM_RIGHT, PWM_SEARCH_SPEED / 2);
            digitalWrite(PIN_DIR_LEFT, HIGH);
            digitalWrite(PIN_DIR_RIGHT, LOW);
            break;
        case STATE_EVADE:
            // Tactical arc retreat to break frontal lock and flank opponent
            analogWrite(PIN_PWM_LEFT, PWM_SEARCH_SPEED / 3);
            analogWrite(PIN_PWM_RIGHT, PWM_SEARCH_SPEED);
            digitalWrite(PIN_DIR_LEFT, LOW);
            digitalWrite(PIN_DIR_RIGHT, HIGH);
            break;
        case STATE_SEARCH:
        default:
            // In-place rotation to sweep optical array across 360-degree arena
            analogWrite(PIN_PWM_LEFT, PWM_SEARCH_SPEED);
            analogWrite(PIN_PWM_RIGHT, PWM_SEARCH_SPEED);
            digitalWrite(PIN_DIR_LEFT, HIGH);
            digitalWrite(PIN_DIR_RIGHT, LOW);
            break;
    }}
}}

// ==============================================================================
// 4. EMBEDDED MAIN SETUP & 20 HZ DETERMINISTIC CONTROL LOOP
// ==============================================================================
void setup() {{
    pinMode(PIN_PWM_LEFT, OUTPUT);
    pinMode(PIN_DIR_LEFT, OUTPUT);
    pinMode(PIN_PWM_RIGHT, OUTPUT);
    pinMode(PIN_DIR_RIGHT, OUTPUT);

    // Official RoboGames Rulebook Article 8: 5.0-second safety countdown delay
    delay(5000);
    last_tick_ms = millis();
}}

void loop() {{
    // Enforce 20 Hz deterministic tick rate (exact 50 ms period)
    unsigned long current_ms = millis();
    unsigned long elapsed = current_ms - last_tick_ms;
    if (elapsed < 50) {{
        delayMicroseconds((50 - elapsed) * 1000);
    }}
    float dt = (millis() - last_tick_ms) / 1000.0f;
    if (dt <= 0.0f) dt = 0.05f;
    last_tick_ms = millis();

    // A. Read calibrated hardware sensor inputs
    SumoSensors sensors;
    sensors.IR_Edge_FL = read_line_reflectance(PIN_EDGE_FL);
    sensors.IR_Edge_FR = read_line_reflectance(PIN_EDGE_FR);

{sensor_reads}

    // B. Calculate dynamic engineered features
    // First-order closing velocity temporal derivative: d(Opp_F0)/dt (cm/s)
    if (last_opp_f0 > 0.0f && sensors.Opp_F0 > 0.0f) {{
        sensors.Delta_Opp_F0 = (sensors.Opp_F0 - last_opp_f0) / dt;
    }} else {{
        sensors.Delta_Opp_F0 = 0.0f;
    }}
    last_opp_f0 = sensors.Opp_F0;

    // Lateral differential asymmetry
    {lat_delta_calc}

    // Multi-sensor harmonic target bearing estimate (-90° to +90°)
    sensors.Opp_Bearing_Est_Deg = compute_bearing(&sensors);

    // C. Sub-microsecond deterministic decision tree evaluation (< 6 μs, 0 SRAM)
    BotState state = evaluate_strategy(&sensors);

    // D. Actuate dual H-Bridge motors according to predicted state
    execute_motor_command(state);
}}
"""
    if output_path is not None:
        p = Path(output_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(harness_code.strip() + "\n", encoding="utf-8")

    return harness_code


# ==============================================================================
# 4. DECISION TREE ARTIFACT PERSISTENCE (data/dt/)
# ==============================================================================
def save_decision_tree_package(
    dt_res: Dict[str, Any],
    dataset_base: str,
    robot_class: str = "KIT_1KG",
    output_dir: Optional[Union[str, Path]] = None,
) -> Dict[str, Any]:
    """Save comprehensive Decision Tree insights, metadata, metrics, and models to data/dt/.

    Generates:
    - {dataset_base}_dt.json (full serializable metadata, metrics, rules, hardware thresholds, and C++ code)
    - {dataset_base}_dt.pkl (trained scikit-learn model and scaler pipeline)
    """
    out_dir = Path(output_dir) if output_dir is not None else DT_DATA_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    dt_model: DecisionTreeClassifier = dt_res["model"]
    feature_names = dt_res["feature_names"]
    classes = dt_res["classes"]

    # Pre-generate C++ code
    cpp_code = generate_cpp_header(
        model=dt_model,
        feature_names=feature_names,
        classes=classes,
        weight_class=robot_class,
    )

    # Convert DataFrame tables to json-serializable dicts
    thresholds_dict = (
        dt_res["thresholds_table"].to_dict(orient="records")
        if isinstance(dt_res.get("thresholds_table"), pd.DataFrame)
        else []
    )
    hardware_summary_dict = (
        dt_res["hardware_summary_table"].to_dict(orient="records")
        if isinstance(dt_res.get("hardware_summary_table"), pd.DataFrame)
        else []
    )

    conf_mat = (
        dt_res["confusion_matrix"].tolist()
        if isinstance(dt_res.get("confusion_matrix"), np.ndarray)
        else dt_res.get("confusion_matrix", [])
    )

    package_dict = {
        "dataset_name": f"{dataset_base}_dt",
        "dataset_base": dataset_base,
        "robot_class": robot_class,
        "model_name": "Decision Tree",
        "timestamp_iso": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "best_params": dt_res.get("best_params", {}),
        "metrics": {
            "cv_macro_f1": float(dt_res.get("best_cv_f1_macro", dt_res.get("cv_score", 0.0))),
            "test_macro_f1": float(dt_res.get("test_f1_macro", 0.0)),
            "test_accuracy": float(dt_res.get("test_accuracy", 0.0)),
            "test_precision_macro": float(dt_res.get("test_precision_macro", 0.0)),
            "test_recall_macro": float(dt_res.get("test_recall_macro", 0.0)),
            "test_f1_weighted": float(dt_res.get("test_f1_weighted", 0.0)),
            "training_time_s": float(dt_res.get("training_time_s", 0.0)),
        },
        "classes": list(classes),
        "feature_names": list(feature_names),
        "feature_importances": {str(k): float(v) for k, v in dt_res.get("feature_importances", {}).items()},
        "tree_depth": int(dt_model.get_depth()),
        "n_leaves": int(dt_model.get_n_leaves()),
        "tree_rules_text": dt_res.get("tree_rules_text", ""),
        "thresholds_table": thresholds_dict,
        "hardware_summary_table": hardware_summary_dict,
        "classification_report": dt_res.get("classification_report", {}),
        "confusion_matrix": conf_mat,
        "cpp_header_code": cpp_code,
    }

    json_path = out_dir / f"{dataset_base}_dt.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(package_dict, f, indent=2)

    pkl_path = out_dir / f"{dataset_base}_dt.pkl"
    with open(pkl_path, "wb") as f:
        pickle.dump(dt_res, f)

    return {
        "json_path": json_path,
        "pkl_path": pkl_path,
        "package": package_dict,
    }


def load_decision_tree_package(target_path: Union[str, Path]) -> Dict[str, Any]:
    """Load a Decision Tree package (.json or .pkl) from data/dt/."""
    p = Path(target_path)
    if not p.exists():
        raise FileNotFoundError(f"Decision Tree package not found: {p}")

    if p.suffix == ".json":
        with open(p, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data
    elif p.suffix == ".pkl":
        with open(p, "rb") as f:
            data = pickle.load(f)
        return data
    else:
        json_cand = p.with_suffix(".json")
        if json_cand.exists():
            with open(json_cand, "r", encoding="utf-8") as f:
                return json.load(f)
        raise ValueError(f"Unsupported file format for decision tree package: {p}")
