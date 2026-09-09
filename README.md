# Autonomous Sumo Robot Digital Twin Studio

**Project Title:** A Digital Twin Framework for Performance Monitoring and Strategy Evaluation in Autonomous Sumo Robots Using Simulation and Data Analytics  
**Academic Program:** Master of Data Science (MDS) at Universiti Malaya    
**Author:** Kaung Htet Shyan

---

## 1. Project Introduction

The **Autonomous Sumo Robot Digital Twin Studio** is an interactive simulation and end-to-end data science platform for competitive autonomous sumo robots (compliant with official [RoboGames](https://robogames.net/rules/all-sumo.php) rules).

The framework provides:
* A deterministic 2D physics simulation environment (kinematics, optical raycasting, downward reflectance, and SAT rigid-body collisions).
* A strict Sim-to-Real ETL pipeline with schema isolation (preventing target coordinate leakage).
* Multi-model machine learning benchmarking with 5-fold cross-validated `GridSearchCV` and automated white-box Decision Tree C++ firmware transpilation.
* Quantitative match analytics, 2D spatial KDE heatmaps, and a 60 FPS HTML5 Canvas top-down match visualizer.

---

## 2. Prerequisites & System Requirements

* **Operating System:** macOS, Linux, or Windows (WSL recommended)
* **Python:** Version 3.10 or higher (Python 3.11+ recommended)
* **Package Manager:** [`uv`](https://github.com/astral-sh/uv) (recommended) or standard `pip`
* **Optional C Compiler:** `clang` or `gcc` (used by the automated test suite to verify generated C99 firmware headers)

---

## 3. Installation & Setup

### Step 1: Clone the Repository
```bash
git clone https://github.com/connorshyan/SumoDigitalTwin.git
cd SumoDigitalTwin
```

### Step 2: Create Virtual Environment and Install Dependencies

Using `uv` (recommended):
```bash
uv venv
uv pip install -e .
```

Alternatively, using standard Python `venv`:
```bash
python3 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -e .
```

---

## 4. How to Run the Project

### 4.1 Launching the Interactive Web Studio (Streamlit)
Launch the 5-Stage Streamlit Web Application:
```bash
uv run streamlit run app/app.py
```
*(Or with active venv: `streamlit run app/app.py`)*

Open your browser and navigate to:
```
http://localhost:8501
```

### 4.2 Running the Automated Verification Test Suite
Execute the comprehensive unit test suite (validates kinematics, physics, ETL schema isolation, decision tree packaging, paired t-tests, and C99 firmware compilation):
```bash
uv run python -m unittest -v tests/test_system_verification.py
```
*(Or with active venv: `python -m unittest -v tests/test_system_verification.py`)*

### 4.3 Running the Research Jupyter Notebooks
Launch JupyterLab to interact with the modular research notebooks:
```bash
uv run jupyter lab
```
Available research notebooks in `notebooks/`:
* `notebooks/data_generation.ipynb`: High-volume batch match simulation and raw Parquet export.
* `notebooks/model_training.ipynb`: Multi-model training, confusion matrices, and decision tree rule extraction.
* `notebooks/strategy_evaluation.ipynb`: Combat performance KPIs, spatial KDE heatmaps, and top-down replays.

---

## 5. Brief Walkthrough of the 5 Project Stages

The web studio is organized into 5 sequential stages mirroring the end-to-end data science lifecycle:

```
Stage 1: Data Gathering ──► Stage 2: Pre-Processing ──► Stage 3: Strategy Modeling
                                                                    │
Stage 5: Real-World Insights ◄── Stage 4: Strategy Analytics ◄──────┘
```

### Stage 1: Data Gathering (Simulation Sandbox)
* **Goal:** Generate rich, reproducible match telemetry across diverse tournament conditions.
* **What it does:** Runs high-frequency (50 ms clock step) Monte Carlo simulations pitting a Candidate Bot against 6 authentic tournament opponent styles (`AGGRESSIVE_CHARGER`, `DEFENSIVE_SWEEPER`, `RANDOM_FLANKER`, `BAIT_AND_SWITCH`, `JUGGERNAUT_PUSH`, and `RANDOM_MIX`) across 5 canonical starting formations (`HEAD_ON`, `ANGLED_INWARD`, `LATERAL_OFFSET`, `SIDE_START`, and `RANDOM_MIX`).
* **Output:** Saves complete match logs to `data/raw/*_raw.parquet`.

### Stage 2: Data Pre-processing (Sim-to-Real ETL & Schema Enforcement)
* **Goal:** Clean and isolate data to guarantee Sim-to-Real fidelity and zero label leakage.
* **What it does:** Performs formal data quality audits (completeness score, sensor domain boundaries, and timestamp monotonicity). Partitions raw telemetry into:
  1. **Agent Dataset (`data/agent/`):** Contains **strictly zero privileged coordinate data** (`Pos_X`, `Pos_Y`, `Heading_Deg`, `Dist_To_Center`). Contains only physical sensor pings and actuator actions.
  2. **Observer Dataset (`data/observer/`):** Retains global ground-truth spatial coordinates and match referee metrics.
* **Output:** Writes optimized Apache Parquet files (`*_agent.parquet`, `*_observer.parquet`, `*_processed.parquet`).

### Stage 3: Strategy Modeling & Multi-Model Benchmark
* **Goal:** Train and evaluate machine learning models to discover optimal combat policies.
* **What it does:**
  * Automatically isolates candidate bot records (`Bot_ID == 'Bot_A'`) to prevent sparring policy label contamination.
  * Engineers first-order closing velocity, lateral asymmetry, and harmonic target bearing angle features.
  * Executes 5-fold Stratified K-Fold `GridSearchCV` across 4 architectures: **Decision Tree**, **Logistic Regression**, **Random Forest**, and **Gradient Boosting**.
  * Computes Permutation Feature Importance and Paired Student's t-test hypothesis testing (p-values) against the baseline Decision Tree.
* **Output:** Saves the optimized model package (`*_dt.json`, `*_dt.pkl`) and exports the trained decision tree rules.

### Stage 4: Strategy Analytics, Spatial Heatmaps, and 2D Replays
* **Goal:** Statistically evaluate tactics and visually inspect match behavior.
* **What it does:**
  * Calculates referee-grade combat KPIs: Time-to-Attack (TTA), Time-to-Ring-Out (TTRO), Ring Center Control %, and win/loss/draw rates.
  * Renders 2D spatial Kernel Density Estimation (KDE) occupancy heatmaps to analyze spatial ring dominance.
  * Provides an interactive 60 FPS HTML5 Canvas top-down match visualizer with timeline scrubbing, playback speed control, and live sensor raycast rendering.

### Stage 5: Real World Application
* **Goal:** Translate data science findings into physical hardware deployments.
* **What it does:**
  * Presents an executive combat summary and strategic win/loss breakdown across opponent architectures.
  * Displays a physical hardware threshold table and microcontroller hardware resource benchmark (profiling execution latency, SRAM usage, Flash ROM footprint, and decision bandwidth on 16 MHz AVR ATmega328P / 32-bit MCUs).
  * Provides a one-click download of the transpiled, zero-dependency C++ `.h` firmware header and production `main.cpp` harness, ready to flash onto an Arduino or ESP32 microcontroller with sub-microsecond deterministic inference.

---

## 6. Project Structure

```
SumoDigitalTwin/
├── app/
│   ├── app.py                      # Centralized 5-Stage Streamlit Studio
│   └── canvas_renderer.py          # HTML5 Canvas 60 FPS top-down match visualizer
├── config/
│   └── config.py                   # Tournament parameters, robot configs & constants
├── data/                           # Partitioned Parquet storage
│   ├── agent/                      # Sim-to-Real compliant agent datasets (0 coordinates)
│   ├── dt/                         # Serialized Decision Tree models & JSON packages
│   ├── observer/                   # Ground-truth referee & spatial telemetry
│   ├── processed/                  # Cleaned full-schema logs
│   └── raw/                        # Raw simulation logs
├── data_pipeline/
│   └── etl_pipeline.py             # Sim-to-Real ETL & schema enforcement engine
├── environment/
│   ├── opponents.py                # 6 Authentic FSM strategy controllers & opening routines
│   └── sumo_env.py                 # 2D Kinematics, SAT collision & raycasting engine
├── models/
│   └── tree_optimizer.py           # Multi-model benchmarking, CV, and C++ transpiler
├── notebooks/                      # Modular research Jupyter notebooks
│   ├── data_generation.ipynb
│   ├── model_training.ipynb
│   └── strategy_evaluation.ipynb
├── tests/
│   └── test_system_verification.py # 34 automated unit tests
├── README.md                       # This setup & walkthrough guide
└── pyproject.toml                  # Python package configuration & dependencies
```

---

## 7. License & Academic Attribution

This project was developed as part of a Master of Data Science (MDS) research project at Universiti Malaya (UM). It is distributed under the MIT License for educational, academic, and research purposes.
