"""HTML5 Canvas 2D Match Replay Presentation Component.

Renders interactive 60 FPS Canvas playback with scrub slider, dual-bot telemetry HUD,
trajectory trails, sensor ray visualizations, and playback speed controls.
"""

from __future__ import annotations

import json

import pandas as pd

from config.config import RobotClassConfig


def render_html5_canvas_replay(
    df_m: pd.DataFrame,
    cfg_obj: RobotClassConfig,
    m_id: str,
) -> str:
    opp_cols = [c for c in df_m.columns if c.startswith("Opp_")]
    edge_cols = [c for c in df_m.columns if c.startswith("IR_Edge_")]

    ticks_data = []
    for t_idx, df_t in df_m.groupby("Tick"):
        status_val = "IN_PROGRESS"
        bots_info = {}
        # Deterministic sorting so Bot_A is always first, Bot_B always second
        for _, row in df_t.sort_values("Bot_ID").iterrows():
            bid = str(row["Bot_ID"])
            status_val = str(row.get("Match_Status", status_val))

            # Extract sensor maps
            opp_s = {c: round(float(row[c]), 1) if pd.notna(row.get(c)) else -1.0 for c in opp_cols}
            edge_s = {c: round(float(row[c]), 3) if pd.notna(row.get(c)) else 0.0 for c in edge_cols}
            pwm_l = int(row["Action_PWM_Left"]) if "Action_PWM_Left" in row and pd.notna(row["Action_PWM_Left"]) else 0
            pwm_r = int(row["Action_PWM_Right"]) if "Action_PWM_Right" in row and pd.notna(row["Action_PWM_Right"]) else 0
            dist_c = round(float(row["Dist_To_Center"]), 1) if "Dist_To_Center" in row and pd.notna(row["Dist_To_Center"]) else 0.0

            bots_info[bid] = {
                "x": round(float(row["Pos_X"]), 2),
                "y": round(float(row["Pos_Y"]), 2),
                "h": round(float(row["Heading_Deg"]), 2),
                "state": str(row.get("Current_State", "SEARCH")),
                "pwm_l": pwm_l,
                "pwm_r": pwm_r,
                "dist_c": dist_c,
                "edge": edge_s,
                "opp": opp_s,
            }
        ticks_data.append({
            "tick": int(t_idx),
            "time_s": round(int(t_idx) * cfg_obj.dt, 2),
            "status": status_val,
            "bots": bots_info,
        })
    df_bot_a = df_m[df_m["Bot_ID"] == "Bot_A"]
    df_bot_b = df_m[df_m["Bot_ID"] == "Bot_B"]
    strat_a = "N/A"
    form_a = "N/A"
    if not df_bot_a.empty:
        r0_a = df_bot_a.iloc[0]
        strat_a = str(r0_a.get("Strategy_Profile", r0_a.get("Strategy", "N/A")))
        form_a = str(r0_a.get("Starting_Formation", "HEAD_ON"))
    strat_b = "N/A"
    form_b = "N/A"
    if not df_bot_b.empty:
        r0_b = df_bot_b.iloc[0]
        strat_b = str(r0_b.get("Strategy_Profile", r0_b.get("Strategy", "N/A")))
        form_b = str(r0_b.get("Starting_Formation", "HEAD_ON"))
    cfg_data = {
        "dohyo_r": cfg_obj.dohyo_radius_cm,
        "inner_r": cfg_obj.inner_ring_radius_cm,
        "center_r": cfg_obj.center_zone_radius_cm,
        "shikiri_sep": cfg_obj.shikiri_separation_cm,
        "shikiri_len": cfg_obj.shikiri_length_cm,
        "robot_l": cfg_obj.robot_length_cm,
        "robot_w": cfg_obj.robot_width_cm,
        "class_name": cfg_obj.class_name,
        "dt": cfg_obj.dt,
        "dt_ms": cfg_obj.dt * 1000.0,
    }

    ticks_json = json.dumps(ticks_data)
    cfg_json = json.dumps(cfg_data)

    return f"""
    <!DOCTYPE html>
    <html>
    <head>
      <meta charset="utf-8">
      <style>
        * {{ box-sizing: border-box; margin: 0; padding: 0; }}
        body {{ background: transparent; color: #f8fafc; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; display: flex; justify-content: center; align-items: center; padding: 4px 0; }}
        .player-card {{ background: #0b1120; border: 1px solid #1e293b; border-radius: 12px; padding: 16px; width: 100%; max-width: 1060px; box-shadow: 0 10px 35px rgba(0,0,0,0.5); }}
        .hud-top {{ display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 10px; margin-bottom: 12px; background: #0f172a; padding: 10px 14px; border-radius: 8px; border: 1px solid #1e293b; }}
        .badge {{ display: inline-flex; align-items: center; gap: 6px; padding: 5px 10px; border-radius: 6px; font-size: 13px; font-weight: 600; }}
        .badge-timer {{ background: #1e293b; border: 1px solid #334155; }}
        .badge-status {{ background: #0284c7; color: #fff; }}
        .badge-bota {{ background: rgba(0, 210, 255, 0.12); border: 1px solid #00d2ff; color: #38bdf8; }}
        .badge-botb {{ background: rgba(244, 63, 94, 0.12); border: 1px solid #f43f5e; color: #fb7185; }}
        .canvas-telemetry-split {{ display: flex; gap: 14px; align-items: stretch; justify-content: space-between; margin-bottom: 12px; }}
        .canvas-container {{ display: flex; justify-content: center; align-items: center; background: #060913; border-radius: 8px; padding: 8px; border: 1px solid #1e293b; width: 480px; height: 480px; flex-shrink: 0; }}
        canvas {{ display: block; border-radius: 6px; max-width: 100%; height: auto; }}
        .telemetry-inspector-box {{ flex: 1; min-width: 440px; display: flex; flex-direction: column; gap: 12px; justify-content: space-between; }}
        .bot-panel {{ background: #0f172a; border: 1px solid #1e293b; border-radius: 8px; padding: 10px 12px; display: flex; flex-direction: column; gap: 7px; box-shadow: 0 4px 12px rgba(0,0,0,0.25); }}
        .bota-panel {{ border-top: 3px solid #00d2ff; }}
        .botb-panel {{ border-top: 3px solid #f43f5e; }}
        .bot-header-row {{ display: flex; justify-content: space-between; align-items: flex-start; }}
        .bot-title {{ font-size: 13.5px; font-weight: 700; display: flex; align-items: center; gap: 6px; }}
        .state-badge {{ font-size: 11px; font-weight: 700; padding: 2px 8px; border-radius: 4px; letter-spacing: 0.5px; }}
        .decision-banner {{ background: #1e293b; border-left: 3px solid #38bdf8; border-radius: 4px; padding: 5px 8px; font-size: 11px; color: #cbd5e1; line-height: 1.35; }}
        .botb-panel .decision-banner {{ border-left-color: #fb7185; }}
        .metrics-2x2-grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 5px; }}
        .metric-tile {{ background: #0b1120; border: 1px solid #1e293b; border-radius: 5px; padding: 4px 8px; display: flex; justify-content: space-between; align-items: center; height: 26px; }}
        .metric-tile-label {{ font-size: 10.5px; color: #94a3b8; }}
        .metric-tile-val {{ font-size: 11px; font-weight: 700; font-family: monospace; }}
        .meta-pill {{ display: inline-flex; align-items: center; font-size: 10px; font-weight: 600; padding: 2px 7px; border-radius: 4px; letter-spacing: 0.3px; }}
        .meta-pill-strat-a {{ background: rgba(14, 165, 233, 0.2); color: #38bdf8; border: 1px solid rgba(56, 189, 248, 0.45); }}
        .meta-pill-strat-b {{ background: rgba(244, 63, 94, 0.2); color: #fb7185; border: 1px solid rgba(251, 113, 133, 0.45); }}
        .meta-pill-form {{ background: rgba(100, 116, 139, 0.2); color: #cbd5e1; border: 1px solid rgba(148, 163, 184, 0.35); }}
        .sensor-array-row {{ display: flex; flex-direction: column; gap: 4px; margin-top: 1px; }}
        .sensor-edge-pill {{ font-size: 10px; font-weight: 700; padding: 2px 6px; border-radius: 3px; font-family: monospace; }}
        .sensor-edge-safe {{ background: rgba(34, 197, 94, 0.2); color: #4ade80; border: 1px solid #22c55e; }}
        .sensor-edge-danger, .sensor-edge-tripped {{ background: rgba(239, 68, 68, 0.35); color: #f87171; border: 1px solid #ef4444; font-weight: 800; }}
        .sensor-pills-grid {{ display: flex; gap: 4px; width: 100%; }}
        .sensor-tile {{ flex: 1; min-width: 0; background: #0b1120; border: 1px solid #1e293b; border-radius: 4px; padding: 3px 1px; text-align: center; display: flex; flex-direction: column; align-items: center; justify-content: center; height: 32px; box-sizing: border-box; }}
        .sensor-tile.hit {{ background: rgba(245, 158, 11, 0.18); border-color: #f59e0b; }}
        .sensor-tile.clear {{ opacity: 0.55; }}
        .sensor-tile-label {{ font-size: 8.5px; color: #94a3b8; font-weight: 600; line-height: 1; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }}
        .sensor-tile-val {{ font-size: 9.5px; font-weight: 700; color: #f59e0b; font-family: monospace; line-height: 1.2; margin-top: 2px; }}
        .controls-bar {{ background: #0f172a; border: 1px solid #1e293b; border-radius: 8px; padding: 10px 14px; display: flex; flex-direction: column; gap: 8px; }}
        .slider-row {{ display: flex; align-items: center; gap: 12px; width: 100%; }}
        input[type="range"] {{ flex: 1; height: 6px; accent-color: #00d2ff; cursor: pointer; }}
        .btn-toolbar {{ display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 8px; }}
        .btn-group {{ display: flex; gap: 6px; align-items: center; }}
        button {{ background: #1e293b; color: #f8fafc; border: 1px solid #334155; border-radius: 6px; padding: 6px 12px; font-size: 12px; font-weight: 600; cursor: pointer; transition: all 0.15s ease; }}
        button:hover {{ background: #334155; border-color: #00d2ff; color: #00d2ff; }}
        button.btn-primary {{ background: #0284c7; border-color: #00d2ff; color: #ffffff; }}
        button.btn-primary:hover {{ background: #0369a1; }}
        .spd-btn {{ padding: 4px 8px; font-size: 11px; }}
        .spd-btn.active {{ background: #0284c7; border-color: #00d2ff; color: #ffffff; }}
      </style>
    </head>
    <body>
      <div class="player-card">
        <div class="hud-top">
          <div style="display: flex; gap: 8px; align-items: center;">
            <div class="badge badge-timer">
              <span id="timeVal" style="font-family: monospace; font-size:14px; font-weight:700;">0.00 s</span>
              <span style="color:#64748b; font-size:12px;">(Tick <span id="tickVal">0</span> / <span id="maxTickVal">0</span>)</span>
            </div>
            <div class="badge badge-status" id="statusBadge">IN_PROGRESS</div>
          </div>
          <div style="display: flex; gap: 8px; align-items: center;">
            <div class="badge badge-bota">
              <span style="display:inline-block; width:10px; height:10px; background:#00d2ff; border-radius:2px;"></span>
              Bot A: <span id="stateAVal" style="font-weight:700; margin-left:3px;">SEARCH</span>
            </div>
            <div class="badge badge-botb">
              <span style="display:inline-block; width:10px; height:10px; background:#f43f5e; border-radius:2px;"></span>
              Bot B: <span id="stateBVal" style="font-weight:700; margin-left:3px;">SEARCH</span>
            </div>
          </div>
        </div>

        <div class="canvas-telemetry-split">
          <div class="canvas-container">
            <canvas id="replayCanvas" width="460" height="460"></canvas>
          </div>

          <div class="telemetry-inspector-box">
            <div class="bot-panel bota-panel">
              <div class="bot-header-row">
                <div>
                  <div class="bot-title" style="color: #38bdf8;">
                    <span style="display:inline-block; width:8px; height:8px; background:#00d2ff; border-radius:50%;"></span>
                    Bot A (Candidate Strategy)
                  </div>
                  <div style="display: flex; gap: 6px; align-items: center; margin-top: 4px; flex-wrap: wrap;">
                    <span class="meta-pill meta-pill-strat-a">Strategy: {strat_a}</span>
                    <span class="meta-pill meta-pill-form">Formation: {form_a}</span>
                  </div>
                </div>
                <span class="state-badge" id="botAStateBadge" style="background: #334155; color: #cbd5e1;">SEARCH</span>
              </div>
              
              <div class="decision-banner" id="botADecisionBanner">
                <span id="botADecisionText">Sweeping arena in search pattern</span>
              </div>

              <div class="metrics-2x2-grid">
                <div class="metric-tile">
                  <span class="metric-tile-label">Motors (PWM):</span>
                  <span class="metric-tile-val" id="botAPwmVal" style="color:#38bdf8;">L: 0 | R: 0</span>
                </div>
                <div class="metric-tile">
                  <span class="metric-tile-label">Distance to Center:</span>
                  <span class="metric-tile-val" id="botADistVal" style="color:#38bdf8;">r = 0.0 cm</span>
                </div>
                <div class="metric-tile">
                  <span class="metric-tile-label">Position (X, Y):</span>
                  <span class="metric-tile-val" id="botAPosVal" style="color:#f8fafc;">(0.0, 0.0)</span>
                </div>
                <div class="metric-tile">
                  <span class="metric-tile-label">Heading Angle:</span>
                  <span class="metric-tile-val" id="botAHeadingVal" style="color:#f8fafc;">0°</span>
                </div>
              </div>

              <div class="sensor-array-row">
                <div style="display: flex; justify-content: space-between; align-items: center; font-size: 10.5px; color: #94a3b8;">
                  <span>Edge Reflectance (QRE1113):</span>
                  <div style="display:flex; gap:4px;">
                    <span class="sensor-edge-pill sensor-edge-safe" id="botAEdgeFL">FL: 0.00</span>
                    <span class="sensor-edge-pill sensor-edge-safe" id="botAEdgeFR">FR: 0.00</span>
                  </div>
                </div>
                <div style="font-size: 10.5px; color: #94a3b8; margin-top: 2px;">Opponent Optical Array (E3Z-D62):</div>
                <div class="sensor-pills-grid" id="botAOppSensors">
                  <div class="sensor-tile clear"><span class="sensor-tile-label">F0°</span><span class="sensor-tile-val">--</span></div>
                  <div class="sensor-tile clear"><span class="sensor-tile-label">L18°</span><span class="sensor-tile-val">--</span></div>
                  <div class="sensor-tile clear"><span class="sensor-tile-label">R18°</span><span class="sensor-tile-val">--</span></div>
                  <div class="sensor-tile clear"><span class="sensor-tile-label">L90°</span><span class="sensor-tile-val">--</span></div>
                  <div class="sensor-tile clear"><span class="sensor-tile-label">R90°</span><span class="sensor-tile-val">--</span></div>
                </div>
              </div>
            </div>

            <div class="bot-panel botb-panel">
              <div class="bot-header-row">
                <div>
                  <div class="bot-title" style="color: #fb7185;">
                    <span style="display:inline-block; width:8px; height:8px; background:#f43f5e; border-radius:50%;"></span>
                    Bot B (Adversary Strategy)
                  </div>
                  <div style="display: flex; gap: 6px; align-items: center; margin-top: 4px; flex-wrap: wrap;">
                    <span class="meta-pill meta-pill-strat-b">Strategy: {strat_b}</span>
                    <span class="meta-pill meta-pill-form">Formation: {form_b}</span>
                  </div>
                </div>
                <span class="state-badge" id="botBStateBadge" style="background: #334155; color: #cbd5e1;">SEARCH</span>
              </div>
              
              <div class="decision-banner" id="botBDecisionBanner">
                <span id="botBDecisionText">Sweeping arena in search pattern</span>
              </div>

              <div class="metrics-2x2-grid">
                <div class="metric-tile">
                  <span class="metric-tile-label">Motors (PWM):</span>
                  <span class="metric-tile-val" id="botBPwmVal" style="color:#fb7185;">L: 0 | R: 0</span>
                </div>
                <div class="metric-tile">
                  <span class="metric-tile-label">Distance to Center:</span>
                  <span class="metric-tile-val" id="botBDistVal" style="color:#fb7185;">r = 0.0 cm</span>
                </div>
                <div class="metric-tile">
                  <span class="metric-tile-label">Position (X, Y):</span>
                  <span class="metric-tile-val" id="botBPosVal" style="color:#f8fafc;">(0.0, 0.0)</span>
                </div>
                <div class="metric-tile">
                  <span class="metric-tile-label">Heading Angle:</span>
                  <span class="metric-tile-val" id="botBHeadingVal" style="color:#f8fafc;">0°</span>
                </div>
              </div>

              <div class="sensor-array-row">
                <div style="display: flex; justify-content: space-between; align-items: center; font-size: 10.5px; color: #94a3b8;">
                  <span>Edge Reflectance (QRE1113):</span>
                  <div style="display:flex; gap:4px;">
                    <span class="sensor-edge-pill sensor-edge-safe" id="botBEdgeFL">FL: 0.00</span>
                    <span class="sensor-edge-pill sensor-edge-safe" id="botBEdgeFR">FR: 0.00</span>
                  </div>
                </div>
                <div style="font-size: 10.5px; color: #94a3b8; margin-top: 2px;">Opponent Optical Array (E3Z-D62):</div>
                <div class="sensor-pills-grid" id="botBOppSensors">
                  <div class="sensor-tile clear"><span class="sensor-tile-label">F0°</span><span class="sensor-tile-val">--</span></div>
                  <div class="sensor-tile clear"><span class="sensor-tile-label">L18°</span><span class="sensor-tile-val">--</span></div>
                  <div class="sensor-tile clear"><span class="sensor-tile-label">R18°</span><span class="sensor-tile-val">--</span></div>
                  <div class="sensor-tile clear"><span class="sensor-tile-label">L90°</span><span class="sensor-tile-val">--</span></div>
                  <div class="sensor-tile clear"><span class="sensor-tile-label">R90°</span><span class="sensor-tile-val">--</span></div>
                </div>
              </div>
            </div>
          </div>
        </div>

        <!-- Interactive Controls Bar -->
        <div class="controls-bar">
          <div class="slider-row">
            <span style="font-size:12px; color:#94a3b8; width:45px;">0.00s</span>
            <input type="range" id="frameSlider" min="0" max="100" value="0" oninput="onSliderDrag(this.value)">
            <span id="endTimeVal" style="font-size:12px; color:#94a3b8; width:45px; text-align:right;">0.00s</span>
          </div>
          <div class="btn-toolbar">
            <div class="btn-group">
              <button id="btnPlayPause" class="btn-primary" onclick="togglePlay()">Play</button>
              <button onclick="resetReplay()">Reset</button>
              <button onclick="stepTick(-1)">-1</button>
              <button onclick="stepTick(1)">+1</button>
              <button onclick="stepTick(-10)">-10</button>
              <button onclick="stepTick(10)">+10</button>
            </div>
            <div style="display: flex; align-items: center; gap: 6px;">
              <span style="font-size:12px; color:#94a3b8; margin-right:2px;">Speed:</span>
              <button class="spd-btn" onclick="setSpeed(0.5, this)">0.5x</button>
              <button class="spd-btn active" onclick="setSpeed(1.0, this)">1.0x</button>
              <button class="spd-btn" onclick="setSpeed(2.0, this)">2.0x</button>
              <button class="spd-btn" onclick="setSpeed(4.0, this)">4.0x</button>
            </div>
          </div>
        </div>
      </div>

      <script>
        const frames = {ticks_json};
        const cfg = {cfg_json};
        const totalFrames = frames.length;
        const maxTickIdx = totalFrames > 0 ? totalFrames - 1 : 0;

        let currentIdx = 0;
        let isPlaying = false;
        let playbackSpeed = 1.0;
        let lastFrameTimestamp = 0;
        let animRequestId = null;

        const canvas = document.getElementById("replayCanvas");
        const ctx = canvas.getContext("2d");
        const slider = document.getElementById("frameSlider");
        const timeVal = document.getElementById("timeVal");
        const tickVal = document.getElementById("tickVal");
        const maxTickVal = document.getElementById("maxTickVal");
        const endTimeVal = document.getElementById("endTimeVal");
        const statusBadge = document.getElementById("statusBadge");
        const stateAVal = document.getElementById("stateAVal");
        const stateBVal = document.getElementById("stateBVal");
        const btnPlay = document.getElementById("btnPlayPause");

        slider.max = maxTickIdx;
        maxTickVal.innerText = frames[maxTickIdx] ? frames[maxTickIdx].tick : 0;
        endTimeVal.innerText = frames[maxTickIdx] ? (frames[maxTickIdx].tick * cfg.dt).toFixed(2) + "s" : "0.00s";

        const cx = 230;
        const cy = 230;
        const limitSpan = cfg.dohyo_r + 5.0;
        const scale = 215.0 / limitSpan;

        function updateBotTelemetry(bData, prefix) {{
          if (!bData) return;

          // 1. State badge
          const stateEl = document.getElementById(prefix + "StateBadge");
          if (stateEl) {{
            stateEl.innerText = bData.state;
            if (bData.state === "ATTACK") {{
              stateEl.style.background = "#ef4444";
              stateEl.style.color = "#ffffff";
            }} else if (bData.state === "EDGE_RECOVERY") {{
              stateEl.style.background = "#f59e0b";
              stateEl.style.color = "#000000";
            }} else if (bData.state === "TRACK") {{
              stateEl.style.background = "#38bdf8";
              stateEl.style.color = "#0f172a";
            }} else if (bData.state === "EVADE") {{
              stateEl.style.background = "#a855f7";
              stateEl.style.color = "#ffffff";
            }} else {{
              stateEl.style.background = "#334155";
              stateEl.style.color = "#cbd5e1";
            }}
          }}

          // 2. Decision banner (Zero emojis)
          const decText = document.getElementById(prefix + "DecisionText");
          if (decText) {{
            let msg = "Sweeping arena in search pattern";
            if (bData.state === "ATTACK") msg = "Target Locked: High-Power Bull-Rush Charge";
            else if (bData.state === "EDGE_RECOVERY") msg = "Dohyo Edge Triggered: Reversing & Inward Pivot";
            else if (bData.state === "TRACK") msg = "Flank Detected: Fast Snap Turn to Opponent Bearing";
            else if (bData.state === "EVADE") msg = "Tactical Matador Evasion: Flank Counter-Slip";
            decText.innerText = msg;
          }}

          // 3. PWM Motors
          const pwmEl = document.getElementById(prefix + "PwmVal");
          if (pwmEl) {{
            const pL = (bData.pwm_l > 0 ? "+" : "") + bData.pwm_l;
            const pR = (bData.pwm_r > 0 ? "+" : "") + bData.pwm_r;
            pwmEl.innerText = `L: ${{pL}} | R: ${{pR}}`;
          }}

          // 4. Distance to Center, Position, Heading
          const distEl = document.getElementById(prefix + "DistVal");
          if (distEl) {{
            distEl.innerText = `r = ${{bData.dist_c.toFixed(1)}} cm`;
          }}
          const posEl = document.getElementById(prefix + "PosVal");
          if (posEl) {{
            posEl.innerText = `(${{bData.x.toFixed(1)}}, ${{bData.y.toFixed(1)}})`;
          }}
          const headingEl = document.getElementById(prefix + "HeadingVal");
          if (headingEl) {{
            headingEl.innerText = `${{bData.h.toFixed(0)}}°`;
          }}

          // 5. Line Reflectance Sensors
          if (bData.edge) {{
            const fl = bData.edge.IR_Edge_FL !== undefined ? bData.edge.IR_Edge_FL : 0.0;
            const fr = bData.edge.IR_Edge_FR !== undefined ? bData.edge.IR_Edge_FR : 0.0;
            const eflEl = document.getElementById(prefix + "EdgeFL");
            const efrEl = document.getElementById(prefix + "EdgeFR");
            if (eflEl) {{
              eflEl.innerText = `FL: ${{fl.toFixed(2)}}`;
              eflEl.className = "sensor-edge-pill " + (fl >= 0.70 ? "sensor-edge-danger" : "sensor-edge-safe");
            }}
            if (efrEl) {{
              efrEl.innerText = `FR: ${{fr.toFixed(2)}}`;
              efrEl.className = "sensor-edge-pill " + (fr >= 0.70 ? "sensor-edge-danger" : "sensor-edge-safe");
            }}
          }}

          // 6. Optical Opponent Sensors (Fixed Static Tiles)
          const oppEl = document.getElementById(prefix + "OppSensors");
          if (oppEl && bData.opp) {{
            let html = "";
            for (const [k, v] of Object.entries(bData.opp)) {{
              let shortK = k.replace("Opp_", "").replace("_", ".");
              if (!shortK.endsWith("°")) shortK += "°";
              const isHit = v > 0.0;
              const tileClass = isHit ? "sensor-tile locked" : "sensor-tile clear";
              const valText = isHit ? `${{v.toFixed(1)}}cm` : "--";
              html += `<div class="${{tileClass}}"><span class="sensor-tile-label">${{shortK}}</span><span class="sensor-tile-val">${{valText}}</span></div>`;
            }}
            oppEl.innerHTML = html;
          }}
        }}

        function drawFrame(frameIdx) {{
          if (frameIdx < 0 || frameIdx >= totalFrames) return;
          const frame = frames[frameIdx];

          ctx.clearRect(0, 0, 460, 460);

          // 1. Dohyo Outer Ring (White Border)
          ctx.beginPath();
          ctx.arc(cx, cy, cfg.dohyo_r * scale, 0, Math.PI * 2);
          ctx.fillStyle = "#cbd5e1";
          ctx.fill();
          ctx.lineWidth = 3;
          ctx.strokeStyle = "#475569";
          ctx.stroke();

          // 2. Dohyo Inner Ring (Black Fighting Surface)
          ctx.beginPath();
          ctx.arc(cx, cy, cfg.inner_r * scale, 0, Math.PI * 2);
          ctx.fillStyle = "#0f172a";
          ctx.fill();
          ctx.lineWidth = 1.5;
          ctx.strokeStyle = "#334155";
          ctx.stroke();

          // 3. Center Zone (Dotted Cyan Ring)
          ctx.beginPath();
          ctx.setLineDash([4, 4]);
          ctx.arc(cx, cy, cfg.center_r * scale, 0, Math.PI * 2);
          ctx.lineWidth = 1.5;
          ctx.strokeStyle = "#0ea5e9";
          ctx.stroke();
          ctx.setLineDash([]);

          // 4. Shikiri Start Lines
          const sSep = (cfg.shikiri_sep / 2.0) * scale;
          const sLen = (cfg.shikiri_len / 2.0) * scale;
          ctx.lineWidth = 3.5;
          ctx.strokeStyle = "#d97706";
          ctx.beginPath();
          ctx.moveTo(cx - sSep, cy - sLen);
          ctx.lineTo(cx - sSep, cy + sLen);
          ctx.stroke();
          ctx.beginPath();
          ctx.moveTo(cx + sSep, cy - sLen);
          ctx.lineTo(cx + sSep, cy + sLen);
          ctx.stroke();

          // 5. Draw Sumo Robots (Bot A Cyan, Bot B Crimson)
          const botStyles = {{
            "Bot_A": {{ fill: "rgba(0, 210, 255, 0.45)", stroke: "#00d2ff", label: "A" }},
            "Bot_B": {{ fill: "rgba(244, 63, 94, 0.45)", stroke: "#f43f5e", label: "B" }}
          }};

          const rL = cfg.robot_l * scale;
          const rW = cfg.robot_w * scale;
          const halfL = rL / 2.0;
          const halfW = rW / 2.0;

          ["Bot_A", "Bot_B"].forEach(bid => {{
            if (!frame.bots[bid]) return;
            const b = frame.bots[bid];
            const px = cx + b.x * scale;
            const py = cy - b.y * scale; // Canvas Y is inverted relative to Cartesian
            const headingRad = (b.h * Math.PI) / 180.0;
            const style = botStyles[bid];

            ctx.save();
            ctx.translate(px, py);
            ctx.rotate(-headingRad);

            // Body Rectangle
            ctx.fillStyle = style.fill;
            ctx.fillRect(-halfL, -halfW, rL, rW);
            ctx.lineWidth = 2.5;
            ctx.strokeStyle = style.stroke;
            ctx.strokeRect(-halfL, -halfW, rL, rW);

            // Front Bumper (Bright Green)
            ctx.beginPath();
            ctx.lineWidth = 4.5;
            ctx.strokeStyle = "#22c55e";
            ctx.moveTo(halfL, -halfW);
            ctx.lineTo(halfL, halfW);
            ctx.stroke();

            // Rear Bumper (Dark Red)
            ctx.beginPath();
            ctx.lineWidth = 3.0;
            ctx.strokeStyle = "#7f1d1d";
            ctx.moveTo(-halfL, -halfW);
            ctx.lineTo(-halfL, halfW);
            ctx.stroke();

            // Center Label
            ctx.fillStyle = "#ffffff";
            ctx.font = "bold 9.5px sans-serif";
            ctx.textAlign = "center";
            ctx.textBaseline = "middle";
            ctx.fillText(style.label, 0, 0);

            // Heading Vector Arrow (Contained Strictly Inside Body)
            const arrowStart = halfL * 0.12;
            const arrowEnd = halfL * 0.76;
            ctx.beginPath();
            ctx.strokeStyle = "#eab308";
            ctx.lineWidth = 2.0;
            ctx.moveTo(arrowStart, 0);
            ctx.lineTo(arrowEnd, 0);
            ctx.stroke();

            // Arrow Head (Strictly Inside Body, Does Not Protrude)
            ctx.beginPath();
            ctx.fillStyle = "#eab308";
            ctx.moveTo(arrowEnd + 3.5, 0);
            ctx.lineTo(arrowEnd - 3.5, -3.0);
            ctx.lineTo(arrowEnd - 3.5, 3.0);
            ctx.closePath();
            ctx.fill();

            ctx.restore();
          }});

          // 6. Update HUD Elements
          timeVal.innerText = (frame.tick * cfg.dt).toFixed(2) + " s";
          tickVal.innerText = frame.tick;
          slider.value = frameIdx;
          statusBadge.innerText = frame.status || "IN_PROGRESS";
          if (frame.status === "BOT_A_WIN") {{
            statusBadge.style.background = "#16a34a";
          }} else if (frame.status === "BOT_B_WIN") {{
            statusBadge.style.background = "#dc2626";
          }} else {{
            statusBadge.style.background = "#0284c7";
          }}

          if (frame.bots["Bot_A"]) {{
            stateAVal.innerText = frame.bots["Bot_A"].state;
            updateBotTelemetry(frame.bots["Bot_A"], "botA");
          }}
          if (frame.bots["Bot_B"]) {{
            stateBVal.innerText = frame.bots["Bot_B"].state;
            updateBotTelemetry(frame.bots["Bot_B"], "botB");
          }}
        }}

        function animationLoop(timestamp) {{
          if (!isPlaying) return;
          if (!lastFrameTimestamp) lastFrameTimestamp = timestamp;
          const elapsed = timestamp - lastFrameTimestamp;
          const frameDurationMs = cfg.dt_ms / playbackSpeed;

          if (elapsed >= frameDurationMs) {{
            const steps = Math.max(1, Math.floor(elapsed / frameDurationMs));
            currentIdx += steps;
            lastFrameTimestamp = timestamp - (elapsed % frameDurationMs);

            if (currentIdx >= maxTickIdx) {{
              currentIdx = maxTickIdx;
              drawFrame(currentIdx);
              pauseReplay();
              return;
            }}
            drawFrame(currentIdx);
          }}

          animRequestId = requestAnimationFrame(animationLoop);
        }}

        function togglePlay() {{
          if (isPlaying) {{
            pauseReplay();
          }} else {{
            if (currentIdx >= maxTickIdx) currentIdx = 0;
            isPlaying = true;
            btnPlay.innerText = "Pause";
            btnPlay.style.background = "#eab308";
            btnPlay.style.color = "#0f172a";
            lastFrameTimestamp = 0;
            animRequestId = requestAnimationFrame(animationLoop);
          }}
        }}

        function pauseReplay() {{
          isPlaying = false;
          btnPlay.innerText = "Play";
          btnPlay.style.background = "#00d2ff";
          btnPlay.style.color = "#0f172a";
          if (animRequestId) cancelAnimationFrame(animRequestId);
        }}

        function resetReplay() {{
          pauseReplay();
          currentIdx = 0;
          drawFrame(0);
        }}

        function stepTick(delta) {{
          pauseReplay();
          currentIdx = Math.max(0, Math.min(maxTickIdx, currentIdx + delta));
          drawFrame(currentIdx);
        }}

        function onSliderDrag(val) {{
          pauseReplay();
          currentIdx = parseInt(val, 10);
          drawFrame(currentIdx);
        }}

        function setSpeed(spd, btn) {{
          playbackSpeed = spd;
          document.querySelectorAll(".spd-btn").forEach(b => b.classList.remove("active"));
          btn.classList.add("active");
        }}

        // Render first frame immediately
        drawFrame(0);
      </script>
    </body>
    </html>
    """

# Render 60 FPS HTML5 Canvas Player with Live Telemetry Inspector
