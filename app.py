"""
EcoMesh: Autonomous Microgrid Energy Management System
Real-time Operator Console & Telemetry Dashboard powered by Groq and LangChain.
"""

import os
import json
from datetime import datetime
from dotenv import load_dotenv
import streamlit as st
import plotly.graph_objects as go

from engine.microgrid import MicrogridEngine, MicrogridTelemetry
from agents.dispatch_agent import DispatchAgent, get_available_groq_models

# Initialize environment
load_dotenv()

# Streamlit Page Configuration
st.set_page_config(
    page_title="EcoMesh - Microgrid Management System",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for high-tech clean operational look
st.markdown("""
<style>
    .main-title {
        font-size: 2.2rem;
        font-weight: 700;
        letter-spacing: -0.5px;
        margin-bottom: 0px;
    }
    .sub-title {
        color: #718096;
        font-size: 0.95rem;
        margin-top: 2px;
        margin-bottom: 18px;
    }
    .metric-card {
        background-color: #f8fafc;
        border-radius: 10px;
        padding: 12px 18px;
        border-left: 4px solid #3b82f6;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }
    .stMetric {
        background: rgba(240, 244, 248, 0.4);
        padding: 12px;
        border-radius: 8px;
        border: 1px solid rgba(226, 232, 240, 0.8);
    }
    .agent-status-badge {
        padding: 4px 10px;
        border-radius: 12px;
        font-size: 0.8rem;
        font-weight: 600;
        display: inline-block;
    }
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------
# State Initialization
# ---------------------------------------------------------

if "engine" not in st.session_state:
    st.session_state.engine = MicrogridEngine(seed=42)

# Resolve Groq key from environment or Streamlit Cloud Secrets
secret_key = ""
try:
    secret_key = st.secrets.get("GROQ_API_KEY", "")
except Exception:
    pass
default_groq_key = os.getenv("GROQ_API_KEY", "") or secret_key

if "agent" not in st.session_state:
    st.session_state.agent = DispatchAgent(st.session_state.engine, api_key=default_groq_key)

if "agent_logs" not in st.session_state:
    st.session_state.agent_logs = []

# Pre-populate one initial step if history is empty
if len(st.session_state.engine.history) == 0:
    first_step = st.session_state.agent.step()
    st.session_state.agent_logs.append(first_step)

engine: MicrogridEngine = st.session_state.engine
agent: DispatchAgent = st.session_state.agent


# ---------------------------------------------------------
# Sidebar Controls
# ---------------------------------------------------------

with st.sidebar:
    st.header("⚡ System Control")
    
    # Groq API Configuration
    st.subheader("Groq Intelligence Gateway")
    current_key = default_groq_key
    api_key_input = st.text_input(
        "Groq API Key",
        value=agent.api_key or current_key,
        type="password",
        help="Enter your Groq API key to power autonomous sub-second reasoning and tool calling."
    )
    if api_key_input != agent.api_key:
        agent.update_api_key(api_key_input)
        st.rerun()

    # Dynamic model selection
    if agent.api_key:
        available_models = get_available_groq_models(agent.api_key)
        if available_models:
            current_idx = available_models.index(agent.model_name) if agent.model_name in available_models else 0
            selected_model = st.selectbox(
                "Foundation Reasoning Model",
                options=available_models,
                index=current_idx,
                help="Select from available reasoning and dispatch models on your Groq account."
            )
            if selected_model != agent.model_name:
                agent.set_model(selected_model)
                st.rerun()

    if agent.has_groq:
        st.success(f"🟢 Groq Active: `{agent.model_name}`", icon="✅")
    else:
        st.warning("🟡 Standby Mode: Deterministic Engine", icon="⚠️")
        st.caption("Add GROQ_API_KEY above or in `.env` to activate autonomous LLM tool calling.")

    st.divider()

    # Simulation Stepping
    st.subheader("Simulation Execution")
    col_step1, col_step2 = st.columns(2)
    
    with col_step1:
        if st.button("▶ Step (1h)", use_container_width=True, type="primary"):
            step_result = agent.step()
            st.session_state.agent_logs.append(step_result)
            st.rerun()

    with col_step2:
        if st.button("⏩ Run 12h", use_container_width=True):
            for _ in range(12):
                res = agent.step()
                st.session_state.agent_logs.append(res)
            st.rerun()

    if st.button("🔄 Reset Simulation", use_container_width=True):
        engine.reset()
        st.session_state.agent_logs.clear()
        initial_res = agent.step()
        st.session_state.agent_logs.append(initial_res)
        st.rerun()

    st.divider()

    # Weather & Environmental Overrides
    st.subheader("Environmental Stressors")
    cloud_cover = st.slider(
        "Cloud Cover (Solar Attenuation)",
        min_value=0,
        max_value=100,
        value=int(engine.cloud_cover * 100),
        step=5,
        format="%d%%"
    )
    engine.cloud_cover = cloud_cover / 100.0

    wind_drop = st.slider(
        "Wind Calms (Wind Attenuation)",
        min_value=0,
        max_value=100,
        value=int(engine.wind_drop * 100),
        step=5,
        format="%d%%"
    )
    engine.wind_drop = wind_drop / 100.0

    st.divider()

    # Manual Dispatch Override Console
    st.subheader("Manual Operator Override")
    manual_enabled = st.checkbox("Enable Manual BESS Override", value=False)
    
    if manual_enabled:
        override_kw = st.slider(
            "BESS Target Power (kW)",
            min_value=-25.0,
            max_value=25.0,
            value=0.0,
            step=1.0,
            help="+25 kW = Charge | -25 kW = Discharge"
        )
        override_curtail = st.slider(
            "Curtailment Override (%)",
            min_value=0,
            max_value=100,
            value=0,
            step=5
        ) / 100.0

        if st.button("⚡ Execute Manual Dispatch", use_container_width=True):
            telemetry = engine.dispatch(target_battery_kw=override_kw, curtailment_pct=override_curtail)
            manual_log = {
                "agent_mode": "Human Operator Override",
                "dispatched_battery_kw": override_kw,
                "curtailment_pct": override_curtail,
                "reasoning": f"Manual override executed by operator: Battery commanded to {override_kw:+.1f} kW; Curtailment set to {override_curtail*100:.0f}%.",
                "tool_calls": [{
                    "name": "manual_dispatch",
                    "arguments": {"target_kw": override_kw, "curtailment_pct": override_curtail},
                    "result": {"status": "manual_override_executed", "dispatched_kw": override_kw}
                }],
                "telemetry": telemetry,
                "state_before": engine.get_state(),
            }
            st.session_state.agent_logs.append(manual_log)
            st.rerun()


# ---------------------------------------------------------
# Main Application Content
# ---------------------------------------------------------

st.markdown('<h1 class="main-title">EcoMesh: Autonomous Microgrid Management</h1>', unsafe_allow_html=True)
st.markdown('<p class="sub-title">Autonomous Microgrid Balancing, Dynamic Battery Dispatch & Real-Time Telemetry powered by Groq and LangChain</p>', unsafe_allow_html=True)

# Fetch latest telemetry snapshot
latest: MicrogridTelemetry = engine.history[-1] if engine.history else engine.dispatch(0.0)
prev: MicrogridTelemetry = engine.history[-2] if len(engine.history) >= 2 else latest

# ---------------------------------------------------------
# 1. Top Metric Ribbon (Section 4.5)
# ---------------------------------------------------------
kpi1, kpi2, kpi3, kpi4 = st.columns(4)

with kpi1:
    gen_delta = round(latest.total_generation_kw - prev.total_generation_kw, 1)
    st.metric(
        label="Total Renewable Gen",
        value=f"{latest.total_generation_kw:.1f} kW",
        delta=f"{gen_delta:+.1f} kW vs prev",
        help=f"Solar: {latest.solar_kw:.1f} kW | Wind: {latest.wind_kw:.1f} kW"
    )

with kpi2:
    load_delta = round(latest.load_kw - prev.load_kw, 1)
    st.metric(
        label="Microgrid Demand",
        value=f"{latest.load_kw:.1f} kW",
        delta=f"{load_delta:+.1f} kW vs prev",
        delta_color="inverse",
        help="Dynamic consumer demand profile"
    )

with kpi3:
    soc_delta = round(latest.battery_soc_pct - prev.battery_soc_pct, 1)
    soc_status = "Optimal" if 10.0 <= latest.battery_soc_pct <= 90.0 else ("Depleted (<10%)" if latest.battery_soc_pct < 10.0 else "Saturated (>90%)")
    st.metric(
        label="Battery SoC",
        value=f"{latest.battery_soc_pct:.1f}%",
        delta=f"{soc_delta:+.1f}% ({soc_status})",
        help="BESS State of Charge. Safe operating bounds: 10% - 90%."
    )

with kpi4:
    if latest.grid_export_kw > 0:
        grid_val = f"+{latest.grid_export_kw:.1f} kW (Export)"
        grid_delta = f"Tariff: ${latest.tariff_export:.2f}/kWh"
        delta_color = "normal"
    elif latest.grid_import_kw > 0:
        grid_val = f"-{latest.grid_import_kw:.1f} kW (Import)"
        grid_delta = f"Tariff: ${latest.tariff_import:.2f}/kWh"
        delta_color = "inverse"
    else:
        grid_val = "0.0 kW (Balanced)"
        grid_delta = "Zero Net Grid Exchange"
        delta_color = "off"

    st.metric(
        label="Macrogrid Exchange",
        value=grid_val,
        delta=grid_delta,
        delta_color=delta_color,
        help="Residual power exchanged with the bulk utility grid."
    )

st.write("")

# ---------------------------------------------------------
# 2. Telemetry Panels (Section 4.5)
# ---------------------------------------------------------
tab_charts, tab_sankey, tab_history = st.tabs(["📊 Real-Time Telemetry & SoC", "🔄 Power Distribution (Sankey)", "📋 Telemetry Logs"])

with tab_charts:
    col_chart, col_gauge = st.columns([2.2, 1.0])

    with col_chart:
        st.subheader("Dynamic Generation vs. Load")
        
        # Build history data
        steps = [t.step for t in engine.history]
        time_labels = [f"Step {t.step} ({int(t.hour):02d}:00)" for t in engine.history]
        solars = [t.solar_kw for t in engine.history]
        winds = [t.wind_kw for t in engine.history]
        loads = [t.load_kw for t in engine.history]
        total_gens = [t.total_generation_kw for t in engine.history]
        imbalances = [t.net_imbalance_kw for t in engine.history]
        battery_acts = [t.battery_dispatch_kw for t in engine.history]

        fig_ts = go.Figure()
        
        # Solar line
        fig_ts.add_trace(go.Scatter(
            x=time_labels, y=solars, name="Solar (kW)",
            mode="lines+markers", line=dict(color="#f59e0b", width=2.5)
        ))
        # Wind line
        fig_ts.add_trace(go.Scatter(
            x=time_labels, y=winds, name="Wind (kW)",
            mode="lines+markers", line=dict(color="#06b6d4", width=2.5)
        ))
        # Total Gen line
        fig_ts.add_trace(go.Scatter(
            x=time_labels, y=total_gens, name="Total Renewables (kW)",
            mode="lines", line=dict(color="#10b981", width=3, dash="dot")
        ))
        # Microgrid Load line
        fig_ts.add_trace(go.Scatter(
            x=time_labels, y=loads, name="Consumer Demand (kW)",
            mode="lines+markers", line=dict(color="#ef4444", width=3)
        ))
        # Net Imbalance line
        fig_ts.add_trace(go.Scatter(
            x=time_labels, y=imbalances, name="Net Imbalance (kW)",
            mode="lines", line=dict(color="#8b5cf6", width=2, dash="dash")
        ))

        fig_ts.update_layout(
            margin=dict(l=20, r=20, t=20, b=20),
            height=340,
            hovermode="x unified",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            xaxis=dict(showgrid=True, gridcolor="#e2e8f0"),
            yaxis=dict(title="Power (kW)", showgrid=True, gridcolor="#e2e8f0"),
        )
        st.plotly_chart(fig_ts, use_container_width=True)

    with col_gauge:
        st.subheader("Battery SoC Gauge")
        
        # Color-coded safety zones per spec:
        # <10% danger (red), 10%-90% optimal (green), >90% saturated (orange)
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=latest.battery_soc_pct,
            number={"suffix": "%", "font": {"size": 32}},
            gauge={
                "axis": {"range": [0, 100], "tickwidth": 1, "tickcolor": "#475569"},
                "bar": {"color": "#1e293b", "thickness": 0.28},
                "steps": [
                    {"range": [0, 10], "color": "rgba(239, 68, 68, 0.45)"},      # Danger
                    {"range": [10, 90], "color": "rgba(34, 197, 94, 0.45)"},     # Optimal
                    {"range": [90, 100], "color": "rgba(249, 115, 22, 0.45)"},   # Saturated
                ],
                "threshold": {
                    "line": {"color": "black", "width": 3},
                    "thickness": 0.85,
                    "value": latest.battery_soc_pct
                }
            }
        ))
        fig_gauge.update_layout(
            margin=dict(l=20, r=20, t=10, b=20),
            height=260,
        )
        st.plotly_chart(fig_gauge, use_container_width=True)
        st.caption("🛡️ **Safety Boundaries:** 10% Floor (Longevity) | 90% Ceiling (Overcharge Prevention)")

with tab_sankey:
    st.subheader("Real-Time Energy Flow Sankey Diagram")
    st.write("Visualizes instantaneous apportionment from energy sources to consumers, battery, and macrogrid.")

    # Calculate Sankey flows based on latest snapshot
    # Nodes:
    # 0: Solar Generation
    # 1: Wind Generation
    # 2: Macrogrid Import
    # 3: Battery Discharge
    # 4: Microgrid Distribution Bus
    # 5: Consumer Load
    # 6: Battery Charge
    # 7: Macrogrid Export
    labels = [
        f"Solar ({latest.solar_kw:.1f} kW)",
        f"Wind ({latest.wind_kw:.1f} kW)",
        f"Grid Import ({latest.grid_import_kw:.1f} kW)",
        f"BESS Discharge ({abs(min(0.0, latest.battery_dispatch_kw)):.1f} kW)",
        "Microgrid Central Bus",
        f"Consumer Demand ({latest.load_kw:.1f} kW)",
        f"BESS Charge ({max(0.0, latest.battery_dispatch_kw):.1f} kW)",
        f"Grid Export ({latest.grid_export_kw:.1f} kW)"
    ]

    sources = []
    targets = []
    values = []

    # Inputs into Central Bus
    if latest.solar_kw > 0.1:
        sources.append(0)
        targets.append(4)
        values.append(max(0.1, latest.solar_kw))

    if latest.wind_kw > 0.1:
        sources.append(1)
        targets.append(4)
        values.append(max(0.1, latest.wind_kw))

    if latest.grid_import_kw > 0.1:
        sources.append(2)
        targets.append(4)
        values.append(max(0.1, latest.grid_import_kw))

    if latest.battery_dispatch_kw < -0.1:
        sources.append(3)
        targets.append(4)
        values.append(abs(latest.battery_dispatch_kw))

    # Outputs from Central Bus
    if latest.load_kw > 0.1:
        sources.append(4)
        targets.append(5)
        values.append(max(0.1, latest.load_kw))

    if latest.battery_dispatch_kw > 0.1:
        sources.append(4)
        targets.append(6)
        values.append(latest.battery_dispatch_kw)

    if latest.grid_export_kw > 0.1:
        sources.append(4)
        targets.append(7)
        values.append(latest.grid_export_kw)

    fig_sankey = go.Figure(data=[go.Sankey(
        node=dict(
            pad=18,
            thickness=22,
            line=dict(color="black", width=0.5),
            label=labels,
            color=["#f59e0b", "#06b6d4", "#dc2626", "#8b5cf6", "#3b82f6", "#ef4444", "#10b981", "#22c55e"]
        ),
        link=dict(
            source=sources,
            target=targets,
            value=values,
            color="rgba(148, 163, 184, 0.4)"
        )
    )])
    fig_sankey.update_layout(
        margin=dict(l=10, r=10, t=10, b=10),
        height=320,
    )
    st.plotly_chart(fig_sankey, use_container_width=True)

with tab_history:
    st.subheader("Discrete Time-Step Telemetry Records")
    history_data = [t.to_dict() for t in engine.history]
    st.dataframe(history_data, use_container_width=True, height=280)

st.divider()

# ---------------------------------------------------------
# 3. Agent Operations Feed (Section 4.5)
# ---------------------------------------------------------
st.subheader("🤖 Groq Agent Operations Feed")

if st.session_state.agent_logs:
    latest_log = st.session_state.agent_logs[-1]
    
    with st.expander(f"🔎 Latest Step Operations Report - {latest_log.get('agent_mode', 'Autonomous Agent')}", expanded=True):
        col_res1, col_res2 = st.columns([1.4, 1.0])
        
        with col_res1:
            st.markdown("#### 🧠 Agent Reasoning Log")
            st.info(latest_log.get("reasoning", "No reasoning tokens logged."))
            
        with col_res2:
            st.markdown("#### ⚙️ Executed Tool Calls & Bindings")
            tool_calls = latest_log.get("tool_calls", [])
            if tool_calls:
                for idx, tc in enumerate(tool_calls, 1):
                    st.markdown(f"**Call #{idx}: `{tc.get('name')}`**")
                    st.json(tc.get("arguments", {}))
                    st.caption("Execution Result:")
                    st.json(tc.get("result", {}))
            else:
                st.write("No explicit tool calls triggered.")
                
        st.markdown("---")
        c_conf1, c_conf2, c_conf3 = st.columns(3)
        c_conf1.markdown(f"**Dispatched Battery Power:** `{latest_log.get('dispatched_battery_kw', 0.0):+.2f} kW`")
        c_conf2.markdown(f"**Renewable Curtailment:** `{latest_log.get('curtailment_pct', 0.0) * 100:.1f}%`")
        c_conf3.markdown(f"**Current Hour Period:** `{latest.timestamp_str}`")

st.caption("EcoMesh Microgrid Energy Management System — Antigravity Engineering Protocol")

