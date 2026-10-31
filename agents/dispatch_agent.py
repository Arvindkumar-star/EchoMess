"""
EcoMesh Autonomous Dispatch Agent
Powered by Groq's high-throughput reasoning models and LangChain tool bindings.
Enforces grid reliability, asset longevity, and economic optimization.
"""

import os
import json
from typing import Dict, Any, List, Optional, Tuple
from dotenv import load_dotenv

from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, ToolMessage
from langchain_groq import ChatGroq
from groq import Groq

from engine.microgrid import MicrogridEngine, MicrogridTelemetry
from tools.grid_tools import create_grid_tools, DispatchBatteryTool, CurtailRenewablesTool

# Load .env file
load_dotenv()

SYSTEM_DISPATCH_PROMPT = """You are the EcoMesh Autonomous Microgrid Dispatch Agent, powered by ultra-low-latency reasoning.
Your primary role is to operate a cyber-physical microgrid consisting of:
- Solar photovoltaic generation (0 to 50 kW peak)
- Wind turbine generation (0 to 40 kW peak)
- Local consumer dynamic load (15 to 60 kW)
- Battery Energy Storage System (BESS): 100 kWh capacity, +/-25 kW inverter rating, 94% round-trip efficiency
- Safe SoC boundaries: 10.0% (minimum) to 90.0% (maximum)
- Macrogrid interconnection for surplus export or deficit import

OPERATIONAL DECISION LOGIC (STRICT PRIORITY ORDER):
1. Priority 1 (Grid Reliability): Prevent microgrid under-voltage or collapse. Local critical consumer load must always be covered.
2. Priority 2 (Asset Longevity): Enforce strict State of Charge limits (10% <= SoC <= 90%). Never attempt to discharge below 10% or charge above 90%. Avoid unnecessary battery micro-cycling.
3. Priority 3 (Economic Optimization):
   - Surplus Condition (Net Generation > Load, P_net > 0):
     * If export tariff is PEAK ($0.24+/kWh), prioritize exporting surplus to macrogrid for high revenue, or top up battery if SoC is low.
     * If tariff is MID or OFF_PEAK, charge the battery to absorb free clean generation up to inverter limit (+25 kW) and SoC ceiling (90%).
   - Deficit Condition (Net Generation < Load, P_net < 0):
     * Discharge the battery (negative target_kw, down to -25 kW) to cover local deficit before importing expensive grid power.
     * If tariff is PEAK ($0.38/kWh), prioritize discharging to eliminate peak import tariffs.
     * If battery SoC reaches 10% minimum threshold, allow necessary grid import to sustain reliability.
   - Renewable Curtailment: Only curtail generation if thermal congestion occurs or grid rejects export.

MANDATORY PROTOCOL:
- You must call `dispatch_battery` with `target_kw` (positive for charging, negative for discharging, 0.0 for idle).
- Optionally call `curtail_renewables` if generation needs to be shed.
- Provide clear, professional engineering reasoning detailing your calculations and operational rationale.
"""


def get_available_groq_models(api_key: str) -> List[str]:
    """Queries the Groq API for available chat/reasoning models on the account."""
    if not api_key or len(api_key.strip()) < 10:
        return []
    try:
        client = Groq(api_key=api_key.strip())
        models = client.models.list()
        chat_models = [
            m.id for m in models.data
            if "whisper" not in m.id.lower() and "guard" not in m.id.lower()
        ]
        priority = ["llama-3.3-70b-versatile", "openai/gpt-oss-120b", "openai/gpt-oss-20b", "qwen/qwen3.8-27b"]
        sorted_models = []
        for p in priority:
            if p in chat_models:
                sorted_models.append(p)
        for m in chat_models:
            if m not in sorted_models:
                sorted_models.append(m)
        return sorted_models
    except Exception:
        return []


class DispatchAgent:
    """
    Autonomous Dispatch Agent wrapper for EcoMesh.
    Handles Groq API connection, prompt orchestration, tool execution, and telemetry feedback.
    """

    def __init__(
        self,
        engine: MicrogridEngine,
        api_key: Optional[str] = None,
        model_name: Optional[str] = None
    ):
        self.engine = engine
        self.tools_dict = create_grid_tools(engine)
        self.tools_list = list(self.tools_dict.values())
        
        # Resolve API key
        self.api_key = (api_key or os.getenv("GROQ_API_KEY", "")).strip()
        self.requested_model = model_name
        self.model_name = model_name or "llama-3.3-70b-versatile"
        self.llm = None
        self.llm_with_tools = None

        self._init_llm()

    def update_api_key(self, api_key: str) -> None:
        """Updates the Groq API key and re-initializes the LLM."""
        self.api_key = api_key.strip()
        self._init_llm()

    def set_model(self, model_name: str) -> None:
        """Sets the Groq foundation model and rebinds tools."""
        self.model_name = model_name.strip()
        self._init_llm()

    def _init_llm(self) -> None:
        """Initializes ChatGroq instance if valid API key is present."""
        if not self.api_key or self.api_key.startswith("your_") or len(self.api_key) <= 10:
            self.llm = None
            self.llm_with_tools = None
            return

        available = get_available_groq_models(self.api_key)
        
        candidates = []
        if available:
            if self.requested_model and self.requested_model in available:
                candidates = [self.requested_model] + [m for m in available if m != self.requested_model]
            elif self.model_name in available:
                candidates = [self.model_name] + [m for m in available if m != self.model_name]
            else:
                candidates = available
        else:
            candidates = [self.model_name or "openai/gpt-oss-120b", "llama-3.3-70b-versatile"]

        for candidate in candidates:
            try:
                llm = ChatGroq(
                    model_name=candidate,
                    temperature=0.1,
                    groq_api_key=self.api_key,
                    max_retries=1,
                )
                self.llm_with_tools = llm.bind_tools(self.tools_list)
                self.llm = llm
                self.model_name = candidate
                return
            except Exception:
                continue

        self.llm = None
        self.llm_with_tools = None

    @property
    def has_groq(self) -> bool:
        """Check if active Groq connection is configured."""
        return self.llm_with_tools is not None

    def _heuristic_dispatch(self, state: Dict[str, Any]) -> Tuple[float, float, str]:
        """
        Deterministic dispatch logic implementing identical priority rules
        when Groq API key is absent or offline.
        """
        net = state["net_imbalance_kw"]
        soc = state["battery_soc_pct"]
        period = state["tariff_period"]
        imp_tariff = state["tariff_import"]
        exp_tariff = state["tariff_export"]
        
        target_battery_kw = 0.0
        curtailment_pct = 0.0
        reasoning_lines = []
        
        reasoning_lines.append(f"Simulation Hour {state['hour']:.1f} ({period}): Solar={state['solar_kw']} kW, Wind={state['wind_kw']} kW, Demand={state['load_kw']} kW.")
        reasoning_lines.append(f"Net Imbalance P_net = {net:+.2f} kW. Battery SoC = {soc:.1f}%.")

        if net > 0:
            if period == "PEAK":
                reasoning_lines.append(f"Peak tariff active (Export @ ${exp_tariff}/kWh). Prioritizing high-revenue grid export.")
                if soc < 40.0:
                    target_battery_kw = min(net * 0.5, 15.0)
                    reasoning_lines.append(f"SoC is moderate ({soc:.1f}%); buffering {target_battery_kw:.2f} kW to battery and exporting balance.")
                else:
                    target_battery_kw = 0.0
                    reasoning_lines.append("Battery SoC is adequate; directing 100% surplus to macrogrid export.")
            else:
                target_battery_kw = min(net, 25.0)
                reasoning_lines.append(f"Off/Mid-peak period. Directing clean surplus of {target_battery_kw:.2f} kW into BESS.")
        else:
            deficit = abs(net)
            reasoning_lines.append(f"Microgrid deficit of {deficit:.2f} kW detected.")
            if soc > 10.5:
                target_battery_kw = -min(deficit, 25.0)
                reasoning_lines.append(f"Discharging BESS at {abs(target_battery_kw):.2f} kW to satisfy local load while preserving SoC >= 10%.")
            else:
                target_battery_kw = 0.0
                reasoning_lines.append(f"Battery SoC is at floor limit ({soc:.1f}%). Halting discharge to preserve cell longevity; relying on grid import.")

        return target_battery_kw, curtailment_pct, "\n".join(reasoning_lines)

    def step(self) -> Dict[str, Any]:
        """
        Executes a single autonomous dispatch cycle.
        """
        state = self.engine.get_state()
        
        target_battery_kw = 0.0
        curtailment_pct = state["curtailment_pct"]
        tool_call_records: List[Dict[str, Any]] = []
        reasoning = ""
        agent_mode = f"Groq {self.model_name}" if self.has_groq else "Deterministic Heuristic (Standby)"

        if self.has_groq:
            try:
                user_content = (
                    f"Current Microgrid Snapshot at hour {state['hour']:.2f} ({state['tariff_period']}):\n"
                    f"- Solar Generation: {state['solar_kw']:.2f} kW\n"
                    f"- Wind Generation: {state['wind_kw']:.2f} kW\n"
                    f"- Total Generation: {state['total_generation_kw']:.2f} kW\n"
                    f"- Consumer Load: {state['load_kw']:.2f} kW\n"
                    f"- Net Imbalance: {state['net_imbalance_kw']:+.2f} kW\n"
                    f"- Battery SoC: {state['battery_soc_pct']:.2f}% (Bounds: 10% - 90%)\n"
                    f"- Grid Import Tariff: ${state['tariff_import']:.2f}/kWh\n"
                    f"- Grid Export Tariff: ${state['tariff_export']:.2f}/kWh\n"
                    f"- Cloud Cover: {state['cloud_cover'] * 100:.0f}%, Wind Drop: {state['wind_drop'] * 100:.0f}%\n\n"
                    "Evaluate operational priorities and execute the dispatch tool calls now."
                )

                messages = [
                    SystemMessage(content=SYSTEM_DISPATCH_PROMPT),
                    HumanMessage(content=user_content),
                ]

                # Invoke Groq model with bound tools
                ai_msg = self.llm_with_tools.invoke(messages)
                reasoning = ai_msg.content or ""

                # Process tool calls
                if hasattr(ai_msg, "tool_calls") and ai_msg.tool_calls:
                    for tc in ai_msg.tool_calls:
                        tool_name = tc.get("name")
                        tool_args = tc.get("args", {})

                        if tool_name in self.tools_dict:
                            tool_inst = self.tools_dict[tool_name]
                            tool_result_str = tool_inst.invoke(tool_args)
                            try:
                                parsed_res = json.loads(tool_result_str)
                            except Exception:
                                parsed_res = tool_result_str

                            tool_call_records.append({
                                "name": tool_name,
                                "arguments": tool_args,
                                "result": parsed_res,
                            })

                            if tool_name == "dispatch_battery":
                                target_battery_kw = float(tool_args.get("target_kw", 0.0))
                            elif tool_name == "curtail_renewables":
                                curtailment_pct = float(tool_args.get("curtailment_pct", 0.0))
                else:
                    h_batt, h_curt, h_reason = self._heuristic_dispatch(state)
                    target_battery_kw = h_batt
                    curtailment_pct = h_curt
                    reasoning += f"\n\n[Autonomous Safeguard Applied]: {h_reason}"

            except Exception as e:
                agent_mode = f"Fallback due to Groq error ({str(e)[:60]})"
                h_batt, h_curt, h_reason = self._heuristic_dispatch(state)
                target_battery_kw = h_batt
                curtailment_pct = h_curt
                reasoning = f"Groq API Notice: {e}\n\nFalling back to deterministic dispatch:\n{h_reason}"
        else:
            h_batt, h_curt, h_reason = self._heuristic_dispatch(state)
            target_battery_kw = h_batt
            curtailment_pct = h_curt
            reasoning = (
                "ℹ️ Groq API key not set in .env. Operating in deterministic rule-based simulation mode.\n\n"
                f"{h_reason}"
            )
            tool_call_records.append({
                "name": "dispatch_battery",
                "arguments": {"target_kw": target_battery_kw},
                "result": {
                    "status": "deterministic_dispatch",
                    "dispatched_kw": target_battery_kw,
                    "rationale": "Priority-ordered heuristic dispatch"
                }
            })

        telemetry = self.engine.dispatch(
            target_battery_kw=target_battery_kw,
            curtailment_pct=curtailment_pct,
        )

        return {
            "agent_mode": agent_mode,
            "dispatched_battery_kw": target_battery_kw,
            "curtailment_pct": curtailment_pct,
            "reasoning": reasoning,
            "tool_calls": tool_call_records,
            "telemetry": telemetry,
            "state_before": state,
        }
