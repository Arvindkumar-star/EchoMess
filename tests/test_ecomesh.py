"""
EcoMesh Automated Test Suite
Validates physics engine mathematical constraints, Pydantic tool schemas,
clamping behavior, and dispatch agent operations.
"""

import sys
import os
import math

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from engine.microgrid import MicrogridEngine, MicrogridTelemetry
from tools.grid_tools import create_grid_tools
from agents.dispatch_agent import DispatchAgent


def test_microgrid_engine_initial_parameters():
    """Validates core parameters against spec."""
    engine = MicrogridEngine(seed=10)
    assert engine.c_max == 100.0
    assert engine.p_limit == 25.0
    assert engine.eta == 0.94
    assert math.isclose(engine.eta_half, math.sqrt(0.94), rel_tol=1e-4)
    assert engine.soc_min == 10.0
    assert engine.soc_max == 90.0
    assert engine.soc == 50.0


def test_battery_charge_math_and_limits():
    """Validates charge equation and 90% SoC upper boundary clamp."""
    engine = MicrogridEngine(initial_soc_pct=85.0, seed=10)
    # Available headroom: (90 - 85) / 100 * 100 = 5.0 kWh
    # Max charge power feasible = 5.0 / (sqrt(0.94) * 1.0) ~ 5.157 kW
    telemetry = engine.dispatch(target_battery_kw=25.0)
    assert engine.soc <= 90.001
    assert telemetry.battery_dispatch_kw < 6.0  # throttled by SoC headroom


def test_battery_discharge_math_and_limits():
    """Validates discharge equation and 10% SoC floor clamp."""
    engine = MicrogridEngine(initial_soc_pct=15.0, seed=10)
    # Available energy: (15 - 10) / 100 * 100 = 5.0 kWh
    # Max discharge power = 5.0 * sqrt(0.94) / 1.0 ~ 4.847 kW
    telemetry = engine.dispatch(target_battery_kw=-25.0)
    assert engine.soc >= 9.999
    assert abs(telemetry.battery_dispatch_kw) < 6.0  # throttled by SoC floor


def test_grid_tools():
    """Validates LangChain tool execution and Pydantic v2 validation."""
    engine = MicrogridEngine(seed=10)
    tools = create_grid_tools(engine)
    
    # 1. GetGridState
    state_res = tools["get_grid_state"].invoke({})
    assert "battery_soc_pct" in state_res
    assert "net_imbalance_kw" in state_res

    # 2. DispatchBattery clamped to rating
    dispatch_res = tools["dispatch_battery"].invoke({"target_kw": 50.0})
    assert "clamped_to_rating_kw" in dispatch_res
    assert "25.0" in dispatch_res

    # 3. CurtailRenewables
    curtail_res = tools["curtail_renewables"].invoke({"curtailment_pct": 0.5})
    assert "50.0%" in curtail_res

    # 4. QueryTariffPricing
    tariff_res = tools["query_tariff_pricing"].invoke({})
    assert "import_tariff_usd_per_kwh" in tariff_res


def test_dispatch_agent_cycle():
    """Validates that dispatch agent steps through physics and logs properly."""
    engine = MicrogridEngine(seed=42)
    agent = DispatchAgent(engine)
    
    result = agent.step()
    assert "agent_mode" in result
    assert "telemetry" in result
    assert len(engine.history) == 1
    assert -25.0 <= result["dispatched_battery_kw"] <= 25.0


if __name__ == "__main__":
    test_microgrid_engine_initial_parameters()
    test_battery_charge_math_and_limits()
    test_battery_discharge_math_and_limits()
    test_grid_tools()
    test_dispatch_agent_cycle()
    print("All unit tests passed successfully!")
