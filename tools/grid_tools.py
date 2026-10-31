"""
EcoMesh Agent Tools
LangChain-compatible tools wrapping physical microgrid controls
with Pydantic v2 schemas and deterministic limit clamping.
"""

from typing import Any, Dict, Optional, Type
import json
from pydantic import BaseModel, Field
from langchain_core.tools import BaseTool

from engine.microgrid import MicrogridEngine


# ---------------------------------------------------------
# Pydantic v2 Tool Input Schemas
# ---------------------------------------------------------

class GetGridStateInput(BaseModel):
    """Input schema for retrieving microgrid telemetry state."""
    reason: Optional[str] = Field(
        default="Inspect current generation, demand, and battery state",
        description="Reason or context for querying grid telemetry state."
    )


class DispatchBatteryInput(BaseModel):
    """Input schema for dispatching battery charge or discharge power."""
    target_kw: float = Field(
        ...,
        description=(
            "Target power in kilowatts (kW). "
            "Positive values (>0) indicate charging the battery from the microgrid (max +25.0 kW). "
            "Negative values (<0) indicate discharging the battery to the microgrid (down to -25.0 kW). "
            "0.0 indicates idle."
        )
    )


class CurtailRenewablesInput(BaseModel):
    """Input schema for curtailing renewable generation."""
    curtailment_pct: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description=(
            "Fraction of renewable generation (solar + wind) to curtail, "
            "ranging from 0.0 (0% curtailment, full generation) to 1.0 (100% curtailment, fully shed)."
        )
    )


class QueryTariffPricingInput(BaseModel):
    """Input schema for querying grid electricity tariffs."""
    hour_override: Optional[float] = Field(
        default=None,
        description="Optional hour of day (0.0 - 23.99) to query tariffs for. Defaults to current simulation hour."
    )


# ---------------------------------------------------------
# Tool Implementations
# ---------------------------------------------------------

class GetGridStateTool(BaseTool):
    name: str = "get_grid_state"
    description: str = (
        "Returns the current microgrid telemetry: solar output (kW), wind output (kW), "
        "total load (kW), battery SoC (%), net generation imbalance (kW), and active electricity spot prices."
    )
    args_schema: Type[BaseModel] = GetGridStateInput
    engine: MicrogridEngine = Field(default=None)

    model_config = {"arbitrary_types_allowed": True}

    def _run(self, reason: Optional[str] = None) -> str:
        if self.engine is None:
            return json.dumps({"error": "MicrogridEngine not attached to tool."})
        state = self.engine.get_state()
        return json.dumps({
            "status": "success",
            "hour": state["hour"],
            "solar_kw": state["solar_kw"],
            "wind_kw": state["wind_kw"],
            "total_generation_kw": state["total_generation_kw"],
            "load_kw": state["load_kw"],
            "battery_soc_pct": state["battery_soc_pct"],
            "net_imbalance_kw": state["net_imbalance_kw"],
            "spot_tariff_import_usd_per_kwh": state["tariff_import"],
            "spot_tariff_export_usd_per_kwh": state["tariff_export"],
            "tariff_period": state["tariff_period"],
            "active_curtailment_pct": state["curtailment_pct"],
        }, indent=2)


class DispatchBatteryTool(BaseTool):
    name: str = "dispatch_battery"
    description: str = (
        "Commands the Battery Energy Storage System (BESS) to charge or discharge. "
        "Accepts target_kw (positive to charge, negative to discharge). "
        "Automatically clamps to physical limits [-25.0, 25.0] kW and respects safe SoC boundaries (10% - 90%)."
    )
    args_schema: Type[BaseModel] = DispatchBatteryInput
    engine: MicrogridEngine = Field(default=None)
    last_dispatched_kw: Optional[float] = Field(default=None)

    model_config = {"arbitrary_types_allowed": True}

    def _run(self, target_kw: float) -> str:
        if self.engine is None:
            return json.dumps({"error": "MicrogridEngine not attached to tool."})

        # Physical clamping to [-25.0, +25.0] kW
        requested_kw = float(target_kw)
        clamped_kw = max(-self.engine.p_limit, min(self.engine.p_limit, requested_kw))
        clamped_by_rating = clamped_kw != requested_kw

        current_soc = self.engine.soc
        feasible_kw = clamped_kw
        soc_limited = False

        if clamped_kw > 0:
            # Charging limit check against SoC_max (90%)
            headroom_kwh = max(0.0, (self.engine.soc_max - current_soc) / 100.0 * self.engine.c_max)
            max_charge_kw = headroom_kwh / (self.engine.eta_half * self.engine.dt)
            if clamped_kw > max_charge_kw:
                feasible_kw = max(0.0, max_charge_kw)
                soc_limited = True
        elif clamped_kw < 0:
            # Discharging limit check against SoC_min (10%)
            available_kwh = max(0.0, (current_soc - self.engine.soc_min) / 100.0 * self.engine.c_max)
            max_discharge_kw = (available_kwh * self.engine.eta_half) / self.engine.dt
            if abs(clamped_kw) > max_discharge_kw:
                feasible_kw = -max(0.0, max_discharge_kw)
                soc_limited = True

        self.last_dispatched_kw = round(feasible_kw, 2)

        return json.dumps({
            "status": "staged_dispatch",
            "requested_target_kw": round(requested_kw, 2),
            "clamped_to_rating_kw": round(clamped_kw, 2),
            "effective_feasible_kw": round(feasible_kw, 2),
            "current_soc_pct": round(current_soc, 2),
            "clamped_by_inverter_rating": clamped_by_rating,
            "limited_by_soc_boundary": soc_limited,
            "message": (
                f"Battery dispatch set to {feasible_kw:.2f} kW. "
                f"{'Clamped by inverter limit (+/-25 kW). ' if clamped_by_rating else ''}"
                f"{'Throttled by 10%-90% SoC safe envelope.' if soc_limited else 'Within safe limits.'}"
            )
        }, indent=2)


class CurtailRenewablesTool(BaseTool):
    name: str = "curtail_renewables"
    description: str = (
        "Applies renewable curtailment (shedding) to solar and wind generation when negative pricing, "
        "severe surplus, or grid congestion occurs. Accepts curtailment_pct (0.0 to 1.0)."
    )
    args_schema: Type[BaseModel] = CurtailRenewablesInput
    engine: MicrogridEngine = Field(default=None)
    last_curtailment_pct: Optional[float] = Field(default=None)

    model_config = {"arbitrary_types_allowed": True}

    def _run(self, curtailment_pct: float) -> str:
        if self.engine is None:
            return json.dumps({"error": "MicrogridEngine not attached to tool."})

        pct = max(0.0, min(1.0, float(curtailment_pct)))
        self.last_curtailment_pct = pct
        self.engine.curtailment_pct = pct

        raw_gen = self.engine.solar_kw + self.engine.wind_kw
        curtailed_kw = raw_gen * pct
        effective_gen = raw_gen - curtailed_kw

        return json.dumps({
            "status": "curtailment_updated",
            "curtailment_pct": round(pct, 3),
            "curtailment_percentage_str": f"{pct * 100:.1f}%",
            "raw_generation_kw": round(raw_gen, 2),
            "shed_generation_kw": round(curtailed_kw, 2),
            "effective_generation_kw": round(effective_gen, 2),
            "message": f"Applied {pct * 100:.1f}% renewable curtailment ({curtailed_kw:.2f} kW shed)."
        }, indent=2)


class QueryTariffPricingTool(BaseTool):
    name: str = "query_tariff_pricing"
    description: str = (
        "Queries current and upcoming grid import/export tariffs ($/kWh) and tariff period "
        "(OFF_PEAK, MID_PEAK, PEAK) to enable economic optimization."
    )
    args_schema: Type[BaseModel] = QueryTariffPricingInput
    engine: MicrogridEngine = Field(default=None)

    model_config = {"arbitrary_types_allowed": True}

    def _run(self, hour_override: Optional[float] = None) -> str:
        if self.engine is None:
            return json.dumps({"error": "MicrogridEngine not attached to tool."})

        target_hour = self.engine.hour_of_day if hour_override is None else float(hour_override) % 24.0
        tariffs = self.engine.get_tariffs(target_hour)

        return json.dumps({
            "status": "success",
            "hour": round(target_hour, 2),
            "period": tariffs["period"],
            "import_tariff_usd_per_kwh": tariffs["import_price"],
            "export_tariff_usd_per_kwh": tariffs["export_price"],
            "pricing_guidance": (
                "Peak period: prioritize discharging battery and exporting surplus. "
                "Off-peak period: low import cost, ideal for battery charging if renewables are insufficient. "
                "Mid-peak period: standard balancing."
            )
        }, indent=2)


def create_grid_tools(engine: MicrogridEngine) -> Dict[str, BaseTool]:
    """Factory creating and binding the four specification tools to a MicrogridEngine instance."""
    return {
        "get_grid_state": GetGridStateTool(engine=engine),
        "dispatch_battery": DispatchBatteryTool(engine=engine),
        "curtail_renewables": CurtailRenewablesTool(engine=engine),
        "query_tariff_pricing": QueryTariffPricingTool(engine=engine),
    }

