"""
EcoMesh Microgrid Physics Engine
Deterministic simulation of renewable generation, dynamic load profiles,
battery energy storage systems (BESS), and macrogrid exchanges.
"""

from dataclasses import dataclass, field
import math
import random
from typing import Dict, Any, List, Optional


@dataclass
class MicrogridTelemetry:
    """Snapshot of microgrid state at a specific simulation timestep."""
    step: int
    hour: float
    solar_kw: float
    wind_kw: float
    total_generation_kw: float
    load_kw: float
    net_imbalance_kw: float
    battery_dispatch_kw: float  # >0 charging, <0 discharging
    battery_soc_pct: float
    grid_import_kw: float
    grid_export_kw: float
    tariff_import: float
    tariff_export: float
    curtailment_pct: float
    timestamp_str: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "step": self.step,
            "hour": round(self.hour, 2),
            "solar_kw": round(self.solar_kw, 2),
            "wind_kw": round(self.wind_kw, 2),
            "total_generation_kw": round(self.total_generation_kw, 2),
            "load_kw": round(self.load_kw, 2),
            "net_imbalance_kw": round(self.net_imbalance_kw, 2),
            "battery_dispatch_kw": round(self.battery_dispatch_kw, 2),
            "battery_soc_pct": round(self.battery_soc_pct, 2),
            "grid_import_kw": round(self.grid_import_kw, 2),
            "grid_export_kw": round(self.grid_export_kw, 2),
            "tariff_import": round(self.tariff_import, 4),
            "tariff_export": round(self.tariff_export, 4),
            "curtailment_pct": round(self.curtailment_pct, 3),
            "timestamp_str": self.timestamp_str,
        }


class MicrogridEngine:
    """
    Physical simulation engine for the EcoMesh microgrid.
    
    Parameters per spec:
      - Battery Capacity (C_max): 100.0 kWh
      - Max Charge/Discharge Rate (P_limit): 25.0 kW
      - Round-Trip Efficiency (eta): 0.94 (sqrt(eta) ~ 0.9695 applied symmetrically)
      - State of Charge Bounds: SoC_min = 10.0%, SoC_max = 90.0%
      - Baseline Demand: 15.0 kW to 60.0 kW with stochastic variations
      - Solar Generation: Diurnal curve scaling to peak 50.0 kW
      - Wind Generation: Weibull-distributed fluctuating between 0.0 kW and 40.0 kW
    """

    def __init__(
        self,
        battery_capacity_kwh: float = 100.0,
        battery_power_limit_kw: float = 25.0,
        round_trip_efficiency: float = 0.94,
        soc_min_pct: float = 10.0,
        soc_max_pct: float = 90.0,
        initial_soc_pct: float = 50.0,
        dt_hours: float = 1.0,
        seed: Optional[int] = 42,
    ):
        self.c_max = battery_capacity_kwh
        self.p_limit = battery_power_limit_kw
        self.eta = round_trip_efficiency
        self.eta_half = math.sqrt(self.eta)  # ~0.9695
        self.soc_min = soc_min_pct
        self.soc_max = soc_max_pct
        self.soc = max(self.soc_min, min(self.soc_max, initial_soc_pct))
        self.dt = dt_hours
        
        self.step_count = 0
        self.hour_of_day = 8.0  # start at 08:00 morning
        
        # Override parameters
        self.cloud_cover = 0.0  # 0.0 to 1.0
        self.wind_drop = 0.0    # 0.0 to 1.0
        self.curtailment_pct = 0.0
        
        # Current instantaneous states (before dispatch in current step)
        self.solar_kw = 0.0
        self.wind_kw = 0.0
        self.load_kw = 30.0
        self.net_imbalance_kw = 0.0
        self.battery_dispatch_kw = 0.0
        self.grid_import_kw = 0.0
        self.grid_export_kw = 0.0
        
        # Random generator for stochastic variations
        self.rng = random.Random(seed)
        
        # History of telemetry steps
        self.history: List[MicrogridTelemetry] = []
        
        # Initialize environmental state for initial step
        self._generate_environment()

    def _generate_solar(self, hour: float) -> float:
        """Diurnal solar curve scaling to peak capacity of 50.0 kW."""
        # Sunlight between 06:00 and 18:00
        if 6.0 <= hour <= 18.0:
            # Normalized solar angle
            angle = math.pi * (hour - 6.0) / 12.0
            base_solar = math.sin(angle) * 50.0
            # Small stochastic solar variability (+/- 5%)
            jitter = 1.0 + self.rng.uniform(-0.05, 0.05)
            solar = max(0.0, base_solar * jitter)
        else:
            solar = 0.0
        
        # Cloud cover attenuation
        solar *= max(0.0, 1.0 - self.cloud_cover)
        return min(50.0, max(0.0, solar))

    def _generate_wind(self) -> float:
        """Weibull-distributed wind profile fluctuating between 0.0 kW and 40.0 kW."""
        # Weibull distribution with shape parameter k=2.0, scale parameter lam=18.0
        # Typical wind power density
        sample = self.rng.weibullvariate(alpha=18.0, beta=2.0)
        # Scale to microgrid 40 kW max capacity
        wind = min(40.0, max(0.0, sample))
        # Wind drop attenuation
        wind *= max(0.0, 1.0 - self.wind_drop)
        return wind

    def _generate_load(self, hour: float) -> float:
        """
        Baseline demand varying between 15.0 kW and 60.0 kW with stochastic variations.
        Dual-peak diurnal human activity profile:
        - Morning peak around 08:00 - 10:00 (~45 kW)
        - Evening peak around 18:00 - 21:00 (~55 kW)
        - Base night load ~18 kW
        """
        # Morning peak Gaussian
        morning_peak = 20.0 * math.exp(-((hour - 9.0) ** 2) / 4.0)
        # Evening peak Gaussian
        evening_peak = 28.0 * math.exp(-((hour - 19.5) ** 2) / 6.0)
        
        base = 18.0 + morning_peak + evening_peak
        # Stochastic variation (+/- 4.0 kW)
        noise = self.rng.uniform(-4.0, 4.0)
        load = base + noise
        return min(60.0, max(15.0, load))

    def get_tariffs(self, hour: float) -> Dict[str, float]:
        """
        Time-of-Use (ToU) electricity tariffs ($/kWh).
        - Off-peak (23:00 - 06:00): Low import ($0.08), low export ($0.04)
        - Mid-peak (06:00 - 16:00, 21:00 - 23:00): Moderate ($0.16 import, $0.09 export)
        - Peak (16:00 - 21:00): High import ($0.38), high export ($0.24)
        """
        if 16.0 <= hour < 21.0:
            return {"import_price": 0.38, "export_price": 0.24, "period": "PEAK"}
        elif 23.0 <= hour or hour < 6.0:
            return {"import_price": 0.08, "export_price": 0.04, "period": "OFF_PEAK"}
        else:
            return {"import_price": 0.16, "export_price": 0.09, "period": "MID_PEAK"}

    def _generate_environment(self) -> None:
        """Calculates solar, wind, load, and net imbalance for current hour."""
        self.solar_kw = self._generate_solar(self.hour_of_day)
        self.wind_kw = self._generate_wind()
        self.load_kw = self._generate_load(self.hour_of_day)
        
        # Effective renewable generation considering active curtailment
        effective_gen = (self.solar_kw + self.wind_kw) * (1.0 - self.curtailment_pct)
        self.net_imbalance_kw = effective_gen - self.load_kw

    def get_state(self) -> Dict[str, Any]:
        """Returns the current state dictionary of the microgrid."""
        tariffs = self.get_tariffs(self.hour_of_day)
        return {
            "step": self.step_count,
            "hour": round(self.hour_of_day, 2),
            "solar_kw": round(self.solar_kw, 2),
            "wind_kw": round(self.wind_kw, 2),
            "total_generation_kw": round(self.solar_kw + self.wind_kw, 2),
            "effective_generation_kw": round((self.solar_kw + self.wind_kw) * (1.0 - self.curtailment_pct), 2),
            "load_kw": round(self.load_kw, 2),
            "battery_soc_pct": round(self.soc, 2),
            "net_imbalance_kw": round(self.net_imbalance_kw, 2),
            "tariff_import": tariffs["import_price"],
            "tariff_export": tariffs["export_price"],
            "tariff_period": tariffs["period"],
            "curtailment_pct": round(self.curtailment_pct, 3),
            "cloud_cover": round(self.cloud_cover, 2),
            "wind_drop": round(self.wind_drop, 2),
        }

    def dispatch(self, target_battery_kw: float, curtailment_pct: Optional[float] = None) -> MicrogridTelemetry:
        """
        Executes a dispatch command on the microgrid and advances the simulation by dt.
        
        :param target_battery_kw: Desired battery power in kW.
               Positive = Charge battery.
               Negative = Discharge battery.
        :param curtailment_pct: Renewable generation curtailment (0.0 to 1.0).
        """
        if curtailment_pct is not None:
            self.curtailment_pct = max(0.0, min(1.0, float(curtailment_pct)))
            
        # Re-evaluate effective generation with any updated curtailment
        effective_gen = (self.solar_kw + self.wind_kw) * (1.0 - self.curtailment_pct)
        self.net_imbalance_kw = effective_gen - self.load_kw
        
        # 1. Clamp target power to battery physical rating [-25.0, 25.0] kW
        requested_kw = max(-self.p_limit, min(self.p_limit, float(target_battery_kw)))
        actual_battery_kw = 0.0
        
        if requested_kw > 0:
            # CHARGING
            # Max energy that can be added before reaching SoC_max (90%)
            headroom_kwh = max(0.0, (self.soc_max - self.soc) / 100.0 * self.c_max)
            # Energy to battery = P_batt * eta_half * dt <= headroom_kwh
            max_charge_kw = headroom_kwh / (self.eta_half * self.dt) if self.dt > 0 else 0.0
            actual_battery_kw = min(requested_kw, max_charge_kw)
            
            # Update SoC
            delta_soc_pct = (actual_battery_kw * self.eta_half * self.dt / self.c_max) * 100.0
            self.soc = min(self.soc_max, self.soc + delta_soc_pct)
            
        elif requested_kw < 0:
            # DISCHARGING
            discharge_power = abs(requested_kw)
            # Max energy that can be extracted before reaching SoC_min (10%)
            available_kwh = max(0.0, (self.soc - self.soc_min) / 100.0 * self.c_max)
            # Energy extracted from chemical storage = (P_discharge * dt) / eta_half <= available_kwh
            max_discharge_kw = (available_kwh * self.eta_half) / self.dt if self.dt > 0 else 0.0
            actual_discharge_kw = min(discharge_power, max_discharge_kw)
            actual_battery_kw = -actual_discharge_kw
            
            # Update SoC
            delta_soc_pct = (actual_discharge_kw * self.dt / (self.eta_half * self.c_max)) * 100.0
            self.soc = max(self.soc_min, self.soc - delta_soc_pct)
            
        else:
            actual_battery_kw = 0.0
            
        self.battery_dispatch_kw = actual_battery_kw
        
        # 2. Grid Exchange Calculation
        # Residual power post-battery dispatch:
        # If charging (actual_battery_kw > 0): battery is additional load
        # If discharging (actual_battery_kw < 0): battery is additional source
        net_residual = effective_gen - self.load_kw - actual_battery_kw
        
        if net_residual > 0:
            self.grid_export_kw = net_residual
            self.grid_import_kw = 0.0
        else:
            self.grid_export_kw = 0.0
            self.grid_import_kw = abs(net_residual)
            
        tariffs = self.get_tariffs(self.hour_of_day)
        timestamp_str = f"Day {int(self.hour_of_day // 24) + 1}, {int(self.hour_of_day % 24):02d}:00"
        
        # 3. Create telemetry record
        telemetry = MicrogridTelemetry(
            step=self.step_count,
            hour=self.hour_of_day,
            solar_kw=self.solar_kw,
            wind_kw=self.wind_kw,
            total_generation_kw=self.solar_kw + self.wind_kw,
            load_kw=self.load_kw,
            net_imbalance_kw=self.net_imbalance_kw,
            battery_dispatch_kw=self.battery_dispatch_kw,
            battery_soc_pct=self.soc,
            grid_import_kw=self.grid_import_kw,
            grid_export_kw=self.grid_export_kw,
            tariff_import=tariffs["import_price"],
            tariff_export=tariffs["export_price"],
            curtailment_pct=self.curtailment_pct,
            timestamp_str=timestamp_str,
        )
        self.history.append(telemetry)
        
        # 4. Advance time step
        self.step_count += 1
        self.hour_of_day = (self.hour_of_day + self.dt) % 24.0
        
        # 5. Generate environment for next step
        self._generate_environment()
        
        return telemetry

    def reset(self, initial_soc_pct: float = 50.0, seed: Optional[int] = 42) -> None:
        """Resets the microgrid engine to its initial state."""
        self.soc = max(self.soc_min, min(self.soc_max, initial_soc_pct))
        self.step_count = 0
        self.hour_of_day = 8.0
        self.cloud_cover = 0.0
        self.wind_drop = 0.0
        self.curtailment_pct = 0.0
        self.history.clear()
        if seed is not None:
            self.rng = random.Random(seed)
        self._generate_environment()

