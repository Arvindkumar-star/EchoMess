EcoMesh: Autonomous Microgrid Energy Management System
Autonomous microgrid balancing, dynamic battery dispatch, and real-time telemetry powered by Groq and LangChain.
1. Project Goal & Overview
EcoMesh is an intelligent, autonomous microgrid optimization and telemetry platform designed to balance distributed renewable generation, local consumer demand, and battery energy storage systems (BESS). By combining a physical microgrid simulation engine with an autonomous LLM-driven dispatch agent powered by Groq's high-throughput llama-3.3-70b-versatile, EcoMesh achieves sub-second decision-making to optimize economic yield, guarantee grid stability, and maximize carbon-neutral operational uptime.

The system continuously models physical grid parameters, calculates energy imbalances, and provides interactive control to human operators while orchestrating automated, structured dispatch operations.
2. Technical Stack & Environment Requirements
EcoMesh relies on deterministic execution, low-latency API inference, and rigorous data models across the entire stack.

Dependency / Component
Version / Specification
Role in Architecture
Python
>=3.11.0
Core runtime platform
Streamlit
>=1.30.0
Reactive operator UI and live control surface
Plotly
>=5.18.0
Real-time interactive telemetry charts and energy flow Sankey diagrams
Groq API Client
groq>=0.4.0
High-speed LLM inference gateway for autonomous dispatching
Target Foundation Model
llama-3.3-70b-versatile
Ultra-low latency reasoning agent for grid balancing
LangChain
>=0.1.0
Agent framework, dynamic tool schemas, and prompt bindings
LangChain Groq
langchain-groq>=0.0.1
ChatGroq integration provider
Pydantic
>=2.0.0
Type enforcement, structured tool arguments, and telemetry validation
python-dotenv
>=1.0.0
Secure environment variable management

3. Required File Directory Structure
Antigravity must generate and structure the repository strictly adhering to the file layout below:

ecomesh/
├── .env
├── requirements.txt
├── app.py
├── engine/
│   ├── __init__.py
│   └── microgrid.py
├── tools/
│   ├── __init__.py
│   └── grid_tools.py
└── agents/
    ├── __init__.py
    └── dispatch_agent.py
Manifest Specifications
requirements.txt: Must contain exact pinned dependencies: streamlit>=1.30.0, plotly>=5.18.0, langchain>=0.1.0, langchain-groq>=0.0.1, groq>=0.4.0, pydantic>=2.0.0, and python-dotenv>=1.0.0.
.env: Must define placeholder configuration keys: GROQ_API_KEY= and STREAMLIT_SERVER_PORT=8501.
4. Module Implementation Directives
4.1. Environment Configuration (.env)
Antigravity must establish .env parsing via python-dotenv across all entry points, gracefully handling instances where GROQ_API_KEY is not set by providing clear error messages within the UI.
4.2. Microgrid Physics Engine (engine/microgrid.py)
Implement the MicrogridEngine class responsible for maintaining and updating state variables across discrete simulation steps.

Core Parameters:
Battery Capacity: $C_{\text{max}} = 100.0\text{ kWh}$
Max Charge/Discharge Rate: $P_{\text{limit}} = 25.0\text{ kW}$
Round-Trip Efficiency: $\eta = 0.94$ (applied symmetrically as $\sqrt{\eta} \approx 0.9695$ during charging and discharging)
State of Charge Bounds: $\text{SoC}{\text{min}} = 10%$, $\text{SoC}{\text{max}} = 90%$
Baseline Demand: Base load profile varying between $15.0\text{ kW}$ and $60.0\text{ kW}$ with stochastic variations.
Solar Generation: Diurnal solar curve scaling to a peak capacity of $50.0\text{ kW}$.
Wind Generation: Weibull-distributed profile fluctuating between $0.0\text{ kW}$ and $40.0\text{ kW}$.
Step Calculations:
$\text{Net Imbalance } (P_{\text{net}}) = (P_{\text{solar}} + P_{\text{wind}}) - P_{\text{load}}$
Surplus occurs when $P_{\text{net}} > 0$; Deficit occurs when $P_{\text{net}} < 0$.
Battery physical limits:
$$\Delta \text{SoC}{\text{charge}} = \frac{P{\text{batt_charge}} \cdot \sqrt{\eta} \cdot \Delta t}{C_{\text{max}}}$$
$$\Delta \text{SoC}{\text{discharge}} = \frac{P{\text{batt_discharge}}}{\sqrt{\eta} \cdot C_{\text{max}}} \cdot \Delta t$$
Grid Exchange: Residual power post-battery dispatch is exported to or imported from the main macrogrid.
4.3. Agent Tools (tools/grid_tools.py)
Wrap physical grid controls into LangChain-compatible tools using Pydantic v2 schemas:

GetGridState: Returns current solar output, wind output, total load, battery SoC, net imbalance, and active electricity spot price.
DispatchBattery: Accepts parameter target_kw (positive for charge, negative for discharge). Clamps power automatically to $[-25.0, 25.0]$ kW and validates against SoC safe thresholds ($10% - 90%$).
CurtailRenewables: Accepts curtailment_pct ($0.0 - 1.0$) to shed excess generation when negative pricing or thermal grid congestion occurs.
QueryTariffPricing: Returns current grid import/export tariffs to allow cost-optimal dispatch decisions.
4.4. Groq Dispatch Agent (agents/dispatch_agent.py)
Initialize ChatGroq(model_name="llama-3.3-70b-versatile", temperature=0.1).

Operational Decision Logic:
Objective Priority 1: Grid Reliability: Prevent microgrid under-voltage or collapse; prioritize local critical load.
Objective Priority 2: Asset Longevity: Enforce strict SoC limits ($10% \le \text{SoC} \le 90%$) and avoid unnecessary battery cycling.
Objective Priority 3: Economic Optimization: If $P_{\text{net}} > 0$ and tariffs are low, charge battery; if tariffs are peak, prioritize macrogrid export. If $P_{\text{net}} < 0$, discharge battery up to $P_{\text{limit}}$ before importing expensive grid power.
Structured Response Protocol: The agent must return reasoning logs along with explicit function execution calls via LangChain tool bindings.
4.5. Dashboard Interface (app.py)
The Streamlit application must deliver a responsive monitoring and control console:

Top Metric Ribbon: Real-time KPI cards displaying Total Generation (kW), Microgrid Demand (kW), Battery SoC (%), and Instantaneous Macrogrid Import/Export (kW).
Telemetry Panels (Plotly):
Time-series line chart showing dynamic generation vs. load over simulation windows.
Gauge chart displaying current Battery SoC with color-coded safety zones ($<10%$ danger, $10%-90%$ optimal, $>90%$ saturated).
Real-time power distribution chart (Sankey or stacked area) visualizing energy source apportionment.
Agent Operations Feed: Expandable container showing step-by-step reasoning tokens, selected tool calls, and execution confirmation from llama-3.3-70b-versatile.
Manual Override Console: Sidebar sliders to force weather events (cloud cover, wind drop) or initiate immediate manual battery charge/discharge overrides.
5. Execution Workflow Instructions for Antigravity
Antigravity must execute the construction of EcoMesh using the following operational sequence:

Step 1: Workspace Initialization: Validate Python environment; generate directory hierarchy (engine/, tools/, agents/); generate .env and requirements.txt.
Step 2: Physics Engine Construction: Author engine/microgrid.py with full state update logic, mathematical constraints, and deterministic stepping functions.
Step 3: Tool Binding & Schema Definition: Author tools/grid_tools.py, declaring explicit Pydantic v2 argument models for all dispatch actions.
Step 4: Groq Agent Orchestration: Author agents/dispatch_agent.py, configuring system prompts, tool registration, and error handling for model responses.
Step 5: Frontend Assembly: Author app.py, linking session state with the MicrogridEngine, running the dispatch agent cycle, and rendering Plotly visualizations.
Step 6: System Verification: Execute streamlit run app.py --server.headless true, verify simulated time steps, validate tool call executions, and ensure zero unhandled exceptions.
