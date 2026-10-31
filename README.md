# EcoMesh: Autonomous Microgrid Energy Management System

Autonomous microgrid balancing, dynamic battery energy storage dispatch (BESS), and real-time telemetry powered by **Groq** high-throughput LLMs, **LangChain**, and **Streamlit**.

---

## ⚡ Overview

EcoMesh is an intelligent cyber-physical microgrid optimization platform designed to balance:
- Distributed renewable generation (diurnal solar curve up to $50\text{ kW}$ + Weibull wind up to $40\text{ kW}$)
- Dynamic consumer load profile ($15\text{ kW} - 60\text{ kW}$)
- Battery Energy Storage Systems (BESS: $100\text{ kWh}$, $\pm 25\text{ kW}$, $\eta = 0.94$, safe SoC $10\% - 90\%$)
- Macrogrid interconnection for surplus export and deficit import based on Time-of-Use (ToU) tariffs

Autonomous sub-second decision making is orchestrated via an LLM dispatch agent equipped with explicit LangChain tool bindings (`dispatch_battery`, `curtail_renewables`, `get_grid_state`, `query_tariff_pricing`).

---

## 🏗️ Architecture

```
EchoMess/
├── .env.example              # Environment variables template
├── .gitignore                # Security filter preventing secret leaks
├── requirements.txt          # Pinned dependencies
├── app.py                    # Streamlit telemetry console & operator UI
├── engine/
│   ├── __init__.py
│   └── microgrid.py          # Deterministic physical simulation engine
├── tools/
│   ├── __init__.py
│   └── grid_tools.py         # Pydantic v2 LangChain tools
├── agents/
│   ├── __init__.py
│   └── dispatch_agent.py     # Autonomous Groq reasoning & dispatch agent
└── tests/
    └── test_ecomesh.py       # Automated physical constraint tests
```

---

## 🚀 Quick Start

### 1. Clone & Set Up Virtual Environment

```bash
git clone https://github.com/Arvindkumar-star/EchoMess.git
cd EchoMess

python -m venv .venv
# Windows
.venv\Scripts\activate
# Linux/macOS
source .venv/bin/activate
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Configure Environment

Copy `.env.example` to `.env` and add your Groq API key:

```bash
cp .env.example .env
```

Edit `.env`:
```env
GROQ_API_KEY=gsk_your_groq_api_key_here
STREAMLIT_SERVER_PORT=8501
```

*(If no key is configured, the system seamlessly runs in deterministic standby mode).*

### 4. Run the Dashboard

```bash
streamlit run app.py
```

Access the UI at `http://localhost:8501`.

---

## 🛡️ Security
- Secrets and API credentials are kept out of source control via strict `.gitignore` rules.
- Never commit `.env` or personal credentials to public repositories.

---

## 🧪 Testing

Run the automated test suite to verify physical constraints, battery math, and tool executions:

```bash
python tests/test_ecomesh.py
```
