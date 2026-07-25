# 🏛️ Problem Statement 1: AI-Powered Suspicious Activity Detection Engine

An autonomous, dynamic AI Agent system built with **LangGraph**, **LangChain**, **Google Gemini**, and **Streamlit** to detect, investigate, and report suspicious financial transaction patterns (such as Structuring/Smurfing and Layering) for BSA/FinCEN regulatory compliance.

---

## 📌 Problem Statement Overview
Financial institutions generate millions of transaction alerts daily. Traditional compliance workflows rely on static rule thresholds or hardcoded sequential pipelines that lack adaptive reasoning. 

This project implements an **Autonomous Dynamic Query Planner**. Rather than following a fixed, linear pipeline, the agent parses natural language queries, extracts structured intents and filters, dynamically constructs an execution plan, and invokes only the tools necessary to answer the specific query—logging skipped nodes and explaining its reasoning transparently.

---

## 📊 Dataset Information & Generation Logic

This project utilizes a synthetic transaction dataset generated via `generate_data.py` (`data/transactions.csv`, 5,339 records).

### Schema & Field Definitions:
* `account_id` (String): Unique identifier for the customer account (e.g., `ACC_1042`, `ACC_SMURF_9003`, `ACC_LAYER_8012`).
* `timestamp` (Datetime): Timestamp of the financial transaction.
* `amount` (Float): Transaction value in USD.
* `transaction_type` (String): Type of fund movement (`PAYMENT`, `TRANSFER`).
* `is_suspicious` (Integer): Binary indicator (1 = Suspicious, 0 = Normal).
* `pattern_type` (String): Ground-truth typology (`Normal`, `Structuring`, `Layering`).

### Injected Typologies & Assumptions:
1. **Structuring (Smurfing)**: Malicious actors execute 4–6 rapid transfers just under the $10,000 Currency Transaction Reporting (CTR) threshold (e.g., $9,000.00 to $9,999.99) within short timeframes (2-hour intervals) to evade BSA reporting requirements.
2. **Layering (Rapid Pass-Through & Fan-Out)**: High-volume incoming transfers ($25,000–$75,000) immediately split and transferred out to multiple accounts within short windows to obscure money trails.

### Dataset Citation & Reference:
Transaction volume, velocity distributions, and evasion thresholds are modeled on publicly available **FinCEN Suspicious Activity Report (SAR) Stats** and synthetic money laundering benchmarks (inspired by Kaggle's *Synthetic Financial Datasets for Fraud Detection*).

---

## 🏗️ Solution Architecture & Dynamic Planning

Unlike a fixed sequential pipeline, the core `agent.py` orchestrates execution via a dynamic JSON planner node (`parse_query_node`).

```text
               +----------------------------------+
               |        User Natural Query        |
               +----------------------------------+
                                |
                                v
               +----------------------------------+
               |   parse_query_node (Gemini LLM)   |
               |  (Extracts Intent, Filters, Plan)|
               +----------------------------------+
                                |
             +------------------+------------------+
             |                                     |
             v (Needs Clarification)               v (Valid Plan)
+-------------------------+             +--------------------------+
|  clarification_node     |             | execute_tools_pipeline   |
| (Human-in-the-Loop Exit)|             | (Dynamic Tool Fan-Out)   |
+-------------------------+             +--------------------------+
             |                                     |
             |       +-----------------------------+-----------------------------+
             |       |                             |                             |
             |       v                             v                             v
             |  [eda_tool]            [feature_engineering_tool]  [pattern_detection_tool]
             |  (Skipped if targeted) (Per-Account Aggregations)   (Structuring/Layering)
             |       |                             |                             |
             |       +-----------------------------+-----------------------------+
             |                                     |
             |                                     v
             |                         [risk_classification_tool]
             |                         (Hybrid ML + Rule Scoring)
             |                                     |
             v                                     v
+----------------------------------------------------------------------------------+
|                            explanation_node                                      |
|            (Generates Execution Summary + BSA Incident Memorandum)               |
+----------------------------------------------------------------------------------+
```

---

## 🛠️ Modular Tool Architecture (`tools.py`)

1. **`run_eda_tool`**: Performs baseline statistical analysis (total volume, transaction count, average amount).
2. **`feature_engineering_tool`**: Computes per-account aggregations (`tx_count`, `total_volume`, `avg_amount`, `max_amount`, `std_amount`), structuring band counts ($9k–$9.99k), near-threshold ratios, and z-score velocity metrics. Operates in full-dataset or single-account mode.
3. **`detect_structuring_tool`**: Rule-based detection uncovering accounts executing 3+ transfers between $9,000 and $9,999.99.
4. **`detect_layering_tool`**: Pattern detector identifying rapid pass-through transfers and fan-out split behaviors.
5. **`risk_classification_tool`**: **Hybrid Scoring Engine** combining Scikit-Learn `IsolationForest` ML anomaly scores with rule hits to assign `Low`, `Medium`, or `High` risk with explicit threshold explanations.

---

## 💻 Tech Stack

* **Orchestration**: LangGraph (`StateGraph`, conditional edges)
* **LLM Engine**: Google Gemini (`gemini-flash-latest`) via `langchain-google-genai`
* **Machine Learning & Analytics**: Scikit-Learn (`IsolationForest`), Pandas, NumPy
* **User Interface**: Streamlit (High-contrast corporate financial theme)
* **Environment Management**: Python 3.10+, `python-dotenv`

---

## 🚀 Setup & Usage Instructions

### 1. Clone & Setup Virtual Environment
```bash
git clone <repository_url>
cd AML-Intelligent-Agent-Version-0
python -m venv aml_env
aml_env\Scripts\activate  # Windows
# source aml_env/bin/activate # Linux/Mac
```

### 2. Install Dependencies
```bash
pip install streamlit pandas numpy scikit-learn langchain-google-genai langgraph python-dotenv
```

### 3. Configure API Key
Create a `.env` file in the project root:
```env
GOOGLE_API_KEY=your_google_ai_studio_api_key
```

### 4. Regenerate Synthetic Dataset (Optional)
```bash
python generate_data.py
```

### 5. Run Terminal CLI Benchmark Test
```bash
python agent.py
```

### 6. Launch Executive Web Dashboard
```bash
streamlit run app.py
```

---

## 🔍 Benchmark Query Matrix (Demonstrating Adaptive Behavior)

| Query | Parsed Intent | Invoked Tools | Skipped Tools | Reasoning / Output |
| :--- | :--- | :--- | :--- | :--- |
| *"Find structuring patterns in the last 30 days"* | `pattern_detection` | `feature_engineering`, `pattern_detection`, `risk_classification` | `eda` | Focuses on sub-threshold CTR evasion. Skips broad EDA. Outputs top high-risk smurfing accounts and recommends SAR filing. |
| *"Which customers made 10+ transactions under $10,000?"* | `aggregation_rule` | `feature_engineering` | `eda`, `pattern_detection`, `risk_classification` | Deterministic rule query. Invokes feature engineering only, skipping ML anomaly detection & EDA. |
| *"Is customer ID ACC_SMURF_9003 suspicious?"* | `single_entity` | `feature_engineering`, `risk_classification` | `eda`, `pattern_detection` | Targeted entity lookup. Extracts single-account features and hybrid risk score (`HIGH RISK`, score 0.95). |

---

## 🤖 Mandatory Disclosures (APIs & AI Assistance)

In accordance with hackathon submission guidelines:
* **LLM API**: This project uses the **Google Gemini API** (`gemini-flash-latest`) via the `langchain-google-genai` SDK.
* **AI Coding Assistance**: Development of this project was assisted by **Antigravity (Google DeepMind Agentic Coding Assistant)** using Gemini 3.1 Pro and Gemini 3.6 Flash for code refactoring, tool integration, and UI styling.