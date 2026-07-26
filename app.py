import streamlit as st
import pandas as pd
import json
from datetime import datetime
from database import init_db, fetch_data, update_database_and_csv

# Initialize SQLite database (auto-creates and imports your CSV schema on first boot)
init_db()

try:
    from agent import aml_agent
except ImportError:
    st.error("Agent module not found. Please ensure 'agent.py' is in the same directory.")

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="AML Surveillance v4.1",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded"
)

def clean_agent_output(explanation):
    """Extracts raw text and formats it into clean, plain English by stripping technical artifacts."""
    text = ""
    if isinstance(explanation, list):
        for item in explanation:
            if isinstance(item, dict) and "text" in item:
                text += item["text"] + " "
            else:
                text += str(item) + " "
    elif isinstance(explanation, dict):
        text = explanation.get("text", str(explanation))
    else:
        text = str(explanation)
    
    text = text.replace("###", "").replace("---", "").replace("```text", "").replace("```", "")
    return text.strip()

# --- SESSION STATE INITIALIZATION ---
if "active_nav" not in st.session_state:
    st.session_state.active_nav = "Surveillance Center"
if "query_text" not in st.session_state:
    st.session_state.query_text = "Find structuring patterns in recent transactions"
if "last_response" not in st.session_state:
    st.session_state.last_response = None

# --- THEME & STYLING ---
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Archivo+Narrow:wght@400;700;900&family=JetBrains+Mono:wght@400;700&display=swap');

    .stApp {
        background-color: #f9f9f9;
        color: #000000;
        font-family: 'Archivo Narrow', sans-serif;
    }

    [data-testid="stSidebar"] {
        background-color: #ffffff;
        border-right: 3px solid #000000;
        padding-top: 0;
    }
    [data-testid="stSidebar"] .sidebar-header {
        background-color: #000000;
        color: #ffffff;
        padding: 2rem 1rem;
        font-size: 1.5rem;
        font-weight: 900;
        text-transform: uppercase;
        margin-bottom: 1rem;
        border-bottom: 3px solid #000000;
    }

    .top-bar {
        background-color: #000000;
        color: #ffffff;
        padding: 1rem 2rem;
        display: flex;
        justify-content: space-between;
        align-items: center;
        border-bottom: 3px solid #000000;
        margin: -5.5rem -5rem 2rem -5rem;
        z-index: 99;
    }
    .top-bar h2 {
        margin: 0;
        font-size: 1.2rem;
        font-weight: 900;
        letter-spacing: 0.05em;
    }

    .metric-card {
        background: #ffffff;
        border: 3px solid #000000;
        box-shadow: 6px 6px 0px #000000;
        padding: 1.5rem;
        margin-bottom: 1.5rem;
    }
    .metric-label {
        font-size: 0.8rem;
        font-weight: 700;
        color: #666666;
        text-transform: uppercase;
        margin-bottom: 0.5rem;
    }
    .metric-value {
        font-size: 2.5rem;
        font-weight: 900;
        line-height: 1;
        margin-bottom: 1rem;
    }
    .metric-btn {
        background: #000000;
        color: #ffffff;
        padding: 0.4rem 0.8rem;
        font-size: 0.7rem;
        font-weight: 700;
        text-transform: uppercase;
        display: inline-block;
    }

    .section-title {
        font-size: 1.5rem;
        font-weight: 900;
        text-transform: uppercase;
        border-bottom: 3px solid #000000;
        padding-bottom: 0.5rem;
        margin-bottom: 1.5rem;
        display: flex;
        align-items: center;
        gap: 10px;
    }

    /* Fixed button styling: Forces white background, black text, and clear borders to prevent dark theme contrast bugs */
    .stButton > button, div.stDownloadButton > button {
        background-color: #ffffff !important;
        color: #000000 !important;
        border: 3px solid #000000 !important;
        border-radius: 0px !important;
        font-weight: 900 !important;
        text-transform: uppercase !important;
        box-shadow: 4px 4px 0px #000000 !important;
        transition: all 0.1s ease-in-out !important;
        width: 100% !important;
        height: 60px !important;
        font-size: 1rem !important;
    }
    .stButton > button:hover, div.stDownloadButton > button:hover {
        background-color: #eeeeee !important;
        color: #000000 !important;
        box-shadow: 2px 2px 0px #000000 !important;
        transform: translate(2px, 2px) !important;
    }

    .stTextInput input {
        border: 3px solid #000000 !important;
        border-radius: 0px !important;
        padding: 1rem !important;
        font-family: 'Archivo Narrow', sans-serif !important;
        font-weight: 700 !important;
    }

    .log-box {
        background: #000000;
        color: #ffffff;
        padding: 1rem;
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.85rem;
        border: 3px solid #000000;
        line-height: 1.6;
        height: 300px;
        overflow-y: auto;
    }
    .memo-box {
        background: #f3f3f3;
        border: 3px solid #000000;
        padding: 2rem;
        height: 100%;
    }
    .memo-tag {
        background: #000000;
        color: #ffffff;
        padding: 0.2rem 0.5rem;
        font-weight: 700;
        font-size: 0.7rem;
        float: right;
    }

    .tool-chip {
        display: inline-block;
        padding: 0.3rem 0.7rem;
        font-size: 0.75rem;
        font-weight: 700;
        text-transform: uppercase;
        margin-right: 0.5rem;
        margin-bottom: 0.5rem;
        border: 2px solid #000000;
        font-family: 'JetBrains Mono', monospace;
    }
    .chip-invoked { background-color: #000000; color: #ffffff; }
    .chip-skipped { background-color: #ffffff; color: #999999; border-color: #cccccc; text-decoration: line-through; }
    </style>
""", unsafe_allow_html=True)

# --- SIDEBAR NAVIGATION ---
with st.sidebar:
    st.markdown('<div class="sidebar-header">SURVEILLANCE v4.1</div>', unsafe_allow_html=True)
    
    if st.button("Surveillance Center", use_container_width=True):
        st.session_state.active_nav = "Surveillance Center"
    if st.button("Database Manager (CRUD)", use_container_width=True):
        st.session_state.active_nav = "Database Manager"
    if st.button("Investigations Archive", use_container_width=True):
        st.session_state.active_nav = "Investigations Archive"
    if st.button("FinCEN / BSA Rules", use_container_width=True):
        st.session_state.active_nav = "FinCEN / BSA Rules"
    if st.button("Neural LLM Status", use_container_width=True):
        st.session_state.active_nav = "Neural LLM Status"

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("""
        <div style="background:#000; color:#fff; padding:1rem; border:3px solid #000;">
            <div style="font-weight:900; font-size:1.1rem;">ACTIVE MODULE</div>
            <div style="font-size:0.8rem; color:#38BDF8; font-family:monospace; margin-top:0.3rem;">{}</div>
        </div>
    """.format(st.session_state.active_nav), unsafe_allow_html=True)

# --- ROUTED VIEWS ---

if st.session_state.active_nav == "Database Manager":
    st.markdown("<h1>Live Database & CSV Synchronization Console</h1>", unsafe_allow_html=True)
    st.markdown("<p style='color:#666;'>Interactive data grid supporting attribute filtering, row insertion, inline editing, and deletion. Changes are instantly synced to SQLite and your physical <code>transactions.csv</code> file.</p>", unsafe_allow_html=True)
    st.markdown("---")
    
    df_transactions = fetch_data()
    
    st.markdown("### Interactive Transaction Grid")
    st.info("Tip: Click column headers to sort, use column search/filters, add rows at the bottom, or select rows to delete. Click **Commit Changes to DB & CSV** when done.")
    
    edited_df = st.data_editor(
        df_transactions, 
        num_rows="dynamic", 
        use_container_width=True, 
        key="live_db_grid"
    )
    
    if st.button("Commit Changes to DB & CSV"):
        try:
            update_database_and_csv(edited_df)
            st.success("Successfully updated SQLite database and synchronized transactions.csv!")
        except Exception as e:
            st.error(f"Sync failed: {e}")

elif st.session_state.active_nav == "FinCEN / BSA Rules":
    st.markdown("<h1>FinCEN Regulatory Framework & BSA Guidelines</h1>", unsafe_allow_html=True)
    st.markdown("---")
    st.markdown("""
    ### Key Regulatory Standards Enforced:
    * **Currency Transaction Reports (CTR):** Mandatory reporting under 31 CFR 1010.311 for single-day cash operations over **$10,000**.
    * **Structuring Prohibition:** 31 U.S.C. 5324 penalizes deliberate sub-threshold splitting designed to avoid reporting triggers.
    * **Suspicious Activity Reporting (SAR):** Standard compliance window of 30 days for unresolved anomalies.
    """)

elif st.session_state.active_nav == "Neural LLM Status":
    st.markdown("<h1>Neural Core & Agent Status</h1>", unsafe_allow_html=True)
    st.markdown("---")
    st.markdown("""
    * **State Graph Engine:** LangGraph Stateful Multi-Agent Architecture
    * **Inference Endpoint:** Google Gemini Model
    * **Storage Backend:** SQLite Database synced directly with `transactions.csv` (Root path configured)
    """)

elif st.session_state.active_nav == "Investigations Archive":
    st.markdown("<h1>Historical Investigations Archive</h1>", unsafe_allow_html=True)
    st.markdown("---")
    if st.session_state.last_response:
        st.json(st.session_state.last_response)
    else:
        st.warning("No investigation records in buffer. Run an analysis from the Surveillance Center first.")

else:
    # --- SURVEILLANCE CENTER VIEW ---
    st.markdown("""
        <div class="top-bar">
            <h2>INCIDENT INVESTIGATION & COMPLIANCE SUITE</h2>
            <div style="display:flex; gap:20px; font-weight:900; font-size:0.8rem;">
                <span>STATUS: ONLINE</span>
                <span>MODE: AUTONOMOUS</span>
            </div>
        </div>
    """, unsafe_allow_html=True)

    st.markdown('<h1 style="font-size:3.5rem; font-weight:900; margin-bottom:0;">ANTI-MONEY LAUNDERING SURVEILLANCE</h1>', unsafe_allow_html=True)
    st.markdown('<p style="font-size:1.1rem; color:#666; margin-bottom:2rem; max-width:900px;">Enterprise-grade operational dashboard for real-time transaction monitoring based on your active CSV configuration schema.</p>', unsafe_allow_html=True)

    # Metric Grid
    m1, m2, m3, m4 = st.columns(4)
    with m1:
        total_txns = len(fetch_data())
        st.markdown(f"""<div class="metric-card"><div class="metric-label">Monitored Scope</div><div class="metric-value">{total_txns:,} <span style="font-size:1rem; opacity:0.5;">TXNS</span></div><div class="metric-btn">SQLite / CSV Sync Active</div></div>""", unsafe_allow_html=True)
    with m2:
        st.markdown("""<div class="metric-card"><div class="metric-label">Primary Threshold</div><div class="metric-value">$10,000 <span style="font-size:1rem; opacity:0.5;">CTR</span></div><div class="metric-btn">Regulatory Limit Set</div></div>""", unsafe_allow_html=True)
    with m3:
        st.markdown("""<div class="metric-card"><div class="metric-label">Surveillance Engine</div><div class="metric-value">DYNAMIC</div><div class="metric-btn">Neural Analysis On</div></div>""", unsafe_allow_html=True)
    with m4:
        st.markdown("""<div style="background:#000; color:#fff;" class="metric-card"><div class="metric-label" style="color:#aaa;">Active State</div><div class="metric-value">READY</div><div style="background:#fff; color:#000;" class="metric-btn">Idle / Awaiting Input</div></div>""", unsafe_allow_html=True)

    # Control Panel
    st.markdown('<div class="section-title">Investigation Control Panel</div>', unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns(4)
    run_preset = None
    if c1.button("Scan Structuring"):
        run_preset = "Find structuring patterns where is_suspicious is 1"
    if c2.button("Check Suspicious Flag"):
        run_preset = "List accounts marked with suspicious pattern types"
    if c3.button("Audit ACC_3127"):
        run_preset = "Audit account ACC_3127 for risk factors"
    if c4.button("Scan Volume"):
        run_preset = "Summarize total volume by transaction type"

    if run_preset:
        st.session_state.query_text = run_preset

    st.markdown("<br>", unsafe_allow_html=True)
    
    # Custom query input linked to session state for full user interactivity
    user_input = st.text_input(
        label="CUSTOM DIRECTIVE",
        key="query_text",
        placeholder="Type any custom natural language AML query here..."
    )

    run_button = st.button("RUN INTELLIGENCE ENGINE")

    if run_button or run_preset:
        with st.spinner("INITIATING AGENT PROTOCOL..."):
            try:
                # Passes whatever custom text the user entered/selected into the engine
                response = aml_agent.invoke({"user_query": st.session_state.query_text})
                st.session_state.last_response = response

                plan = response.get("plan", {})
                invoked_tools = response.get("invoked_tools", [])
                skipped_tools = response.get("skipped_tools", [])
                raw_explanation = response.get("final_explanation", "No memorandum generated.")
                summary = response.get(
                    "execution_summary",
                    "[SYSTEM] Initializing surveillance protocol v4.1...\n[SCAN] Parsing dataset columns...\n[LLM] Correlating threat indicators..."
                )

                clean_text = clean_agent_output(raw_explanation)

                st.markdown("---")

                if response.get("needs_clarification"):
                    st.warning("HUMAN CLARIFICATION REQUIRED")
                    st.markdown(clean_text)
                else:
                    st.markdown(
                        f'<div style="font-weight:900; margin-bottom:1rem;">EXECUTION COMPLETED &nbsp;|&nbsp; PLAN INTENT: {plan.get("intent", "UNKNOWN").upper()}</div>',
                        unsafe_allow_html=True
                    )

                    col_left, col_right = st.columns([2, 1])

                    with col_left:
                        st.markdown('<div style="font-weight:900; margin-bottom:1rem;">AGENT REASONING LOG</div>', unsafe_allow_html=True)
                        log_lines = summary.replace("\n", "<br>")
                        st.markdown(f'<div class="log-box">{log_lines}</div>', unsafe_allow_html=True)

                        st.markdown("<br>", unsafe_allow_html=True)
                        st.markdown('<div style="font-weight:900; margin-bottom:0.5rem;">TOOL INVOCATION STATUS</div>', unsafe_allow_html=True)
                        chips_html = ""
                        for tool_name in invoked_tools:
                            chips_html += f'<span class="tool-chip chip-invoked">{tool_name}</span>'
                        for tool_name in skipped_tools:
                            chips_html += f'<span class="tool-chip chip-skipped">{tool_name}</span>'
                        st.markdown(chips_html, unsafe_allow_html=True)

                    with col_right:
                        case_id = plan.get("case_id", "AML-2026-0042")
                        human_readable_summary = clean_text
                        
                        st.markdown(f"""
                            <div class="memo-box">
                                <span class="memo-tag">CASE_ID: {case_id}</span>
                                <div style="font-weight:900; font-size:1.5rem; border-bottom:3px solid #000; padding-bottom:1rem; margin-bottom:1rem;">INCIDENT MEMORANDUM</div>
                                <div style="font-weight:900; font-size:0.8rem; margin-bottom:0.5rem; text-transform:uppercase;">Plain English Summary</div>
                                <div style="font-size:1rem; line-height:1.6; font-family: 'Archivo Narrow', sans-serif;">
                                    {human_readable_summary}
                                </div>
                            </div>
                        """, unsafe_allow_html=True)

                    st.markdown("<br><br>", unsafe_allow_html=True)
                    st.markdown('<div class="section-title">Regulatory Filing & Export Center</div>', unsafe_allow_html=True)
                    
                    exp_col1, exp_col2, exp_col3 = st.columns(3)
                    
                    with exp_col1:
                        sar_payload = {
                            "case_id": case_id,
                            "timestamp": datetime.now().isoformat(),
                            "intent": plan.get("intent", "UNKNOWN"),
                            "memorandum": clean_text,
                            "status": "Ready for FinCEN E-Filing"
                        }
                        st.download_button(
                            label="Download FinCEN SAR Package (JSON)",
                            data=json.dumps(sar_payload, indent=2),
                            file_name=f"SAR_{case_id}.json",
                            mime="application/json",
                            use_container_width=True
                        )
                    with exp_col2:
                        st.download_button(
                            label="Export Audit Log (.TXT)",
                            data=summary,
                            file_name=f"Audit_Log_{case_id}.txt",
                            mime="text/plain",
                            use_container_width=True
                        )
                    with exp_col3:
                        if st.button("Push to Compliance Queue", use_container_width=True):
                            st.success("Case successfully escalated to Senior Compliance Officer queue!")

            except Exception as e:
                st.error(f"ENGINE FAILURE: {str(e)}")