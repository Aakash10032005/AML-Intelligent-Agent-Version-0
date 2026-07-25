import streamlit as st
import pandas as pd

# Note: 'aml_agent' is imported as per your original code.
# Ensure agent.py exists in your local directory.
try:
    from agent import aml_agent
except ImportError:
    st.error("Agent module not found. Please ensure 'agent.py' is in the same directory.")

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="AML Surveillance v4.1",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Helper function to extract ONLY clean narrative text from Gemini outputs.
# Restored from the original build — Gemini sometimes returns a list of content
# blocks or a dict instead of a plain string, and this normalizes it safely.
def clean_agent_output(explanation):
    if isinstance(explanation, list):
        for item in explanation:
            if isinstance(item, dict) and "text" in item:
                return item["text"]
        return str(explanation)
    elif isinstance(explanation, dict):
        if "text" in explanation:
            return explanation["text"]
        return str(explanation)
    return str(explanation)

# --- THEME & STYLING (Monochrome Sentinel / Neo-Brutalist) ---
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Archivo+Narrow:wght@400;700;900&family=JetBrains+Mono:wght@400;700&display=swap');

    /* Global Overrides */
    .stApp {
        background-color: #f9f9f9;
        color: #000000;
        font-family: 'Archivo Narrow', sans-serif;
    }

    /* Sidebar Customization */
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
        margin-bottom: 2rem;
        border-bottom: 3px solid #000000;
    }

    /* Navigation Menu Items */
    .nav-item {
        padding: 1rem;
        border-bottom: 1px solid #eeeeee;
        font-weight: 700;
        display: flex;
        align-items: center;
        gap: 10px;
        cursor: pointer;
    }
    .nav-item:hover {
        background-color: #f3f3f3;
    }
    .nav-item.active {
        background-color: #000000;
        color: #ffffff;
    }

    /* Top Bar */
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

    /* Metric Cards (Neo-Brutalist) */
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

    /* Control Panel Section */
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

    /* Neo-Brutalist Buttons */
    .stButton > button {
        background-color: #ffffff !important;
        color: #000000 !important;
        border: 3px solid #000000 !important;
        border-radius: 0px !important;
        font-weight: 900 !important;
        text-transform: uppercase !important;
        box-shadow: 4px 4px 0px #000000 !important;
        transition: all 0.1s ease-in-out !important;
        width: 100% !important;
        height: 80px !important;
        font-size: 1.1rem !important;
    }
    .stButton > button:hover {
        box-shadow: 2px 2px 0px #000000 !important;
        transform: translate(2px, 2px) !important;
    }
    .stButton > button:active {
        box-shadow: 0px 0px 0px #000000 !important;
        transform: translate(4px, 4px) !important;
    }

    /* Special "Run" Button */
    div[data-testid="stVerticalBlock"] > div:nth-child(4) .stButton > button {
        background-color: #000000 !important;
        color: #ffffff !important;
        height: 50px !important;
    }

    /* Input Field */
    .stTextInput input {
        border: 3px solid #000000 !important;
        border-radius: 0px !important;
        padding: 1rem !important;
        font-family: 'Archivo Narrow', sans-serif !important;
        font-weight: 700 !important;
    }

    /* Logs & Memorandum Area */
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

    /* Tool Invocation Chips */
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
    .chip-invoked {
        background-color: #000000;
        color: #ffffff;
    }
    .chip-skipped {
        background-color: #ffffff;
        color: #999999;
        border-color: #cccccc;
        text-decoration: line-through;
    }

    /* Footer Status */
    .system-footer {
        position: fixed;
        bottom: 0;
        left: 0;
        width: 260px; /* matches sidebar width */
        background: #dadada;
        border-top: 3px solid #000000;
        padding: 0.5rem 1rem;
        font-size: 0.7rem;
        font-weight: 700;
        display: flex;
        justify-content: space-between;
    }
    </style>
""", unsafe_allow_html=True)

# --- SIDEBAR NAVIGATION ---
with st.sidebar:
    st.markdown('<div class="sidebar-header">SURVEILLANCE v4.1</div>', unsafe_allow_html=True)

    st.markdown("""
        <div class="nav-item active">👁 Surveillance</div>
        <div class="nav-item">🔍 Investigations</div>
        <div class="nav-item">⚖️ Compliance</div>
        <div class="nav-item">⚙️ LLM Status</div>
        <div class="nav-item">📋 Audit Logs</div>
    """, unsafe_allow_html=True)

    st.markdown("<br><br>", unsafe_allow_html=True)
    st.markdown("""
        <div style="background:#000; color:#fff; padding:1rem; border:3px solid #000;">
            <div style="font-weight:900; font-size:1.1rem;">MONITORING ACTIVE</div>
            <div style="font-size:0.7rem; opacity:0.7;">LLM: GEMINI-FLASH-LATEST</div>
            <hr style="border-color:#333">
            <div style="font-size:0.6rem; font-family:monospace;">CAPS: SCAN, FLAG, REASON</div>
        </div>
    """, unsafe_allow_html=True)

    st.markdown("""
        <div class="system-footer">
            <span>SYSTEM UPTIME</span>
            <span>99.998%</span>
        </div>
    """, unsafe_allow_html=True)

# --- MAIN PAGE TOP BAR ---
st.markdown("""
    <div class="top-bar">
        <h2>🛡️ INCIDENT INVESTIGATION</h2>
        <div style="display:flex; gap:20px; font-weight:900; font-size:0.8rem;">
            <span>DASHBOARD</span>
            <span>ALERTS</span>
            <span>ARCHIVE</span>
            <span>⚙️</span>
        </div>
    </div>
""", unsafe_allow_html=True)

# --- TITLE SECTION ---
st.markdown('<h1 style="font-size:4rem; font-weight:900; margin-bottom:0;">ANTI-MONEY LAUNDERING SURVEILLANCE</h1>', unsafe_allow_html=True)
st.markdown('<p style="font-size:1.2rem; color:#666; margin-bottom:3rem; max-width:900px;">High-density operational dashboard for real-time transaction monitoring, structural analysis, and compliance verification. The system uses LLM reasoning to flag non-linear financial patterns.</p>', unsafe_allow_html=True)

# --- METRIC GRID ---
m1, m2, m3, m4 = st.columns(4)
with m1:
    st.markdown("""<div class="metric-card"><div class="metric-label">Monitored Scope</div><div class="metric-value">5,339 <span style="font-size:1rem; opacity:0.5;">TXNS</span></div><div class="metric-btn">Real-time Stream Active</div></div>""", unsafe_allow_html=True)
with m2:
    st.markdown("""<div class="metric-card"><div class="metric-label">Primary Threshold</div><div class="metric-value">$10,000 <span style="font-size:1rem; opacity:0.5;">CTR</span></div><div class="metric-btn">Regulatory Limit Set</div></div>""", unsafe_allow_html=True)
with m3:
    st.markdown("""<div class="metric-card"><div class="metric-label">Surveillance Engine</div><div class="metric-value">DYNAMIC</div><div class="metric-btn">Neural Analysis On</div></div>""", unsafe_allow_html=True)
with m4:
    st.markdown("""<div style="background:#000; color:#fff;" class="metric-card"><div class="metric-label" style="color:#aaa;">Active State</div><div class="metric-value">READY</div><div style="background:#fff; color:#000;" class="metric-btn">Idle / Awaiting Input</div></div>""", unsafe_allow_html=True)

# --- CONTROL PANEL ---
st.markdown('<div class="section-title">⌨️ Investigation Control Panel</div>', unsafe_allow_html=True)

if "query_text" not in st.session_state:
    st.session_state.query_text = "Find structuring patterns in the last 30 days"

c1, c2, c3, c4 = st.columns(4)
if c1.button("Scan\nStructuring"):
    st.session_state.query_text = "Find structuring patterns in the last 30 days"
if c2.button("10+\nTXNS"):
    st.session_state.query_text = "Which customers made 10+ transactions under $10,000?"
if c3.button("Check\nAccount"):
    st.session_state.query_text = "Is customer ID ACC_SMURF_9003 suspicious?"
if c4.button("Scan\nLayering"):
    st.session_state.query_text = "Detect layering patterns across transactions"

st.markdown("<br>", unsafe_allow_html=True)
user_input = st.text_input(
    label="CUSTOM DIRECTIVE",
    key="query_text",
    placeholder="Enter custom surveillance logic or entity name..."
)

run_button = st.button("RUN INTELLIGENCE ENGINE")

# --- RESULTS SECTION ---
if run_button and user_input:
    with st.spinner("INITIATING AGENT PROTOCOL..."):
        try:
            # Execute LangGraph Autonomous Agent
            response = aml_agent.invoke({"user_query": user_input})

            plan = response.get("plan", {})
            invoked_tools = response.get("invoked_tools", [])
            skipped_tools = response.get("skipped_tools", [])
            raw_explanation = response.get("final_explanation", "No memorandum generated.")
            summary = response.get(
                "execution_summary",
                "[SYSTEM] Initializing surveillance protocol v4.1...\n[SCAN] Parsing transactions...\n[LLM] Identifying non-linear velocity patterns..."
            )

            clean_text = clean_agent_output(raw_explanation)

            st.markdown("---")

            # Handle Human Clarification Early Exit — restored from the original.
            # The agent can come back needing more info instead of a finished memo;
            # showing the log/memo panels for that case would be misleading.
            if response.get("needs_clarification"):
                st.warning("⚠️ HUMAN CLARIFICATION REQUIRED")
                st.markdown(clean_text)
            else:
                st.markdown(
                    f'<div style="font-weight:900; margin-bottom:1rem;">EXECUTION COMPLETED &nbsp;|&nbsp; PLAN INTENT: {plan.get("intent", "UNKNOWN").upper()}</div>',
                    unsafe_allow_html=True
                )

                col_left, col_right = st.columns([2, 1])

                with col_left:
                    st.markdown('<div style="font-weight:900; margin-bottom:1rem;">📟 AGENT REASONING LOG</div>', unsafe_allow_html=True)
                    log_lines = summary.replace("\n", "<br>")
                    st.markdown(f'<div class="log-box">{log_lines}</div>', unsafe_allow_html=True)

                    st.markdown("<br>", unsafe_allow_html=True)
                    st.markdown('<div style="font-weight:900; margin-bottom:0.5rem;">TOOL INVOCATION STATUS</div>', unsafe_allow_html=True)
                    chips_html = ""
                    for tool_name in invoked_tools:
                        chips_html += f'<span class="tool-chip chip-invoked">✓ {tool_name}</span>'
                    for tool_name in skipped_tools:
                        chips_html += f'<span class="tool-chip chip-skipped">✗ {tool_name}</span>'
                    st.markdown(chips_html, unsafe_allow_html=True)

                with col_right:
                    case_id = plan.get("case_id", "AML-2024-0012")
                    st.markdown(f"""
                        <div class="memo-box">
                            <span class="memo-tag">CASE_ID: {case_id}</span>
                            <div style="font-weight:900; font-size:1.5rem; border-bottom:3px solid #000; padding-bottom:1rem; margin-bottom:1rem;">INCIDENT MEMORANDUM</div>
                            <div style="font-weight:900; font-size:0.8rem; margin-bottom:0.5rem; text-transform:uppercase;">Subject Summary</div>
                            <div style="font-size:1.1rem; line-height:1.5;">
                                {clean_text}
                            </div>
                        </div>
                    """, unsafe_allow_html=True)

        except Exception as e:
            st.error(f"ENGINE FAILURE: {str(e)}")