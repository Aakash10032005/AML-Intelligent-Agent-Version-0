import streamlit as st
import pandas as pd
from agent import aml_agent

# Page Configuration
st.set_page_config(
    page_title="AML Compliance & Surveillance Engine",
    page_icon="🏛️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Helper function to extract ONLY clean narrative text from Gemini outputs
def clean_agent_output(explanation):
    """Safely extracts text and handles dictionary or list structures from agent responses."""
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

# Professional Corporate Financial Theme (Clean, High Contrast)
st.markdown("""
    <style>
    /* Global Base */
    .stApp {
        background-color: #0F172A;
        color: #F8FAFC;
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
    }
    
    /* Top Banner Header */
    .header-banner {
        background: linear-gradient(180deg, #1E293B 0%, #0F172A 100%);
        border-bottom: 1px solid #334155;
        padding: 1.5rem 2rem;
        margin-bottom: 2rem;
        border-radius: 8px;
    }
    .header-title {
        font-size: 1.8rem;
        font-weight: 700;
        color: #F8FAFC;
        margin: 0;
        letter-spacing: -0.02em;
    }
    .header-subtitle {
        font-size: 0.9rem;
        color: #94A3B8;
        margin-top: 0.25rem;
    }

    /* Metric Cards */
    .metric-container {
        background-color: #1E293B;
        border: 1px solid #334155;
        border-radius: 8px;
        padding: 1.25rem;
        transition: transform 0.2s ease, border-color 0.2s ease;
    }
    .metric-container:hover {
        border-color: #0284C7;
        transform: translateY(-2px);
    }
    .metric-label {
        font-size: 0.75rem;
        color: #94A3B8;
        text-transform: uppercase;
        font-weight: 600;
        letter-spacing: 0.05em;
    }
    .metric-value {
        font-size: 1.6rem;
        color: #F8FAFC;
        font-weight: 700;
        margin-top: 0.25rem;
    }
    .metric-subtext {
        font-size: 0.8rem;
        color: #38BDF8;
        margin-top: 0.25rem;
        font-weight: 500;
    }

    /* Badges & Tool Chips */
    .tool-chip {
        display: inline-block;
        padding: 0.25rem 0.65rem;
        border-radius: 9999px;
        font-size: 0.8rem;
        font-weight: 600;
        margin-right: 0.4rem;
        margin-bottom: 0.4rem;
    }
    .chip-invoked {
        background-color: #065F46;
        color: #34D399;
        border: 1px solid #059669;
    }
    .chip-skipped {
        background-color: #1E293B;
        color: #64748B;
        border: 1px solid #334155;
        text-decoration: line-through;
    }

    /* Form & Input Customization */
    .stTextInput input {
        background-color: #0F172A !important;
        color: #F8FAFC !important;
        border: 1px solid #475569 !important;
        border-radius: 6px !important;
        padding: 0.75rem 1rem !important;
        font-size: 0.95rem !important;
    }
    .stTextInput input:focus {
        border-color: #0284C7 !important;
        box-shadow: 0 0 0 1px #0284C7 !important;
    }

    /* Primary Action Button */
    .stButton>button {
        background-color: #0284C7 !important;
        color: #FFFFFF !important;
        font-weight: 600 !important;
        border-radius: 6px !important;
        border: none !important;
        padding: 0.75rem 1.5rem !important;
        font-size: 0.95rem !important;
        transition: background-color 0.15s ease !important;
    }
    .stButton>button:hover {
        background-color: #0369A1 !important;
    }

    /* Sidebar Styling Override */
    [data-testid="stSidebar"] {
        background-color: #0B1120;
        border-right: 1px solid #1E293B;
    }
    </style>
""", unsafe_allow_html=True)

# --- SIDEBAR: SYSTEM INFORMATION ---
with st.sidebar:
    st.markdown("### Compliance Intelligence Engine")
    st.caption("BSA / FinCEN Automated Surveillance")
    st.markdown("---")
    
    st.markdown("**System Status:** `ACTIVE`")
    st.markdown("**Core Agent:** Autonomous Dynamic Planner")
    st.markdown("**LLM Provider:** Gemini Flash (`gemini-flash-latest`)")
    st.markdown("**Data Source:** `data/transactions.csv`")
    
    st.markdown("---")
    st.markdown("### Capabilities")
    st.markdown("• **Structuring / Smurfing Analysis**\n• **Layering & Fan-Out Detection**\n• **Single-Entity Risk Profiling**\n• **Deterministic Rule Aggregation**")

# --- MAIN EXECUTIVE HEADER ---
st.markdown("""
    <div class="header-banner">
        <div class="header-title">Anti-Money Laundering Surveillance Dashboard</div>
        <div class="header-subtitle">Dynamic AI Agent for Regulatory Alert Investigation & Incident Memorandum Generation</div>
    </div>
""", unsafe_allow_html=True)

# --- METRIC CARDS ROW ---
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.markdown("""
        <div class="metric-container">
            <div class="metric-label">Monitored Scope</div>
            <div class="metric-value">5,339</div>
            <div class="metric-subtext">Transactions</div>
        </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
        <div class="metric-container">
            <div class="metric-label">Primary Threshold</div>
            <div class="metric-value">$10,000</div>
            <div class="metric-subtext">CTR Limit (BSA)</div>
        </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown("""
        <div class="metric-container">
            <div class="metric-label">Surveillance Engine</div>
            <div class="metric-value">Dynamic</div>
            <div class="metric-subtext">Planner Orchestration</div>
        </div>
    """, unsafe_allow_html=True)

with col4:
    st.markdown("""
        <div class="metric-container">
            <div class="metric-label">Active State</div>
            <div class="metric-value">Ready</div>
            <div class="metric-subtext">Adaptive Routing</div>
        </div>
    """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# --- INVESTIGATION CONTROL CENTER ---
st.markdown("### Investigation Control Panel")

# Benchmark Quick-Run Template Buttons
st.markdown("**Benchmark Query Templates (Click to Auto-Run):**")
c1, c2, c3, c4 = st.columns(4)

preset_query = ""
if c1.button("Scan Structuring (30 Days)", use_container_width=True):
    preset_query = "Find structuring patterns in the last 30 days"
if c2.button("10+ Txns under $10,000", use_container_width=True):
    preset_query = "Which customers made 10+ transactions under $10,000?"
if c3.button("Check ACC_SMURF_9003", use_container_width=True):
    preset_query = "Is customer ID ACC_SMURF_9003 suspicious?"
if c4.button("Scan Layering Typology", use_container_width=True):
    preset_query = "Detect layering patterns across transactions"

# User Input Field
default_value = preset_query if preset_query else "Find structuring patterns in the last 30 days"

user_input = st.text_input(
    label="Enter custom directive, compliance prompt, or specific Account ID:",
    value=default_value,
    placeholder="e.g., Scan for suspicious transactions over $9,000"
)

run_button = st.button("Run Intelligence Engine", use_container_width=True)

# --- EXECUTION & RESULTS DISPLAY ---
if run_button and user_input:
    with st.spinner("Parsing query intent, dynamically constructing execution plan, and invoking tools..."):
        try:
            # Execute LangGraph Autonomous Agent
            response = aml_agent.invoke({"user_query": user_input})
            
            plan = response.get("plan", {})
            invoked_tools = response.get("invoked_tools", [])
            skipped_tools = response.get("skipped_tools", [])
            raw_explanation = response.get("final_explanation", "No output generated.")
            summary = response.get("execution_summary", "")

            clean_text = clean_agent_output(raw_explanation)

            st.markdown("---")

            # Handle Human Clarification Early Exit
            if response.get("needs_clarification"):
                st.warning("⚠️ **Human Clarification Required**")
                st.markdown(clean_text)
            else:
                st.success(f"**Execution Completed** | Plan Intent: **`{plan.get('intent', 'UNKNOWN').upper()}`**")

                # Prominent Agent Reasoning Expander Header
                with st.expander("🧩 **Agent Reasoning & Adaptive Execution Plan**", expanded=True):
                    st.code(summary, language="text")
                    
                    st.markdown("**Tool Invocation Status:**")
                    chips_html = ""
                    for tool_name in invoked_tools:
                        chips_html += f'<span class="tool-chip chip-invoked">✓ {tool_name}</span>'
                    for tool_name in skipped_tools:
                        chips_html += f'<span class="tool-chip chip-skipped">✗ {tool_name} (skipped)</span>'
                    st.markdown(chips_html, unsafe_allow_html=True)

                st.markdown("<br>", unsafe_allow_html=True)
                st.markdown("### Incident Memorandum")
                st.markdown(clean_text)

        except Exception as e:
            st.error(f"Execution Error: {str(e)}")