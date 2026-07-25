import os
import json
from typing import TypedDict, List, Dict, Any, Optional
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import StateGraph, END

# Import analytical tools from tools.py
from tools import (
    run_eda_tool,
    feature_engineering_tool,
    detect_structuring_tool,
    detect_layering_tool,
    risk_classification_tool
)

load_dotenv(override=True)
api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("google_api_key")

if not api_key:
    raise ValueError("GOOGLE_API_KEY not found in environment variables. Check your .env file!")


# 1. State Definition
class AgentState(TypedDict):
    user_query: str
    plan: Dict[str, Any]
    needs_clarification: bool
    clarifying_question: Optional[str]
    tool_outputs: List[str]
    invoked_tools: List[str]
    skipped_tools: List[str]
    final_explanation: str
    execution_summary: str


# Helper to initialize LLM
def get_llm():
    return ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",   # pinned, stable — NOT "gemini-flash-latest"
        temperature=0,
        google_api_key=api_key
    )


# 2. Node: Dynamic Query Planner
def parse_query_node(state: AgentState):
    """
    Parses the user query into a structured execution plan JSON object.
    Determines intent, target pattern, filters, required modules, tools to invoke, and skip reasons.
    """
    llm = get_llm()

    prompt = f"""
    You are an expert Anti-Money Laundering (AML) Compliance Agent and Planner.
    Analyze the user's natural language query and construct a dynamic execution plan.

    Respond ONLY with a valid JSON object strictly matching this schema:
    {{
      "intent": "eda | pattern_detection | single_entity | aggregation_rule | comparison | insight_query",
      "target_pattern": "structuring | smurfing | layering | none",
      "filters": {{
        "date_range": {{"start": "YYYY-MM-DD or null", "end": "YYYY-MM-DD or null"}},
        "account_id": "string or null",
        "transaction_type": "string or null",
        "amount_min": "number or null",
        "amount_max": "number or null",
        "min_transaction_count": "number or null"
      }},
      "requires_eda": boolean,
      "requires_feature_engineering": boolean,
      "requires_anomaly_detection": boolean,
      "requires_risk_classification": boolean,
      "tools_to_invoke": ["ordered list of tool names from: eda, feature_engineering, pattern_detection, risk_classification"],
      "needs_clarification": boolean,
      "clarifying_question": "string or null (ask clarifying question if query is completely ambiguous or lacks necessary context)",
      "reasoning": "one sentence explaining why these specific tools were chosen and why others were skipped"
    }}

    IMPORTANT: For aggregation_rule intent queries that state an explicit count/threshold
    (e.g. "10+ transactions under $10,000"), you MUST populate both "min_transaction_count"
    and "amount_max" in filters so the feature_engineering tool can answer the query directly
    without needing ML anomaly detection.

    FEW-SHOT EXAMPLES:
    Example 1: "Find structuring patterns in the last 30 days"
    -> intent: "pattern_detection", target_pattern: "structuring", filters: {{date_range: {{start: "2025-06-25", end: "2025-07-25"}}}}, requires_eda: false, requires_feature_engineering: true, requires_anomaly_detection: true, requires_risk_classification: true, tools_to_invoke: ["feature_engineering", "pattern_detection", "risk_classification"], needs_clarification: false, reasoning: "Targeted pattern detection query skips broad EDA to focus on feature extraction, structuring detection, and risk scoring."

    Example 2: "Which customers made 10+ transactions under $10,000?"
    -> intent: "aggregation_rule", target_pattern: "none", filters: {{amount_max: 10000, min_transaction_count: 10}}, requires_eda: false, requires_feature_engineering: true, requires_anomaly_detection: false, requires_risk_classification: false, tools_to_invoke: ["feature_engineering"], needs_clarification: false, reasoning: "Deterministic threshold aggregation query requires feature engineering to count sub-10k transfers, skipping ML anomaly detection."

    Example 3: "Is customer ID ACC_SMURF_9003 suspicious?"
    -> intent: "single_entity", target_pattern: "none", filters: {{account_id: "ACC_SMURF_9003"}}, requires_eda: false, requires_feature_engineering: true, requires_anomaly_detection: true, requires_risk_classification: true, tools_to_invoke: ["feature_engineering", "risk_classification"], needs_clarification: false, reasoning: "Single-entity lookup extracts targeted account features and evaluates hybrid risk classification."

    Example 4: "Analyze transactions"
    -> needs_clarification: true, clarifying_question: "Could you please specify whether you want a broad baseline EDA summary, a scan for structuring/layering patterns, or a risk score for a specific Account ID?"

    User Query: "{state['user_query']}"
    """

    try:
        response = llm.invoke(prompt)
        content = response.content.strip()

        # Clean JSON markdown fences if present
        if content.startswith("```"):
            content = content.split("```")[1]
            if content.startswith("json"):
                content = content[4:]
        content = content.strip()

        plan = json.loads(content)
    except Exception:
        # Fallback plan if JSON parsing fails
        plan = {
            "intent": "eda",
            "target_pattern": "none",
            "filters": {},
            "requires_eda": True,
            "requires_feature_engineering": False,
            "requires_anomaly_detection": False,
            "requires_risk_classification": False,
            "tools_to_invoke": ["eda"],
            "needs_clarification": False,
            "clarifying_question": None,
            "reasoning": "Fallback routing to baseline EDA due to unparseable response."
        }

    all_possible = ["eda", "feature_engineering", "pattern_detection", "risk_classification"]
    invoked = plan.get("tools_to_invoke", [])
    skipped = [t for t in all_possible if t not in invoked]

    return {
        "plan": plan,
        "needs_clarification": plan.get("needs_clarification", False),
        "clarifying_question": plan.get("clarifying_question"),
        "invoked_tools": invoked,
        "skipped_tools": skipped,
        "tool_outputs": []
    }


# 3. Router Edge Function
def route_after_parse(state: AgentState):
    """Routes execution based on whether human clarification is needed or tool execution should start."""
    if state.get("needs_clarification", False):
        return "human_clarification"
    return "execute_tools_pipeline"


# 4. Node: Human-in-the-loop Clarification
def clarification_node(state: AgentState):
    question = state.get("clarifying_question") or "Could you please clarify your request?"
    explanation = f"### ⚠️ Clarification Needed\n\n{question}"
    return {
        "final_explanation": explanation,
        "execution_summary": f"**QUERY**: \"{state['user_query']}\"\n**STATUS**: Awaiting Clarification (Ambiguous Input)"
    }


# 5. Pipeline Node: Execute Planned Tools
def execute_tools_node(state: AgentState):
    plan = state.get("plan", {})
    tools_to_run = plan.get("tools_to_invoke", [])
    filters = plan.get("filters", {}) if isinstance(plan.get("filters"), dict) else {}
    target_pattern = plan.get("target_pattern", "none")

    acc_id = filters.get("account_id")
    date_range = filters.get("date_range", {}) if isinstance(filters.get("date_range"), dict) else {}
    date_start = date_range.get("start")
    date_end = date_range.get("end")
    amount_min = filters.get("amount_min")
    amount_max = filters.get("amount_max")
    min_tx_count = filters.get("min_transaction_count")
    tx_type = filters.get("transaction_type")

    outputs = []

    # Modular execution of invoked tools, with planner filters threaded through
    if "eda" in tools_to_run:
        outputs.append(
            "--- EDA MODULE OUTPUT ---\n" +
            run_eda_tool.invoke({"query": "all", "date_start": date_start, "date_end": date_end})
        )

    if "feature_engineering" in tools_to_run:
        param = acc_id if acc_id else "all"
        outputs.append(
            "--- FEATURE ENGINEERING OUTPUT ---\n" +
            feature_engineering_tool.invoke({
                "account_id": param,
                "date_start": date_start,
                "date_end": date_end,
                "amount_min": amount_min,
                "amount_max": amount_max,
                "transaction_type": tx_type,
                "min_transaction_count": min_tx_count,
            })
        )

    if "pattern_detection" in tools_to_run:
        if target_pattern == "layering":
            outputs.append(
                "--- LAYERING PATTERN DETECTION OUTPUT ---\n" +
                detect_layering_tool.invoke({
                    "query": "all", "date_start": date_start, "date_end": date_end
                })
            )
        else:
            outputs.append(
                "--- STRUCTURING PATTERN DETECTION OUTPUT ---\n" +
                detect_structuring_tool.invoke({
                    "date_start": date_start, "date_end": date_end
                })
            )

    if "risk_classification" in tools_to_run:
        param = acc_id if acc_id else "all"
        outputs.append(
            "--- HYBRID RISK CLASSIFICATION OUTPUT ---\n" +
            risk_classification_tool.invoke({
                "account_id": param,
                "target_pattern": target_pattern,
                "date_start": date_start,
                "date_end": date_end,
            })
        )

    return {"tool_outputs": outputs}


# 6. Node: Query-Aware Compliance Memorandum Generator
def explanation_node(state: AgentState):
    plan = state.get("plan", {})
    tools_invoked = state.get("invoked_tools", [])
    tools_skipped = state.get("skipped_tools", [])
    outputs = "\n\n".join(state.get("tool_outputs", []))

    llm = get_llm()

    prompt = f"""
    You are a Senior Bank AML Compliance Officer preparing a formal Incident Memorandum.
    Synthesize the original query, dynamic execution plan, and tool analytical outputs into a professional regulatory memorandum.

    Original Query: "{state['user_query']}"
    Planner Intent: {plan.get('intent')}
    Target Typology: {plan.get('target_pattern')}
    Invoked Tools: {', '.join(tools_invoked)}
    Raw Tool Outputs:
    {outputs}

    REQUIREMENTS:
    1. Tie every narrative observation back to the original user query and identified target pattern.
    2. Outline specific account flags, amounts, and statistical anomaly indicators.
    3. Conclude with an explicit, auditable Escalation Recommendation based on risk severity:
       - HIGH RISK -> Recommend immediate SAR (Suspicious Activity Report) filing with FinCEN.
       - MEDIUM RISK -> Recommend placing under 30-day Enhanced Due Diligence (EDD) monitoring.
       - LOW RISK -> Recommend standard monitoring with no immediate escalation.
    4. Format using clean GitHub-style Markdown with clear headings and bullet points.
    """

    try:
        res = llm.invoke(prompt)
        explanation_text = res.content
        if isinstance(explanation_text, list):
            text_content = ""
            for block in explanation_text:
                if isinstance(block, dict) and "text" in block:
                    text_content += block["text"]
            explanation_text = text_content if text_content else str(explanation_text)
    except Exception as e:
        explanation_text = (
            f"### AML Incident Analysis Summary\n\n"
            f"**Query Analyzed**: {state['user_query']}\n\n"
            f"**Tool Output Analysis**:\n{outputs}\n\n"
            f"**Escalation Recommendation**: Review high-risk accounts and assess for SAR filing with FinCEN."
        )

    # Format Query-Aware Execution Summary Header
    summary_header = format_execution_summary(state)

    return {
        "final_explanation": str(explanation_text),
        "execution_summary": summary_header
    }


# Helper: Build Query-Aware Execution Summary
def format_execution_summary(state: AgentState) -> str:
    plan = state.get("plan", {})
    intent = plan.get("intent", "Unknown")
    pattern = plan.get("target_pattern", "none")
    filters = plan.get("filters", {})
    invoked = state.get("invoked_tools", [])
    skipped = state.get("skipped_tools", [])
    reasoning = plan.get("reasoning", "N/A")

    filters_str = json.dumps(filters) if filters else "None"
    invoked_str = " -> ".join(invoked) if invoked else "None"
    skipped_str = ", ".join(skipped) if skipped else "None"

    return f"""```text
QUERY: "{state['user_query']}"
DETECTED INTENT: {intent} (target_pattern: {pattern})
FILTERS APPLIED: {filters_str}
TOOLS INVOKED: {invoked_str}
TOOLS SKIPPED: {skipped_str}
PLANNER REASONING: {reasoning}
```"""


# --- BUILD LANGGRAPH WORKFLOW ---
workflow = StateGraph(AgentState)

# Add Nodes
workflow.add_node("parse_query", parse_query_node)
workflow.add_node("human_clarification", clarification_node)
workflow.add_node("execute_tools_pipeline", execute_tools_node)
workflow.add_node("generate_explanation", explanation_node)

# Set Entry Point
workflow.set_entry_point("parse_query")

# Add Conditional Edges
workflow.add_conditional_edges(
    "parse_query",
    route_after_parse,
    {
        "human_clarification": "human_clarification",
        "execute_tools_pipeline": "execute_tools_pipeline"
    }
)

# Connect edges
workflow.add_edge("execute_tools_pipeline", "generate_explanation")
workflow.add_edge("human_clarification", END)
workflow.add_edge("generate_explanation", END)

# Compile Executable Graph
aml_agent = workflow.compile()

if __name__ == "__main__":
    import time
    test_queries = [
        "Find structuring patterns in the last 30 days",
        "Which customers made 10+ transactions under $10,000?",
        "Is customer ID ACC_SMURF_9003 suspicious?"
    ]

    for query in test_queries:
        print(f"\n" + "=" * 60)
        print(f"TESTING QUERY: {query}")
        print("=" * 60)

        result = aml_agent.invoke({"user_query": query})
        print(result['execution_summary'])
        print("\n--- INCIDENT MEMORANDUM ---")
        print(result['final_explanation'])
        time.sleep(2)