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
        model="gemini-2.5-flash",
        temperature=0,
        google_api_key=api_key
    )


# 2. Node: Dynamic Query Planner with Accurate Account & Intent Parsing
def parse_query_node(state: AgentState):
    llm = get_llm()
    query_lower = state['user_query'].lower()

    forced_tools = None
    forced_intent = None
    forced_pattern = "none"

    if any(k in query_lower for k in ["structuring", "smurf", "structur"]):
        forced_tools = ["feature_engineering", "pattern_detection", "risk_classification"]
        forced_intent = "pattern_detection"
        forced_pattern = "structuring"
    elif any(k in query_lower for k in ["layering", "layer"]):
        forced_tools = ["feature_engineering", "pattern_detection", "risk_classification"]
        forced_intent = "pattern_detection"
        forced_pattern = "layering"
    elif any(k in query_lower for k in ["account", "acc_", "customer", "audit"]):
        forced_tools = ["feature_engineering", "risk_classification"]
        forced_intent = "single_entity"
    elif any(k in query_lower for k in ["volume", "summary", "eda", "overview", "transaction type"]):
        forced_tools = ["eda"]
        forced_intent = "eda"

    prompt = f"""
    You are an expert Anti-Money Laundering (AML) Compliance Agent and Planner.
    Analyze the user's natural language query and extract filters (especially account IDs like ACC_3127 if present).

    Respond ONLY with a valid JSON block containing these exact keys:
    {{
      "intent": "eda | pattern_detection | single_entity | aggregation_rule | comparison | insight_query",
      "target_pattern": "structuring | smurfing | layering | none",
      "filters": {{
        "date_range": {{"start": null, "end": null}},
        "account_id": "Extract exact account ID if present e.g. ACC_3127 or null",
        "transaction_type": null,
        "amount_min": null,
        "amount_max": null,
        "min_transaction_count": null
      }},
      "tools_to_invoke": ["feature_engineering", "risk_classification"],
      "needs_clarification": false,
      "clarifying_question": null,
      "reasoning": "string explaining choices"
    }}

    User Query: "{state['user_query']}"
    """

    content = ""
    plan = {}

    try:
        response = llm.invoke(prompt)
        content = response.content.strip()
        if "```json" in content:
            content = content.split("```json")[1].split("```")[0].strip()
        elif "```" in content:
            content = content.split("```")[1].split("```")[0].strip()
        plan = json.loads(content)
    except Exception as e:
        print(f"Parser Error: {e}")

    # Fallback / Overrides
    if forced_tools:
        plan["tools_to_invoke"] = forced_tools
        plan["intent"] = forced_intent
        plan["target_pattern"] = forced_pattern
    elif not plan.get("tools_to_invoke"):
        plan["tools_to_invoke"] = ["eda"]
        plan["intent"] = "eda"

    # Ensure account_id is dynamically parsed from query text if LLM missed it
    if "filters" not in plan or not isinstance(plan["filters"], dict):
        plan["filters"] = {}
    
    if not plan["filters"].get("account_id"):
        for word in state['user_query'].split():
            if word.upper().startswith("ACC_"):
                plan["filters"]["account_id"] = word.strip(".,!?")
                break

    all_possible = ["eda", "feature_engineering", "pattern_detection", "risk_classification"]
    invoked = plan.get("tools_to_invoke", ["eda"])
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
    if state.get("needs_clarification", False):
        return "human_clarification"
    return "execute_tools_pipeline"


# 4. Node: Human-in-the-loop Clarification
def clarification_node(state: AgentState):
    question = state.get("clarifying_question") or "Could you please clarify your request?"
    return {
        "final_explanation": f"### ⚠️ Clarification Needed\n\n{question}",
        "execution_summary": f"**QUERY**: \"{state['user_query']}\"\n**STATUS**: Awaiting Clarification"
    }


# 5. Pipeline Node: Execute Planned Tools Dynamically
def execute_tools_node(state: AgentState):
    plan = state.get("plan", {})
    tools_to_run = plan.get("tools_to_invoke", [])
    filters = plan.get("filters", {}) if isinstance(plan.get("filters"), dict) else {}
    target_pattern = plan.get("target_pattern", "none")

    acc_id = filters.get("account_id")
    date_range = filters.get("date_range", {}) if isinstance(filters.get("date_range"), dict) else {}
    
    date_start = date_range.get("start") if date_range and date_range.get("start") else "2023-01-01"
    date_end = date_range.get("end") if date_range and date_range.get("end") else "2025-12-31"
    
    if date_start in ["null", None]: date_start = "2023-01-01"
    if date_end in ["null", None]: date_end = "2025-12-31"

    amount_min = filters.get("amount_min")
    amount_max = filters.get("amount_max")
    min_tx_count = filters.get("min_transaction_count")
    tx_type = filters.get("transaction_type")

    outputs = []

    if "eda" in tools_to_run:
        outputs.append(
            "--- EDA MODULE OUTPUT ---\n" +
            run_eda_tool.invoke({"query": state['user_query'], "date_start": str(date_start), "date_end": str(date_end)})
        )

    if "feature_engineering" in tools_to_run:
        param = acc_id if acc_id and acc_id != "null" else "all"
        fe_args = {"account_id": param, "date_start": str(date_start), "date_end": str(date_end)}
        if amount_min not in [None, "null"]: fe_args["amount_min"] = float(amount_min)
        if amount_max not in [None, "null"]: fe_args["amount_max"] = float(amount_max)
        if tx_type not in [None, "null"]: fe_args["transaction_type"] = str(tx_type)
        if min_tx_count not in [None, "null"]: fe_args["min_transaction_count"] = int(min_tx_count)

        outputs.append(
            "--- FEATURE ENGINEERING OUTPUT ---\n" +
            feature_engineering_tool.invoke(fe_args)
        )

    if "pattern_detection" in tools_to_run:
        if target_pattern == "layering":
            outputs.append(
                "--- LAYERING PATTERN DETECTION OUTPUT ---\n" +
                detect_layering_tool.invoke({"query": state['user_query'], "date_start": str(date_start), "date_end": str(date_end)})
            )
        else:
            outputs.append(
                "--- STRUCTURING PATTERN DETECTION OUTPUT ---\n" +
                detect_structuring_tool.invoke({"date_start": str(date_start), "date_end": str(date_end)})
            )

    if "risk_classification" in tools_to_run:
        param = acc_id if acc_id and acc_id != "null" else "all"
        outputs.append(
            "--- HYBRID RISK CLASSIFICATION OUTPUT ---\n" +
            risk_classification_tool.invoke({
                "account_id": param,
                "target_pattern": str(target_pattern),
                "date_start": str(date_start),
                "date_end": str(date_end),
            })
        )

    return {"tool_outputs": outputs}


# 6. Node: Query-Aware Compliance Memorandum Generator with Strict Markdown Sanitization
def explanation_node(state: AgentState):
    plan = state.get("plan", {})
    tools_invoked = state.get("invoked_tools", [])
    outputs = "\n\n".join(state.get("tool_outputs", []))
    llm = get_llm()

    prompt = f"""
    You are a Senior Bank AML Compliance Officer preparing a formal Incident Memorandum.
    Synthesize the raw analytical tool outputs below directly answering the user's specific query.

    Original Query: "{state['user_query']}"
    Planner Intent: {plan.get('intent')}
    Target Typology: {plan.get('target_pattern')}
    Invoked Tools: {', '.join(tools_invoked)}
    
    Raw Tool Outputs to Synthesize:
    {outputs}

    STRICT FORMATTING RULES:
    1. NEVER use raw math/LaTeX underscores for account IDs. Enclose all account IDs in standard code blocks (e.g., `ACC_3127` or `ACC_SMURF_9801`) so Markdown does not break.
    2. Completely strip out raw headers like 'FEATURE ENGINEERING OUTPUT' or 'HYBRID RISK CLASSIFICATION OUTPUT'. Replace them with clean executive sections.
    3. Write a concise **Plain English Summary** directly addressing the query, followed by bullet points of findings and an explicit **Escalation Recommendation**.
    """

    try:
        res = llm.invoke(prompt)
        explanation_text = res.content
        if isinstance(explanation_text, list):
            explanation_text = "".join([block.get("text", "") for block in explanation_text if isinstance(block, dict)])
    except Exception:
        explanation_text = f"### AML Incident Summary\n\n{outputs}"

    summary_header = f"""```text
QUERY: "{state['user_query']}"
DETECTED INTENT: {plan.get('intent', 'Unknown')} (target_pattern: {plan.get('target_pattern', 'none')})
FILTERS APPLIED: {plan.get('filters', {})}
TOOLS INVOKED: {" -> ".join(state.get('invoked_tools', []))}
TOOLS SKIPPED: {', '.join(state.get('skipped_tools', []))}
PLANNER REASONING: {plan.get('reasoning', 'N/A')}
```"""

    return {"final_explanation": str(explanation_text), "execution_summary": summary_header}


# --- BUILD LANGGRAPH WORKFLOW ---
workflow = StateGraph(AgentState)
workflow.add_node("parse_query", parse_query_node)
workflow.add_node("human_clarification", clarification_node)
workflow.add_node("execute_tools_pipeline", execute_tools_node)
workflow.add_node("generate_explanation", explanation_node)

workflow.set_entry_point("parse_query")
workflow.add_conditional_edges("parse_query", route_after_parse, {
    "human_clarification": "human_clarification",
    "execute_tools_pipeline": "execute_tools_pipeline"
})
workflow.add_edge("execute_tools_pipeline", "generate_explanation")
workflow.add_edge("human_clarification", END)
workflow.add_edge("generate_explanation", END)

aml_agent = workflow.compile()