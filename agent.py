import os
from typing import TypedDict
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import StateGraph, END

# Import the analytical tools built in tools.py
from tools import run_eda_tool, detect_structuring_tool, risk_scoring_tool

# Force dotenv to load the API key from .env file
load_dotenv(override=True)
api_key = os.getenv("GOOGLE_API_KEY")

if not api_key:
    raise ValueError("GOOGLE_API_KEY not found in environment variables. Check your .env file!")

# 1. State Definition
class AgentState(TypedDict):
    user_query: str
    identified_intent: str
    tool_output: str
    final_explanation: str

# 2. Node: Parse Intent
def parse_intent_node(state: AgentState):
    """Uses Gemini to determine which analytical tool is required based on user intent."""
    llm = ChatGoogleGenerativeAI(
        model="gemini-flash-latest", 
        temperature=0, 
        google_api_key=api_key
    )
    
    prompt = f"""
    Analyze the following user query and determine the exact intent.
    Return ONLY ONE of these three exact strings:
    - 'eda' (if they want broad summaries, data exploration, or baseline statistics)
    - 'structuring' (if they want to find smurfing, structuring, patterns, or evasion)
    - 'single_entity' (if they ask about a specific customer ID or account)

    User Query: {state['user_query']}
    """
    response = llm.invoke(prompt)
    intent = str(response.content).strip().lower()
    
    # Fallback safety routing logic
    if 'single_entity' in intent: 
        intent = 'single_entity'
    elif 'structuring' in intent: 
        intent = 'structuring'
    else: 
        intent = 'eda'
        
    return {"identified_intent": intent}

# 3. Conditional Routing Function
def route_to_tool(state: AgentState):
    """Tells LangGraph which tool node to execute based on identified intent."""
    return state['identified_intent']

# 4. Nodes: Tool Execution
def execute_eda_node(state: AgentState):
    output = run_eda_tool.invoke({"query": "all"})
    return {"tool_output": output}

def execute_structuring_node(state: AgentState):
    output = detect_structuring_tool.invoke({"timeframe": "30 days"})
    return {"tool_output": output}

def execute_single_entity_node(state: AgentState):
    """Extracts entity identifier using Gemini and runs the risk scoring engine."""
    llm = ChatGoogleGenerativeAI(
        model="gemini-flash-latest", 
        temperature=0, 
        google_api_key=api_key
    )
    id_prompt = f"Extract only the exact Account ID from this query. Example output: 'ACC_1234' or 'ACC_SMURF_9999'. Query: {state['user_query']}"
    acc_id = str(llm.invoke(id_prompt).content).strip()
    
    output = risk_scoring_tool.invoke({"account_id": acc_id})
    return {"tool_output": output}

# 5. Node: Explanation and Escalation
def explanation_node(state: AgentState):
    """Translates tool output into clear AML compliance narratives and recommendations."""
    llm = ChatGoogleGenerativeAI(
        model="gemini-flash-latest", 
        temperature=0, 
        google_api_key=api_key
    )
    prompt = f"""
    You are a bank compliance officer. Based on the user's original query and the tool's raw output, 
    generate a human-readable explanation of why a flag occurred and recommend a basic escalation action (monitor, review, or report).
    
    Original Query: {state['user_query']}
    Tool Output: {state['tool_output']}
    
    Provide the response using professional formatting and bullet points.
    """
    response = llm.invoke(prompt)
    return {"final_explanation": str(response.content)}

# --- BUILD THE LANGGRAPH WORKFLOW ---
workflow = StateGraph(AgentState)

# Add all process nodes
workflow.add_node("parse_intent", parse_intent_node)
workflow.add_node("execute_eda", execute_eda_node)
workflow.add_node("execute_structuring", execute_structuring_node)
workflow.add_node("execute_single_entity", execute_single_entity_node)
workflow.add_node("generate_explanation", explanation_node)

# Set the start node
workflow.set_entry_point("parse_intent")

# Add conditional routing edges
workflow.add_conditional_edges(
    "parse_intent",
    route_to_tool,
    {
        "eda": "execute_eda",
        "structuring": "execute_structuring",
        "single_entity": "execute_single_entity"
    }
)

# Connect execution nodes to the final explanation generator
workflow.add_edge("execute_eda", "generate_explanation")
workflow.add_edge("execute_structuring", "generate_explanation")
workflow.add_edge("execute_single_entity", "generate_explanation")

# Terminate execution graph
workflow.add_edge("generate_explanation", END)

# Compile agent executable
aml_agent = workflow.compile()

if __name__ == "__main__":
    test_query = "Find structuring patterns in the last 30 days"
    print(f"Testing Query: {test_query}\n" + "-"*40)
    
    # Execute graph
    result = aml_agent.invoke({"user_query": test_query})
    
    print(f"Agent's Routing Decision: Routed to {result['identified_intent']} tool.")
    print("\n--- FINAL COMPLIANCE EXPLANATION ---")
    print(result['final_explanation'])