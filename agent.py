import os
from datetime import datetime, timedelta
from typing import Annotated, Literal
from typing_extensions import TypedDict

from langchain_core.tools import tool
from langchain_core.messages import BaseMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_tavily import TavilySearch
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode

# ==========================================
# 1. DEFINE EXTERNAL TOOLS
# ==========================================

# Tool A: Web Search via Tavily (higher result count + deeper search)
web_search_tool = TavilySearch(max_results=10, search_depth="advanced")

# Tool B: Custom File Automation Tool
MIN_LISTINGS = 20
OUTPUT_FILENAME = "apprenticeship_listings.md"

@tool
def save_research_report(content: str) -> str:
    """Saves the final, compiled apprenticeship listings into a local file.
    Use this tool ONLY when you have compiled at least 20 distinct listings.
    Do not call this tool with fewer than 20 listings - keep searching instead."""
    # Rough row count: count markdown table rows, minus header + separator rows
    row_count = max(content.count("\n|") - 2, 0)

    if row_count < MIN_LISTINGS:
        return (
            f"Error: Only ~{row_count} listings found, but at least {MIN_LISTINGS} are required. "
            "Do NOT save yet. Run more searches with different queries (different job portals, "
            "role titles, or a wider time window) and try saving again once you have enough."
        )

    try:
        with open(OUTPUT_FILENAME, "w", encoding="utf-8") as f:
            f.write(content)
        return f"Success: Report with ~{row_count} listings saved to local file '{OUTPUT_FILENAME}'."
    except Exception as e:
        return f"Error saving file: {str(e)}"

tools = [web_search_tool, save_research_report]
tool_node = ToolNode(tools)

# ==========================================
# 2. CONFIGURE THE LLM BRAIN
# ==========================================
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.1)
llm_with_tools = llm.bind_tools(tools)

# ==========================================
# 3. DEFINE THE AGENT STATE & NODES
# ==========================================
class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]

def call_model(state: AgentState):
    """The brain node. Instructs Gemini on time constraints, count requirements, and search strategy."""
    messages = state['messages']

    current_time = datetime.now()
    cutoff_time = current_time - timedelta(hours=48)

    time_context = (
        f"Current Time: {current_time.strftime('%Y-%m-%d %H:%M')}. "
        f"Prefer listings posted or updated AFTER {cutoff_time.strftime('%Y-%m-%d %H:%M')} "
        f"(within the last 48 hours)."
    )

    system_instruction = (
        "You are an elite talent acquisition and web-scraping agent.\n"
        f"{time_context}\n\n"
        "Your task is to search the web for apprenticeship opportunities for freshers and recent "
        "graduates in Computer Science, Data Science, and Analytics in India.\n\n"
        "CRITICAL RULES:\n"
        f"1. You MUST find and compile AT LEAST {MIN_LISTINGS} distinct listings before calling "
        "save_research_report. If your searches so far have returned fewer than that, you MUST "
        "continue searching with new, more specific queries — do not stop early.\n"
        "2. Vary your search queries: try different job portals by name (e.g. 'site:internshala.com', "
        "'site:linkedin.com/jobs', 'site:naukri.com'), different role titles (data analyst apprentice, "
        "ML engineer trainee, software apprentice, graduate trainee, associate analyst), and different "
        "companies.\n"
        "3. Prefer listings posted/active within the last 48 hours, but if that filter leaves you under "
        f"{MIN_LISTINGS} results, relax it to 'within the last 2 weeks' rather than reporting too few. "
        "Note in the Date Posted column when a listing's date is approximate/inferred.\n"
        "4. Include the direct, actionable URL/link to the application page or job portal for every "
        "listing. Do not invent or guess URLs.\n"
        f"5. Only call 'save_research_report' once you have compiled at least {MIN_LISTINGS} listings. "
        "If the tool returns an error saying you have too few, keep searching and try again.\n"
        "6. Format the markdown output with columns: Company, Role, Field, Date Posted, Application Link."
    )

    response = llm_with_tools.invoke([{"role": "system", "content": system_instruction}] + messages)
    return {"messages": [response]}

def route_next_node(state: AgentState) -> Literal["tools", "__end__"]:
    last_message = state['messages'][-1]
    if last_message.tool_calls:
        return "tools"
    return END

# ==========================================
# 4. ORCHESTRATE THE AGENT GRAPH
# ==========================================
workflow = StateGraph(AgentState)
workflow.add_node("agent", call_model)
workflow.add_node("tools", tool_node)

workflow.add_edge(START, "agent")
workflow.add_conditional_edges("agent", route_next_node)
workflow.add_edge("tools", "agent")

app = workflow.compile()

# ==========================================
# 5. EXECUTE THE AGENT WITH YOUR CUSTOM PROMPT
# ==========================================
if __name__ == "__main__":
    user_prompt = (
        f"Find newly posted apprenticeship opportunities for freshers and recent graduates in "
        f"computer science, data science, and analytics in India. You must compile at least "
        f"{MIN_LISTINGS} listings with valid source application links into a local file."
    )
    print(f"Starting Agentic Search Workflow...\n")

    inputs = {"messages": [("user", user_prompt)]}
    config = {"recursion_limit": 50}
    for output in app.stream(inputs, stream_mode="values", config=config):
        last_msg = output["messages"][-1]
        print(f"--- {last_msg.type.upper()} ({getattr(last_msg, 'name', 'Model')}) ---")
        print(f"{last_msg.content}\n")
