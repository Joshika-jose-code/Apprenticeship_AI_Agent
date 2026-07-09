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

# Tool A: Web Search via Tavily (Optimized for thorough results)
web_search_tool = TavilySearch(max_results=5)

# Tool B: Custom File Automation Tool
@tool
def save_research_report(content: str, filename: str = "apprenticeship_listings.md") -> str:
    """Saves the final, compiled apprenticeship listings into a local file. 
    Use this tool ONLY when you have completed all necessary research."""
    try:
        with open(filename, "w", encoding="utf-8") as f:
            f.write(content)
        return f"Success: Report successfully saved to local file '{filename}'."
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
    """The brain node. Instructs Gemini on the exact time constraints and fields needed."""
    messages = state['messages']
    
    # Calculate current time parameters dynamically
    current_time = datetime.now()
    cutoff_time = current_time - timedelta(hours=48)
    
    time_context = (
        f"Current Time: {current_time.strftime('%Y-%m-%d %H:%M')}. "
        f"You must strictly find listings posted or updated AFTER {cutoff_time.strftime('%Y-%m-%d %H:%M')} (within the last 48 hours)."
    )
    
    system_instruction = (
        "You are an elite talent acquisition and web-scraping agent.\n"
        f"{time_context}\n\n"
        "Your task is to search the web for apprenticeship opportunities for freshers and recent graduates "
        "in Computer Science, Data Science, and Analytics. \n\n"
        "CRITICAL RULES:\n"
        "1. Filter for listings posted/active within the last 48 hours in India.\n"
        "2. Include the direct, actionable URL/link to the application page or job portal for every single listing.\n"
        "3. Once you have a valid list, call the 'save_research_report' tool to write a clean markdown table to a file.\n"
        "4. Format the markdown output with columns: Company, Role, Field, Date Posted, Application Link."
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
        "Find newly posted apprenticeship opportunities (within the past 48 hours) for freshers and "
        "recent graduates in computer science, data science, and analytics in India. Compile them with "
        "valid source application links into a local file."
    )
    print(f"Starting Agentic Search Workflow...\n")
    
    inputs = {"messages": [("user", user_prompt)]}
    for output in app.stream(inputs, stream_mode="values"):
        last_msg = output["messages"][-1]
        print(f"--- {last_msg.type.upper()} ({getattr(last_msg, 'name', 'Model')}) ---")
        print(f"{last_msg.content}\n")
