from state.state import TravelState
from tools.tavily_tool import tavily_search
from mcp_client import tavily_mcp_search
from langchain_core.messages import AIMessage
import asyncio

def hotel_agent(state: TravelState):
    query = f"Best hotels for {state['user_query']}"
    hotel_results = asyncio.run(
        tavily_mcp_search(query)
    )

    return {
        "hotel_results": hotel_results,
        "messages": [
            AIMessage(content="Hotel information fetched")
        ],
        "llm_calls": state.get("llm_calls", 0) + 1
    }