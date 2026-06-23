import os
import asyncio

from dotenv import load_dotenv
from langchain_mcp_adapters.client import MultiServerMCPClient
load_dotenv()


TAVILY_API_KEY =    os.getenv("TAVILY_API_KEY")

## creating a MCP client
client = MultiServerMCPClient(

    {
  "tavily_search": {
    "transport": "streamable_http",
    "url": f"https://mcp.tavily.com/mcp/?tavilyApiKey={TAVILY_API_KEY}"
  }
    },



)



# async def func_tools():
#     tools = await client.get_tools()
#     print("\n Available Tavily Tools are:")

#     for tool in tools:
#         print(tool.name)



# if __name__ == "__main__":
#     asyncio.run(func_tools())


search_tools = None

async def initialize_mcp():
    
    global search_tool

    if search_tools is not None:
        return search_tools
    
    tools = await client.get_tools()

    print("Available MCP Tavily Tools: ")

    for tool in tools:
        print(tool.name)

    search_tool = next(
        
      (tool for tool in tools if tool.name == "tavily_search"),
          None

    )
    

async def tavily_mcp_search(query: str):
    await initialize_mcp()
    result = await search_tool.ainvoke(
        {
            "query": query
        }
    )
    return result

