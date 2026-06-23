import os
from typing import TypedDict, Annotated
import operator
from nodes.flight_node import *
from nodes.hotel_agent_node import *
from nodes.itenary_node import *
from nodes.final_agent_node import *
import psycopg
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.postgres import PostgresSaver
from langchain_core.messages import (
    AnyMessage,
    HumanMessage,
    AIMessage,
    SystemMessage,
)

from langchain_groq import ChatGroq
from state.state import TravelState
from tools.tavily_tool import tavily_search
from tools.flight_tool import search_flights
from dotenv import load_dotenv
load_dotenv()


DB_URL = os.getenv("DATABASE_URL")

llm = ChatGroq(
    model="llama-3.3-70b-versatile"
)


## Define the graph
graph_builder = StateGraph(TravelState)


graph_builder.add_node("flight_node", flight_agent)
graph_builder.add_node("hotel_node", hotel_agent)
graph_builder.add_node("itenary_node", itinerary_agent)
graph_builder.add_node("final_node", final_agent)


graph_builder.add_edge(START , "flight_node")
graph_builder.add_edge("flight_node" , "hotel_node")
graph_builder.add_edge("hotel_node" , "itenary_node")
graph_builder.add_edge("itenary_node" , "final_node")
graph_builder.add_edge("final_node" , END)


conn = psycopg.connect(DB_URL, autocommit=True)
checkpointer = PostgresSaver(conn)
checkpointer.setup()


app = graph_builder.compile(checkpointer=checkpointer)

if __name__ == "__main__":
    config = {
        "configurable": {
            "thread_id": "user_huzaifa"
        }
    }

    user_input = input("Enter travel request: ")

    result = app.invoke(
        {
            "messages": [
                HumanMessage(content=user_input)
            ],
            "user_query": user_input,
            "flight_results": "",
            "hotel_results": "",
            "itinerary": "",
            "llm_calls": 0
        },
        config=config
    )

    print("\nFINAL RESPONSE:\n")

    for msg in result["messages"]:
        print(msg.content)