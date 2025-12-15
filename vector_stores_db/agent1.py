## trial_agent_script

from typing import Annotated, List, Union, TypedDict
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langgraph.graph.message import add_messages
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END

from dotenv import load_dotenv
load_dotenv()

class AgentState(TypedDict):
    messages: Annotated[list, add_messages]
    response : str

llm = ChatOpenAI(model="gpt-4o")

def process(state: AgentState) -> AgentState:
    response = llm.invoke(state["messages"])
    print(response.content)
    return {"messages": response, 
            "response": response}

graph = StateGraph(AgentState)
graph.add_node("process", process)
graph.add_edge(START, "process")
graph.add_edge("process", END)
agent = graph.compile()

final_response = agent.invoke({"messages": [SystemMessage("You are a deep learning specialist who lead many innovative projects"),
                          HumanMessage("What do you think about autoencoders replacing GAN's?")]})