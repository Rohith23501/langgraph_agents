
from typing import Annotated, Sequence, TypedDict
from dotenv import load_dotenv
from langchain_core.messages import BaseMessage, SystemMessage, HumanMessage, AIMessage
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langgraph.graph.message import add_messages
from langgraph.graph import StateGraph, START , END
from langgraph.prebuilt import ToolNode

load_dotenv()

# Reducer Function
# Rule that controls how updates from nodes are combined with the existing state
# Tells us how to merge new data into the current state

# Without a reducer, updates would  have replaced the existing value entirely

class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]


@tool
def add(a: int, b: int):
    """ 
        An addition function which adds the two given numbers together
    """
    return a + b

@tool
def subtract(a: int, b: int):
    """ 
        An subtraction function which calculates the difference of two given numbers together
    """
    return a - b

@tool
def multiply(a: int, b: int):
    """ 
        An multiplication function which finds the product of two given numbers
    """
    return a * b


tools = [add, subtract, multiply]

model = ChatOpenAI(model="gpt-4o").bind_tools(tools)

def model_call(state:AgentState) -> AgentState:
    system_prompt = SystemMessage(content="""You are my helpful associate, who answers my queries to 
                                  the best of your abilities and always to tries to clear and concise"""
                                  )
    response = model.invoke([system_prompt] + state["messages"])
    return {"messages":[response]}

def should_continue(state: AgentState):
    messages = state["messages"]
    last_message = messages[-1]
    if not last_message.tool_calls:
        return "end"
    else:
        return "continue"



graph = StateGraph(AgentState)
graph.add_node("our_agent", model_call)

tool_node = ToolNode(tools=tools)
graph.add_node("tools", tool_node)
graph.add_edge("tools", "our_agent")
graph.add_edge(START, "our_agent")
graph.add_conditional_edges("our_agent",
                            should_continue,
                            {"continue": "tools",
                             "end": END})


app = graph.compile()


def print_stream(stream):
    for s in stream:
        message = s["messages"][-1]
        if isinstance(message, tuple):
            print(message)
        else:
            message.pretty_print()

inputs = {"messages": [("user", "Add 3 + 4 and then multiply the results by 10")]}
print_stream(app.stream(inputs, stream_mode="values"))
