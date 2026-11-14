from typing_extensions import TypedDict
from typing import Annotated
from langgraph.graph.message import add_messages
from langgraph.graph import StateGraph, START, END

class State(TypedDict): 
  messages:Annotated[list,add_messages]

def chatbot(state: State):
  print("\n\nInside chat node",state)
  return {"messages":["Hi, I am a chatbot. How can I assist you today?"]}

def samplenode(state: State):
  print("\n\nInside sample node",state)
  return {"messages":["This is a sample node."]}

graph_builder = StateGraph(State)

graph_builder.add_node("chatbot", chatbot)
graph_builder.add_node("samplenode", samplenode)

graph_builder.add_edge(START, "chatbot")
graph_builder.add_edge("chatbot", "samplenode")

graph_builder.add_edge("samplenode", END)

graph = graph_builder.compile()

updated_state=graph.invoke(State({"messages":['Hi, my name is sanjai.']}))

print("\n\nUpdated state after graph invocation:", updated_state)
