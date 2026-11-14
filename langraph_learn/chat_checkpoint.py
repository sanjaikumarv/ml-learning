
from dotenv import load_dotenv
load_dotenv()

from typing_extensions import TypedDict
from typing import Annotated

from langgraph.graph.message import add_messages
from langgraph.graph import StateGraph, START, END

from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage
from langgraph.checkpoint.mongodb import MongoDBSaver
 
# 1. Define State
class State(TypedDict):
    messages: Annotated[list, add_messages]


# 2. Initialize LLM
llm = init_chat_model(
    model="gpt-4.1-mini",
    model_provider="openai"
)


# 3. Chatbot Node
def chatbot(state: State):

    print("\nInside chatbot node")
    # Invoke LLM
    response = llm.invoke(state["messages"])

    return {
        "messages": [response]
    }


# 4. Sample Node


# 5. Create Graph Builder
graph_builder = StateGraph(State)


# 6. Register Nodes
graph_builder.add_node("chatbot", chatbot)


# 7. Define Edges
graph_builder.add_edge(START, "chatbot")
graph_builder.add_edge("chatbot", END)


# 8. Compile Graph
graph = graph_builder.compile()

def compile_graph_with_check_pointer(checkpointer):
    return graph_builder.compile(checkpointer=checkpointer)
   
DB_URI = 'mongodb://localhost:27017'
with MongoDBSaver.from_conn_string(DB_URI) as checkpointer:
    graph_with_check_pointer = compile_graph_with_check_pointer(checkpointer)
# 9. Execute Graph

    config = {
    "configurable":{
        "thread_id":"sanjai"
    }
}
    for chunk in graph_with_check_pointer.stream({
    "messages": [
        HumanMessage(content="What iam learning ?")
    ]
    },config,
    stream_mode="values"
    ):
        chunk["messages"][-1].pretty_print()
        


    print("\nUpdated state:")
   
