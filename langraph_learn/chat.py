
from dotenv import load_dotenv
load_dotenv()

from typing_extensions import TypedDict
from typing import Annotated

from langgraph.graph.message import add_messages
from langgraph.graph import StateGraph, START, END

from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage


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
    print("Current state:", state)

    # Invoke LLM
    response = llm.invoke(state["messages"])

    return {
        "messages": [response]
    }


# 4. Sample Node
def samplenode(state: State):

    print("\nInside sample node")
    print("Current state:", state)

    return {
        "messages": [
            HumanMessage(content="This is a sample node.")
        ]
    }


# 5. Create Graph Builder
graph_builder = StateGraph(State)


# 6. Register Nodes
graph_builder.add_node("chatbot", chatbot)
graph_builder.add_node("samplenode", samplenode)


# 7. Define Edges
graph_builder.add_edge(START, "chatbot")
graph_builder.add_edge("chatbot", "samplenode")
graph_builder.add_edge("samplenode", END)


# 8. Compile Graph
graph = graph_builder.compile()


# 9. Execute Graph
updated_state = graph.invoke({
    "messages": [
        HumanMessage(content="Hi, my name is Sanjai.")
    ]
})


print("\nUpdated state:")
for message in updated_state["messages"]:
    print(message.type, ":", message.content)
