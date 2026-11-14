from dotenv import load_dotenv
from langgraph.graph import StateGraph
load_dotenv()
from typing import Optional,Literal
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from openai import OpenAI
from langchain_core.messages import HumanMessage


client = OpenAI()

class State(TypedDict):
    user_query:str
    llm_output: Optional[str]
    is_good: Optional[bool]


def chatbot(state: State):

    # Invoke LLM
    print
    response = client.chat.completions.create(
        model="gpt-4.1-mini",
        messages=[{"role": "user", "content": state["user_query"]}]
    )
    state['llm_output'] = response.choices[0].message.content   
    return state

def evaluation_response(state: State)->Literal["chat_gemini", 'endnode']:
    print("\n\nstate",state)
    if state["is_good"]:
        return 'endnode'
    
    return "chat_gemini"

def chat_gemini(state: State):
    # Invoke LLM
    response = client.chat.completions.create(
        model="gemini-1.5",
        messages=[{"role": "user", "content": state["user_query"]}]
    )
    state['llm_output'] = response.choices[0].message.content   
    return state
 
def endnode(state: State):
    return state

graph_builder = StateGraph(State)

graph_builder.add_node("chatbot", chatbot)
graph_builder.add_node("chat_gemini", chat_gemini)
graph_builder.add_node("endnode", endnode)


graph_builder.add_edge(START, "chatbot")
graph_builder.add_conditional_edges("chatbot", evaluation_response)

graph_builder.add_edge("chat_gemini",'endnode')
graph_builder.add_edge("endnode", END)


graph = graph_builder.compile()

updated_state = graph.invoke(State({"user_query":"Hey, What 2+2?"}))


print(updated_state)