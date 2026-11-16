from dotenv import load_dotenv
import os

from mem0 import Memory
from openai import OpenAI
import json

load_dotenv()

client = OpenAI()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
NEO_USERNAME = os.getenv("NEO_USERNAME")
NEO_PASSWORD = os.getenv("NEO_PASSWORD")
NEO_COLLECTION_STRING = os.getenv("NEO_COLLECTION_STRING")


config = {
    "version": "v1.1",
    "enable_graph": True,
    "embedder": {
        "provider": "openai",
        "config": {
            "api_key": OPENAI_API_KEY,
            "model": "text-embedding-3-small",
        },
    },
    "llm": {
        "provider": "openai",
        "config": {
            "api_key": OPENAI_API_KEY,
            "model": "gpt-4.1",
        },
    },
    "vector_store": {
        "provider": "qdrant",
        "config": {
            "host": "localhost",
            "port": 6333,
        },
    },
    "graph_store": {
        "provider": "neo4j",
        "config": {
            "url": NEO_COLLECTION_STRING,
            "username": NEO_USERNAME,
            "password": NEO_PASSWORD,
        },
    },
}

mem_client = Memory.from_config(config)
while True:
    user_query = input("> ")

    search_memory = mem_client.search(query=user_query, filters={"user_id": "sanjai"})

    memories = [
        f"ID: {mem.get('id')}\nMemory: {mem.get('memory')} "
        for mem in search_memory.get("results")
    ]

    SYSTEM_PROMPT = f"""
    Here is the context about the user:
    {json.dumps(memories)}

    """

    response = client.chat.completions.create(
        model="gpt-4.1-mini",
        messages=[
            {"role": "user", "content": user_query},
            {"role": "system", "content": SYSTEM_PROMPT},
        ],
    )

    ai_response = response.choices[0].message.content

    print("AI response:", ai_response)

    # Store conversation in Mem0
    result = mem_client.add(
        messages=[
            {"role": "user", "content": user_query},
            {"role": "assistant", "content": ai_response},
        ],
        user_id="sanjai",
    )
    print("Memory:", result)
