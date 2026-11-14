import json
import os
from typing import Any

from dotenv import load_dotenv
from mem0 import Memory
from openai import OpenAI

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
USER_ID = "sanjai"

if not OPENAI_API_KEY:
    raise RuntimeError("Set OPENAI_API_KEY in the environment or .env file.")
if not USER_ID:
    raise RuntimeError("Set MEM0_USER_ID to a stable, unique user identifier.")

client = OpenAI(api_key=OPENAI_API_KEY)

config = {
    "version": "v1.1",
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
}

mem_client = Memory.from_config(config)


def search_memories(query: str) -> list[dict[str, str]]:
    response = mem_client.search(query, filters={"user_id": USER_ID})
    raw_results = (
        response.get("results", []) if isinstance(response, dict) else response
    )

    if not isinstance(raw_results, list):
        raise TypeError("Mem0 search returned an unexpected result.")

    memories = []
    for item in raw_results:
        if not isinstance(item, dict):
            raise TypeError("Mem0 search returned a malformed memory.")

        memory_id = item.get("id")
        memory_text = item.get("memory")
        if not isinstance(memory_id, str) or not isinstance(memory_text, str):
            raise TypeError("Mem0 search returned a memory without an ID or text.")
        memories.append({"id": memory_id, "memory": memory_text})

    return memories


def reconcile_memories(user_message: str) -> None:
    existing = search_memories(user_message)
    if not existing:
        mem_client.add(
            messages=[{"role": "user", "content": user_message}],
            user_id=USER_ID,
        )
        return

    response = client.chat.completions.create(
        model="gpt-4.1",
        response_format={"type": "json_object"},
        messages=[
            {
                "role": "system",
                "content": (
                    "Reconcile the latest user statement with the existing memories. "
                    "Treat the latest statement as authoritative only when it clearly "
                    "corrects or contradicts a memory. Update only those directly "
                    "conflicting memories; do not remove unrelated or ambiguous facts. "
                    "Return JSON with exactly these keys: "
                    '"updates" (array of {"id": existing memory ID, '
                    '"text": replacement memory text}) and '
                    '"new_memories" (array of concise, durable facts from the latest '
                    "statement that are not already represented by an update). "
                    "Use only IDs supplied in the existing memories. Do not infer facts."
                ),
            },
            {
                "role": "user",
                "content": json.dumps(
                    {
                        "latest_statement": user_message,
                        "existing_memories": existing,
                    }
                ),
            },
        ],
    )

    content = response.choices[0].message.content
    if content is None:
        raise RuntimeError("The memory reconciliation model returned no response.")

    try:
        plan: Any = json.loads(content)
    except json.JSONDecodeError as exc:
        raise ValueError(
            "The memory reconciliation response was not valid JSON."
        ) from exc

    if not isinstance(plan, dict):
        raise ValueError("The memory reconciliation response must be a JSON object.")

    updates = plan.get("updates")
    new_memories = plan.get("new_memories")
    if not isinstance(updates, list) or not isinstance(new_memories, list):
        raise ValueError("The memory reconciliation response has an invalid shape.")

    existing_ids = {memory["id"] for memory in existing}
    updated_ids = set()
    validated_updates = []
    for update in updates:
        if not isinstance(update, dict):
            raise ValueError(
                "The memory reconciliation response contains an invalid update."
            )

        memory_id = update.get("id")
        text = update.get("text")
        if (
            not isinstance(memory_id, str)
            or memory_id not in existing_ids
            or memory_id in updated_ids
            or not isinstance(text, str)
            or not text.strip()
        ):
            raise ValueError(
                "The memory reconciliation response contains an invalid update."
            )

        updated_ids.add(memory_id)
        validated_updates.append({"id": memory_id, "text": text.strip()})

    validated_new_memories = []
    for memory in new_memories:
        if not isinstance(memory, str) or not memory.strip():
            raise ValueError(
                "The memory reconciliation response contains an invalid fact."
            )
        validated_new_memories.append(memory.strip())

    for update in validated_updates:
        mem_client.update(memory_id=update["id"], text=update["text"])

    for memory in validated_new_memories:
        mem_client.add(
            messages=[{"role": "user", "content": memory}],
            user_id=USER_ID,
            infer=False,
        )


while True:
    try:
        user_query = input("> ").strip()
    except EOFError:
        break

    if user_query.lower() in {"exit", "quit"}:
        break
    if not user_query:
        continue

    reconcile_memories(user_query)
    relevant_memories = search_memories(user_query)
    memory_context = "\n".join(f"- {memory['memory']}" for memory in relevant_memories)

    response = client.chat.completions.create(
        model="gpt-4.1-mini",
        messages=[
            {
                "role": "system",
                "content": (
                    "Answer the user using relevant saved memories when applicable. "
                    "Treat a saved, explicitly updated preference as current. "
                    f"Relevant memories:\n{memory_context or '(none)'}"
                ),
            },
            {"role": "user", "content": user_query},
        ],
    )

    ai_response = response.choices[0].message.content
    print("AI response:", ai_response or "")
