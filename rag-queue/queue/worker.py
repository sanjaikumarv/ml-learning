from openai import OpenAI
from langchain_openai import OpenAIEmbeddings
from langchain_qdrant import QdrantVectorStore
from dotenv import load_dotenv

load_dotenv()

openai_client = OpenAI()

embedding_model = OpenAIEmbeddings(model="text-embedding-3-large")

vector_store = QdrantVectorStore.from_existing_collection(
    embedding=embedding_model,
    url="http://localhost:6333",
    collection_name="learning-rag",
)


def process_query(query: str):
    print("Processing query:", query)

    # Retrieve relevant chunks from Qdrant
    search_result = vector_store.similarity_search(query=query, k=4)

    # Handle empty search results
    if not search_result:
        return "No relevant information found in the documents."

    # Combine retrieved chunks
    content = "\n\n".join([f"""
Page Content: {result.page_content}
Page Number: {result.metadata.get('page', 'N/A')}
File Location: {result.metadata.get('source', 'N/A')}
""" for result in search_result])

    SYSTEM_PROMPT = f"""
You are a helpful assistant who answers user queries
based only on the context retrieved from PDF documents.

Instructions:
1. Answer only using the provided context.
2. Do not make up information.
3. Mention the relevant page numbers.
4. Guide the user to the correct page for more information.
5. If the answer is not available in the context,
   clearly say that you could not find the information.

Context:
{content}
"""

    # Generate response using OpenAI
    result = openai_client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": query},
        ],
    )

    return result.choices[0].message.content
