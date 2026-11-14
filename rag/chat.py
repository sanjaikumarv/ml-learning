from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv
from langchain_qdrant import QdrantVectorStore
from openai import OpenAI
load_dotenv()


openai_client = OpenAI()

embeding_model = OpenAIEmbeddings(
    model='text-embedding-3-large'
)



vector_store = QdrantVectorStore.from_existing_collection(
    embedding=embeding_model,
    url="http://localhost:6333",
    collection_name="learning-rag"
)

# Take user input and perform a similarity search
user_query = input("Enter your query: ")


# relevant chunks from the vector DB
search_result = vector_store.similarity_search(
    query=user_query)

content = "\n\n\n".join([f"Page Content: {result.page_content}\nPage Number: {result.metadata['page']}\nFile Location: {result.metadata['source']}" for result in search_result])

SYSTEM_PROMPT = f"""You are a helpful assistant who answers user query based on the available context retrieved from a PDF file along with page_contents and page number.

You should only answer the user based on the following context ans navigate the user to open the right  page number of know more.


Context:     
{content}
"""

result = openai_client.chat.completions.create(
    model="gpt-4o",
    messages=[
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_query}
    ]
)


print("Answer: ", result.choices[0].message.content)

