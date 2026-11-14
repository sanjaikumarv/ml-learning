from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
pdf_path = Path(__file__).parent / "nodejs.pdf"
from langchain_qdrant import QdrantVectorStore
from dotenv import load_dotenv
load_dotenv()
# Loading pdf file using PyPDFLoader
loader = PyPDFLoader(str(pdf_path))
docs = loader.load()

print(docs[30])


# Splitting the document into chunks using RecursiveCharacterTextSplitter
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=400
)

chunks = text_splitter.split_documents(docs)

# Vector embeddings

embeding_model = OpenAIEmbeddings(
    model='text-embedding-3-large'
)

vector_store = QdrantVectorStore.from_documents(
    documents=chunks,
    embedding=embeding_model,
    url="http://localhost:6333",
    collection_name="learning-rag"
)

print("Vector store created successfully!")