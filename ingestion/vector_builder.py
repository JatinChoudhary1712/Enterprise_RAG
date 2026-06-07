from langchain_openai import OpenAIEmbeddings
from ingestion.chunking import split_documents
from langchain_community.vectorstores import FAISS
from dotenv import load_dotenv
load_dotenv()

chunks = split_documents()
embedding_model = OpenAIEmbeddings()

vector_store = FAISS.from_documents(
    documents=chunks,
    embedding = embedding_model
)

vector_store.save_local("vector_db")
