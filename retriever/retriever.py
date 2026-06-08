from langchain_openai import OpenAIEmbeddings
from langchain_community.retrievers import BM25Retriever 
from langchain_classic.retrievers import EnsembleRetriever
from langchain_community.vectorstores import FAISS
from ingestion.chunking import split_documents

from dotenv import load_dotenv
load_dotenv()

embedding_model = OpenAIEmbeddings()

vector_store = FAISS.load_local(
    folder_path="vector_db",
    embeddings= embedding_model,
    allow_dangerous_deserialization=True
)

faiss_retriever = vector_store.as_retriever(search_kwargs={"k": 4})
bm25 = BM25Retriever.from_documents(split_documents())


main_retriever = EnsembleRetriever(
    retrievers=[
        bm25,
        faiss_retriever
    ],
    weights = [0.5 , 0.5]
)

def retriever(question : str):
    """ The hybrid search took place here """
    response = main_retriever.invoke(question)
    return response