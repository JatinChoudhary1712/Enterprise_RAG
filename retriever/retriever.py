from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from dotenv import load_dotenv
load_dotenv()

embedding_model = OpenAIEmbeddings()

vector_store = FAISS.load_local(
    folder_path="vector_db",
    embeddings= embedding_model,
    allow_dangerous_deserialization=True
)


def retriever(query : str):
    """ retrieve the top k chunks """
    ret = vector_store.similarity_search(
        query=query,
        k = 4
    )
    return ret
