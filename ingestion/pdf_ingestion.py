from langchain_community.document_loaders import PyMuPDFLoader , DirectoryLoader
from dotenv import load_dotenv
load_dotenv()

def load_Document():
    loader = DirectoryLoader(
    path="documents", 
    glob="*.pdf",
    loader_cls=PyMuPDFLoader
    )
    
    docs = loader.load()
    
    return docs

