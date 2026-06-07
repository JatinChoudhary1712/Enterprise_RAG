from ingestion.pdf_ingestion import load_Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

text_splitters = RecursiveCharacterTextSplitter(
    chunk_size = 800,
    chunk_overlap = 200
)

def split_documents(): 
    chunks = text_splitters.split_documents(documents=load_Document())
    return chunks 