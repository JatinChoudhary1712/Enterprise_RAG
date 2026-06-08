from fastapi import FastAPI
from pydantic import BaseModel
from generation_layer.genertor import generate_answer

app = FastAPI()

class ChatRequest(BaseModel):
    question : str
    

@app.post("/chat")
def chat_input(request : ChatRequest): 
    response = generate_answer(request.question)
    return {
        "answer" : response
    }
    
    
@app.get("/health")
def health_check():
    return {
        "status" : 200
    }