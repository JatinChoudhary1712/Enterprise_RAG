from prompts.prompts import System_Prompt
from retriever.retriever import retriever
from langchain_openai import ChatOpenAI

from dotenv import load_dotenv
from langchain_core.messages import SystemMessage,AIMessage,HumanMessage
load_dotenv()
llm = ChatOpenAI(
    model = "gpt-4o"
)
def generate_answer(question: str):
    """Get the response from the LLM for a given question."""
    result = retriever(question=question)
    context = "\n\n".join(doc.page_content for doc in result)
    messages = [
        SystemMessage(content=System_Prompt),
        HumanMessage(content=f"Context:\n{context}\n\nQuestion: {question}"),
    ]
    response = llm.invoke(messages)
    return response.content
