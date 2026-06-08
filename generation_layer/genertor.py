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

    sources = []
    for doc in result:
        sources.append(
            {
                "document": doc.metadata["source"],
                "page": doc.metadata["page"]
            }
        )

    # Build context for the LLM
    context = "\n\n".join(
        doc.page_content for doc in result
    )

    # Prepare prompt
    messages = [
        SystemMessage(content=System_Prompt),
        HumanMessage(
            content=f"Context:\n{context}\n\nQuestion: {question}"
        ),
    ]

    # Get LLM response
    response = llm.invoke(messages)

    # Return answer along with citations
    return {
        "answer": response.content,
        "sources": sources
    }
