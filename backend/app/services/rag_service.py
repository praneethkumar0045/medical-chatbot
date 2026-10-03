from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough

from app.services.llm_service import llm
from app.services.vector_service import retriever

system_prompt = (
    "You are an assistant for question-answering tasks. "
    "Use the following pieces of retrieved context to answer "
    "the question. If you don't know the answer, say that you "
    "don't know. Use three sentences maximum and keep the "
    "answer concise."
    "\n\n"
    "{context}"
)


prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("human", "{input}"),
    ]
)


def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


rag_chain = (
    {
        "context": retriever | format_docs,
        "input": RunnablePassthrough(),
    }
    | prompt
    | llm
)


def ask_question(query: str):

    try:
        response = rag_chain.invoke(query)

        if isinstance(response.content, str):
            return response.content

        if isinstance(response.content, list):
            return "".join(
                item.get("text", "")
                for item in response.content
                if isinstance(item, dict)
            )

        return str(response.content)

    except Exception as e:
        raise RuntimeError(f"RAG processing failed: {str(e)}") from e
