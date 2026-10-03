from langchain_core.runnables import RunnablePassthrough
from langchain_core.prompts import ChatPromptTemplate
from app.services.vector_service import retriever
from app.services.llm_service import llm

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


# Format retrieved documents
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


# Create RAG chain
rag_chain = (
    {"context": retriever | format_docs, "input": RunnablePassthrough()} | prompt | llm
)


def ask_question(query: str):
    response = rag_chain.invoke(query)

    if isinstance(response.content, str):
        return response.content

    if isinstance(response.content, list):
        return "".join(
            item.get("text", "") for item in response.content if isinstance(item, dict)
        )

    return str(response.content)
