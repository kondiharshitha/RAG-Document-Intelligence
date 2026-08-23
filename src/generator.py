from typing import List, Tuple

from langchain_core.documents import Document
from langchain_groq import ChatGroq


MODEL_NAME = "openai/gpt-oss-120b"


def create_llm(api_key: str):
    """
    Create the Groq LLM used for answer generation.
    """

    return ChatGroq(
        api_key=api_key,
        model=MODEL_NAME,
        temperature=0,
    )


def build_context(
    retrieved_documents: List[Tuple[Document, float]],
) -> str:
    """
    Convert retrieved documents into a context string
    containing source and page metadata.
    """

    context_parts = []

    for index, (document, score) in enumerate(
        retrieved_documents,
        start=1,
    ):
        source = document.metadata.get(
            "source",
            "Unknown",
        )

        page = document.metadata.get(
            "page",
            "Unknown",
        )

        context_parts.append(
            f"""
SOURCE {index}
Document: {source}
Page: {page}

Content:
{document.page_content}
"""
        )

    return "\n".join(context_parts)


def generate_answer(
    llm,
    question: str,
    retrieved_documents: List[Tuple[Document, float]],
):
    """
    Generate an answer using only retrieved document context.
    """

    context = build_context(retrieved_documents)

    prompt = f"""
You are an evidence-grounded document question-answering assistant.

Answer the user's question using ONLY the information
contained in the retrieved document context.

Rules:

1. Do not use outside knowledge.
2. Do not invent facts.
3. If the retrieved context does not contain enough
   information to answer the question, say:
   "I don't have enough evidence in the provided documents
   to answer this question."
4. Cite the document name and page number for the
   information used in your answer.
5. Keep the answer clear and concise.

Retrieved Document Context:
{context}

User Question:
{question}

Answer:
"""

    response = llm.invoke(prompt)

    return response.content
