from typing import List

from langchain_chroma import Chroma
from langchain_core.documents import Document


COLLECTION_NAME = "rag_documents"
PERSIST_DIRECTORY = "chroma_db"


def create_vector_store(
    documents: List[Document],
    embedding_model,
):
    """
    Create a persistent Chroma vector store
    from document chunks.
    """

    vector_store = Chroma.from_documents(
        documents=documents,
        embedding=embedding_model,
        collection_name=COLLECTION_NAME,
        persist_directory=PERSIST_DIRECTORY,
    )

    return vector_store
