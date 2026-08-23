from typing import List

from langchain_chroma import Chroma
from langchain_core.documents import Document


COLLECTION_NAME = "rag_documents"
PERSIST_DIRECTORY = "chroma_db"


def create_vector_store(
    documents: List[Document],
    embedding_model,
    persist_directory: str = PERSIST_DIRECTORY,
    collection_name: str = COLLECTION_NAME,
):
    """
    Create a persistent Chroma vector store
    from document chunks.
    """

    vector_store = Chroma.from_documents(
        documents=documents,
        embedding=embedding_model,
        collection_name=collection_name,
        persist_directory=persist_directory,
    )

    return vector_store
