from typing import List, Tuple

from langchain_core.documents import Document


def retrieve_documents(
    vector_store,
    query: str,
    k: int = 5,
) -> List[Tuple[Document, float]]:
    """
    Retrieve the top-k most relevant document chunks
    for a user query.

    Returns:
        List of (Document, similarity_score)
    """

    results = vector_store.similarity_search_with_score(
        query,
        k=k,
    )

    return results
