import json
import sys
from pathlib import Path

from src.chunker import split_documents
from src.document_loader import load_pdf
from src.embeddings import get_embedding_model
from src.vector_store import create_vector_store
from src.retriever import retrieve_documents


# --------------------------------------------------
# Configuration
# --------------------------------------------------

PDF_PATH = "sampletest_project2.pdf"

QUESTIONS_PATH = Path(
    "evaluation/questions.json"
)

TOP_K = 5

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200


# --------------------------------------------------
# Load evaluation questions
# --------------------------------------------------

def load_questions():

    with open(
        QUESTIONS_PATH,
        "r",
        encoding="utf-8",
    ) as file:

        return json.load(file)


# --------------------------------------------------
# Evaluate retrieval
# --------------------------------------------------

def evaluate():

    print("=" * 60)
    print("RAG RETRIEVAL EVALUATION")
    print("=" * 60)

    print()
    print("Loading PDF...")

    with open(
        PDF_PATH,
        "rb",
    ) as file:

        pdf_bytes = file.read()

    documents = load_pdf(
        file_bytes=pdf_bytes,
        file_name=Path(PDF_PATH).name,
    )

    print(
        f"Loaded {len(documents)} pages."
    )

    # --------------------------------------------------
    # Chunk documents
    # --------------------------------------------------

    print()
    print(
        f"Creating chunks "
        f"({CHUNK_SIZE} / {CHUNK_OVERLAP})..."
    )

    chunks = split_documents(
        documents=documents,
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )

    print(
        f"Created {len(chunks)} chunks."
    )

    # --------------------------------------------------
    # Embedding model
    # --------------------------------------------------

    print()
    print("Loading embedding model...")

    embedding_model = get_embedding_model()

    # --------------------------------------------------
    # Create vector store
    # --------------------------------------------------

    print()
    print("Building vector store...")

    vector_store = create_vector_store(
        documents=chunks,
        embedding_model=embedding_model,
    )

    print("Vector store ready.")

    # --------------------------------------------------
    # Load questions
    # --------------------------------------------------

    questions = load_questions()

    print()
    print(
        f"Evaluating {len(questions)} questions..."
    )

    print()

    hits = 0

    results = []

    # --------------------------------------------------
    # Evaluate every question
    # --------------------------------------------------

    for index, item in enumerate(
        questions,
        start=1,
    ):

        question = item["question"]

        relevant_pages = set(
            item["relevant_pages"]
        )

        retrieved_documents = (
            retrieve_documents(
                vector_store=vector_store,
                query=question,
                k=TOP_K,
            )
        )

        retrieved_pages = []

        for document, score in (
            retrieved_documents
        ):

            page = document.metadata.get(
                "page"
            )

            try:
                page = int(page)
            except (
                TypeError,
                ValueError,
            ):
                pass

            retrieved_pages.append(page)

        hit = any(
            page in relevant_pages
            for page in retrieved_pages
        )

        if hit:
            hits += 1

        results.append(
            {
                "question": question,
                "relevant_pages": list(
                    relevant_pages
                ),
                "retrieved_pages": (
                    retrieved_pages
                ),
                "hit": hit,
            }
        )

        status = "✅ HIT" if hit else "❌ MISS"

        print(
            f"{index:02d}. {status} | "
            f"{question}"
        )

        print(
            f"    Expected pages: "
            f"{sorted(relevant_pages)}"
        )

        print(
            f"    Retrieved pages: "
            f"{retrieved_pages}"
        )

        print()

    # --------------------------------------------------
    # Calculate retrieval hit rate
    # --------------------------------------------------

    total_questions = len(
        questions
    )

    hit_rate = (
        hits / total_questions
    ) * 100

    print("=" * 60)
    print("RESULTS")
    print("=" * 60)

    print(
        f"Questions evaluated: "
        f"{total_questions}"
    )

    print(
        f"Successful retrievals: "
        f"{hits}"
    )

    print(
        f"Failed retrievals: "
        f"{total_questions - hits}"
    )

    print(
        f"Top-{TOP_K} Retrieval Hit Rate: "
        f"{hit_rate:.2f}%"
    )

    print("=" * 60)

    # --------------------------------------------------
    # Save results
    # --------------------------------------------------

    output_path = Path(
        "evaluation/results.json"
    )

    with open(
        output_path,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            {
                "chunk_size": CHUNK_SIZE,
                "chunk_overlap": CHUNK_OVERLAP,
                "top_k": TOP_K,
                "total_questions": (
                    total_questions
                ),
                "hits": hits,
                "hit_rate": hit_rate,
                "results": results,
            },
            file,
            indent=4,
        )

    print()
    print(
        f"Detailed results saved to: "
        f"{output_path}"
    )


# --------------------------------------------------
# Main
# --------------------------------------------------

if __name__ == "__main__":

    evaluate()
