import json
import shutil
import sys
from pathlib import Path

# --------------------------------------------------
# Make project root importable
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from src.document_loader import load_pdf
from src.chunker import split_documents
from src.embeddings import get_embedding_model
from src.vector_store import create_vector_store
from src.retriever import retrieve_documents


# --------------------------------------------------
# Configuration
# --------------------------------------------------

PDF_PATH = PROJECT_ROOT / "data" / "sampletest_project2.pdf"

QUESTIONS_PATH = (
    PROJECT_ROOT
    / "evaluation"
    / "questions.json"
)

EVALUATION_DB = (
    PROJECT_ROOT
    / "evaluation_chroma_db"
)

TOP_K = 5

# Baseline chunking configuration
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

COLLECTION_NAME = "evaluation_collection"


# --------------------------------------------------
# Load evaluation questions
# --------------------------------------------------

def load_questions():
    """
    Load the manually verified evaluation questions
    and their expected relevant pages.
    """

    with open(
        QUESTIONS_PATH,
        "r",
        encoding="utf-8",
    ) as file:

        return json.load(file)


# --------------------------------------------------
# Clean previous evaluation database
# --------------------------------------------------

def clean_evaluation_database():
    """
    Remove the previous evaluation Chroma database.

    This ensures every experiment starts with a
    completely fresh vector store.
    """

    if EVALUATION_DB.exists():

        print(
            "Removing previous evaluation database..."
        )

        shutil.rmtree(EVALUATION_DB)


# --------------------------------------------------
# Load PDF
# --------------------------------------------------

def load_evaluation_document():
    """
    Load the evaluation PDF from the data directory.
    """

    if not PDF_PATH.exists():

        raise FileNotFoundError(
            f"Evaluation PDF not found:\n"
            f"{PDF_PATH}"
        )

    print(
        f"Loading PDF: {PDF_PATH.name}"
    )

    with open(
        PDF_PATH,
        "rb",
    ) as file:

        pdf_bytes = file.read()

    documents = load_pdf(
        file_bytes=pdf_bytes,
        file_name=PDF_PATH.name,
    )

    if not documents:

        raise ValueError(
            "No readable text was found in the PDF."
        )

    print(
        f"Loaded {len(documents)} pages."
    )

    return documents


# --------------------------------------------------
# Create vector store
# --------------------------------------------------

def build_vector_store(documents):
    """
    Chunk the document, create embeddings and build
    a fresh Chroma vector store.
    """

    print()
    print(
        "Creating document chunks..."
    )

    print(
        f"Chunk size: {CHUNK_SIZE} characters"
    )

    print(
        f"Chunk overlap: {CHUNK_OVERLAP} characters"
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
    print(
        "Loading Hugging Face embedding model..."
    )

    embedding_model = get_embedding_model()

    # --------------------------------------------------
    # Chroma vector store
    # --------------------------------------------------

    print()
    print(
        "Building Chroma vector store..."
    )

    vector_store = create_vector_store(
        documents=chunks,
        embedding_model=embedding_model,
        persist_directory=str(
            EVALUATION_DB
        ),
        collection_name=COLLECTION_NAME,
    )

    print(
        "Vector store ready."
    )

    return vector_store


# --------------------------------------------------
# Evaluate retrieval
# --------------------------------------------------

def evaluate_retrieval(
    vector_store,
    questions,
):
    """
    Evaluate whether the top-k retrieved chunks
    contain at least one manually verified relevant page.
    """

    hits = 0

    results = []

    print()
    print("=" * 70)
    print(
        f"EVALUATING {len(questions)} QUESTIONS"
    )
    print("=" * 70)
    print()

    for index, item in enumerate(
        questions,
        start=1,
    ):

        question = item["question"]

        relevant_pages = set(
            item["relevant_pages"]
        )

        # --------------------------------------------------
        # Retrieve top-k chunks
        # --------------------------------------------------

        retrieved_documents = (
            retrieve_documents(
                vector_store=vector_store,
                query=question,
                k=TOP_K,
            )
        )

        retrieved_pages = []

        scores = []

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

            retrieved_pages.append(
                page
            )

            scores.append(
                float(score)
            )

        # --------------------------------------------------
        # Determine whether retrieval was successful
        # --------------------------------------------------

        hit = any(
            page in relevant_pages
            for page in retrieved_pages
        )

        if hit:

            hits += 1

            status = "✅ HIT"

        else:

            status = "❌ MISS"

        # --------------------------------------------------
        # Save result
        # --------------------------------------------------

        results.append(
            {
                "question": question,
                "relevant_pages": sorted(
                    relevant_pages
                ),
                "retrieved_pages": (
                    retrieved_pages
                ),
                "scores": scores,
                "hit": hit,
            }
        )

        # --------------------------------------------------
        # Display result
        # --------------------------------------------------

        print(
            f"{index:02d}. {status}"
        )

        print(
            f"    Question: {question}"
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

    return hits, results


# --------------------------------------------------
# Calculate metrics
# --------------------------------------------------

def calculate_metrics(
    total_questions,
    hits,
):
    """
    Calculate retrieval hit rate.
    """

    if total_questions == 0:

        return 0.0

    return (
        hits / total_questions
    ) * 100


# --------------------------------------------------
# Save evaluation results
# --------------------------------------------------

def save_results(
    questions,
    hits,
    hit_rate,
    results,
):
    """
    Save detailed evaluation results as JSON.
    """

    output_path = (
        PROJECT_ROOT
        / "evaluation"
        / "results.json"
    )

    output = {
        "evaluation_document": (
            PDF_PATH.name
        ),
        "embedding_model": (
            "sentence-transformers/"
            "all-MiniLM-L6-v2"
        ),
        "chunk_size": CHUNK_SIZE,
        "chunk_overlap": CHUNK_OVERLAP,
        "top_k": TOP_K,
        "total_questions": len(
            questions
        ),
        "hits": hits,
        "misses": (
            len(questions) - hits
        ),
        "hit_rate": hit_rate,
        "results": results,
    }

    with open(
        output_path,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            output,
            file,
            indent=4,
        )

    return output_path


# --------------------------------------------------
# Main evaluation pipeline
# --------------------------------------------------

def main():

    print()
    print("=" * 70)
    print(
        "RAG DOCUMENT RETRIEVAL EVALUATION"
    )
    print("=" * 70)

    print()
    print(
        f"Top-K: {TOP_K}"
    )

    print(
        f"Chunk size: {CHUNK_SIZE}"
    )

    print(
        f"Chunk overlap: {CHUNK_OVERLAP}"
    )

    # --------------------------------------------------
    # 1. Clean old database
    # --------------------------------------------------

    clean_evaluation_database()

    # --------------------------------------------------
    # 2. Load PDF
    # --------------------------------------------------

    documents = (
        load_evaluation_document()
    )

    # --------------------------------------------------
    # 3. Build vector store
    # --------------------------------------------------

    vector_store = build_vector_store(
        documents
    )

    # --------------------------------------------------
    # 4. Load questions
    # --------------------------------------------------

    questions = load_questions()

    print()
    print(
        f"Loaded {len(questions)} evaluation questions."
    )

    # --------------------------------------------------
    # 5. Run retrieval evaluation
    # --------------------------------------------------

    hits, results = (
        evaluate_retrieval(
            vector_store=vector_store,
            questions=questions,
        )
    )

    # --------------------------------------------------
    # 6. Calculate hit rate
    # --------------------------------------------------

    total_questions = len(
        questions
    )

    hit_rate = calculate_metrics(
        total_questions=total_questions,
        hits=hits,
    )

    # --------------------------------------------------
    # 7. Display final results
    # --------------------------------------------------

    print()
    print("=" * 70)
    print("FINAL RESULTS")
    print("=" * 70)

    print(
        f"Questions evaluated : "
        f"{total_questions}"
    )

    print(
        f"Successful retrievals: "
        f"{hits}"
    )

    print(
        f"Failed retrievals    : "
        f"{total_questions - hits}"
    )

    print(
        f"Top-{TOP_K} Hit Rate    : "
        f"{hit_rate:.2f}%"
    )

    print("=" * 70)

    # --------------------------------------------------
    # 8. Save results
    # --------------------------------------------------

    output_path = save_results(
        questions=questions,
        hits=hits,
        hit_rate=hit_rate,
        results=results,
    )

    print()
    print(
        f"Detailed results saved to:"
    )

    print(
        output_path
    )

    print()


# --------------------------------------------------
# Entry point
# --------------------------------------------------

if __name__ == "__main__":

    main()
