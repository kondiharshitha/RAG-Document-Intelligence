import streamlit as st

from src.document_loader import load_pdf
from src.chunker import split_documents
from src.embeddings import get_embedding_model
from src.vector_store import create_vector_store
from src.retriever import retrieve_documents
from src.generator import create_llm, generate_answer


# --------------------------------------------------
# Page configuration
# --------------------------------------------------

st.set_page_config(
    page_title="RAG Document Intelligence",
    page_icon="📚",
    layout="wide",
)


# --------------------------------------------------
# Header
# --------------------------------------------------

st.title("📚 RAG Document Intelligence")

st.write(
    "Upload documents and ask questions using "
    "evidence-grounded retrieval augmented generation."
)


# --------------------------------------------------
# Session state
# --------------------------------------------------

if "vector_store" not in st.session_state:
    st.session_state.vector_store = None

if "llm" not in st.session_state:
    st.session_state.llm = None

if "documents_processed" not in st.session_state:
    st.session_state.documents_processed = 0

if "pages_processed" not in st.session_state:
    st.session_state.pages_processed = 0

if "chunks_created" not in st.session_state:
    st.session_state.chunks_created = 0


# --------------------------------------------------
# Sidebar
# --------------------------------------------------

with st.sidebar:

    st.header("Configuration")

    groq_api_key = st.secrets.get(
        "GROQ_API_KEY",
        "",
    )

    uploaded_files = st.file_uploader(
        "Upload PDF documents",
        type=["pdf"],
        accept_multiple_files=True,
    )

    build_button = st.button(
        "Build Knowledge Base",
        type="primary",
        use_container_width=True,
    )

    uploaded_files = st.file_uploader(
        "Upload PDF documents",
        type=["pdf"],
        accept_multiple_files=True,
    )

    build_button = st.button(
        "Build Knowledge Base",
        type="primary",
        use_container_width=True,
    )


# --------------------------------------------------
# Build knowledge base
# --------------------------------------------------

if build_button:

    if not groq_api_key:

        st.error(
            "Please enter your Groq API key."
        )

    elif not uploaded_files:

        st.error(
            "Please upload at least one PDF."
        )

    else:

        try:

            with st.spinner(
                "Reading and processing documents..."
            ):

                # ------------------------------------------
                # 1. Load PDF pages
                # ------------------------------------------

                all_documents = []

                for uploaded_file in uploaded_files:

                    file_bytes = uploaded_file.getvalue()

                    documents = load_pdf(
                        file_bytes=file_bytes,
                        file_name=uploaded_file.name,
                    )

                    all_documents.extend(documents)

                if not all_documents:

                    st.error(
                        "No readable text was found "
                        "in the uploaded PDFs."
                    )

                    st.stop()

                # ------------------------------------------
                # 2. Split into chunks
                # ------------------------------------------

                chunks = split_documents(
                    documents=all_documents,
                    chunk_size=1000,
                    chunk_overlap=200,
                )

                # ------------------------------------------
                # 3. Create embedding model
                # ------------------------------------------

                embedding_model = (
                    get_embedding_model()
                )

                # ------------------------------------------
                # 4. Store embeddings in Chroma
                # ------------------------------------------

                vector_store = create_vector_store(
                    documents=chunks,
                    embedding_model=embedding_model,
                )

                # ------------------------------------------
                # 5. Create LLM
                # ------------------------------------------

                llm = create_llm(
                    api_key=groq_api_key
                )

                # ------------------------------------------
                # Save to session state
                # ------------------------------------------

                st.session_state.vector_store = (
                    vector_store
                )

                st.session_state.llm = llm

                st.session_state.documents_processed = (
                    len(uploaded_files)
                )

                st.session_state.pages_processed = (
                    len(all_documents)
                )

                st.session_state.chunks_created = (
                    len(chunks)
                )

            st.success(
                "Knowledge base created successfully! 🎉"
            )

        except Exception as error:

            st.error(
                f"An error occurred while processing "
                f"the documents: {error}"
            )


# --------------------------------------------------
# Knowledge base statistics
# --------------------------------------------------

if st.session_state.vector_store is not None:

    st.sidebar.divider()

    st.sidebar.subheader(
        "Knowledge Base"
    )

    st.sidebar.write(
        f"📄 Documents: "
        f"{st.session_state.documents_processed}"
    )

    st.sidebar.write(
        f"📑 Pages: "
        f"{st.session_state.pages_processed}"
    )

    st.sidebar.write(
        f"🧩 Chunks: "
        f"{st.session_state.chunks_created}"
    )

    st.sidebar.write(
        "🧠 Embeddings: "
        "all-MiniLM-L6-v2"
    )

    st.sidebar.write(
        "🗄️ Vector DB: Chroma"
    )


# --------------------------------------------------
# Question answering
# --------------------------------------------------

if st.session_state.vector_store is not None:

    st.divider()

    st.subheader(
        "💬 Ask your documents"
    )

    question = st.text_input(
        "Question",
        placeholder=(
            "e.g. What was the company's revenue?"
        ),
    )

    ask_button = st.button(
        "Ask Question",
        type="primary",
    )

    if ask_button:

        if not question.strip():

            st.warning(
                "Please enter a question."
            )

        else:

            try:

                with st.spinner(
                    "Searching the knowledge base..."
                ):

                    retrieved_documents = (
                        retrieve_documents(
                            vector_store=(
                                st.session_state.vector_store
                            ),
                            query=question,
                            k=5,
                        )
                    )

                with st.spinner(
                    "Generating answer..."
                ):

                    answer = generate_answer(
                        llm=st.session_state.llm,
                        question=question,
                        retrieved_documents=(
                            retrieved_documents
                        ),
                    )

                # ------------------------------------------
                # Answer
                # ------------------------------------------

                st.subheader(
                    "Answer"
                )

                st.write(answer)

                # ------------------------------------------
                # Retrieved sources
                # ------------------------------------------

                st.subheader(
                    "📚 Retrieved Sources"
                )

                for index, (
                    document,
                    score,
                ) in enumerate(
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

                    chunk_id = document.metadata.get(
                        "chunk_id",
                        "Unknown",
                    )

                    with st.expander(
                        f"Source {index} — "
                        f"{source} | Page {page}"
                    ):

                        st.write(
                            f"**Chunk ID:** {chunk_id}"
                        )

                        st.write(
                            f"**Distance score:** "
                            f"{score:.4f}"
                        )

                        st.write(
                            document.page_content
                        )

            except Exception as error:

                st.error(
                    f"An error occurred while answering "
                    f"the question: {error}"
                )
