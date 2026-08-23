from io import BytesIO
from typing import List

from pypdf import PdfReader
from langchain_core.documents import Document


def load_pdf(file_bytes: bytes, file_name: str) -> List[Document]:
    """
    Extract text from each page of a PDF.

    Each page becomes a LangChain Document with metadata
    containing the original file name and page number.
    """

    reader = PdfReader(BytesIO(file_bytes))

    documents = []

    for page_number, page in enumerate(reader.pages, start=1):
        text = page.extract_text()

        if not text or not text.strip():
            continue

        document = Document(
            page_content=text.strip(),
            metadata={
                "source": file_name,
                "page": page_number,
            },
        )

        documents.append(document)

    return documents
