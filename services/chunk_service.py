# Import text splitter from LangChain
from langchain_text_splitters import RecursiveCharacterTextSplitter  # For splitting text into chunks
import sys  # System operations
import os  # File operations

# Ensure proper imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

"""Chunking helper: split pages into smaller text chunks with metadata.

Each returned chunk contains `text`, `page`, `source`, `document_id` and
`chunk_id`. `document_id` is a stable SHA256 of the source filename and
`chunk_id` uniquely identifies the chunk within the document.
"""

# Import chunk size and overlap settings
from config import CHUNK_SIZE, CHUNK_OVERLAP  # Configuration for chunking

# Create text splitter with configured parameters
splitter = RecursiveCharacterTextSplitter(
    chunk_size=CHUNK_SIZE,  # Maximum characters per chunk
    chunk_overlap=CHUNK_OVERLAP  # Characters to overlap between chunks
)


def chunk_pages(pages, source_name):
    """Split document pages into smaller text chunks with metadata."""
    chunks = []  # List to store chunk dictionaries

    # Create a stable document id based on the source name
    import hashlib  # Import here to avoid global import
    document_id = hashlib.sha256(source_name.encode("utf-8")).hexdigest()  # SHA256 hash of filename

    for page_idx, page in enumerate(pages):  # For each page
        split_texts = splitter.split_text(page["text"])  # Split page text into chunks

        for i, chunk in enumerate(split_texts):  # For each chunk in the page
            chunks.append({  # Add chunk metadata
                "text": chunk,  # The chunk text
                "page": page["page"],  # Page number
                "source": source_name,  # Source filename
                "document_id": document_id,  # Unique document ID
                "chunk_id": f"{document_id}-{page_idx}-{i}"  # Unique chunk ID
            })

    return chunks  # Return list of chunk dictionaries
