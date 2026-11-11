"""PDF text extraction helpers.

This module uses PyMuPDF (`fitz`) to extract page-level text from PDFs.
The returned structure is a list of dicts: {"page": int, "text": str}.
"""

# Import PyMuPDF library
import fitz  # For PDF text extraction


def extract_text_pdf(path):
    """Extract textual content from each page of a PDF file."""
    doc = fitz.open(path)  # Open the PDF document
    pages = []  # List to store page data
    for page_number, page in enumerate(doc):  # For each page
        text = page.get_text()  # Extract text from page
        pages.append({"page": page_number + 1, "text": text})  # Add page dict (1-indexed)
    return pages  # Return list of page dictionaries
