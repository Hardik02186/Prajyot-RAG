"""
Document processing helpers.

`process_document(path)` returns a list of pages where each page is a dict
with keys `page` and `text`. It attempts normal PDF extraction first and
falls back to OCR for scanned PDFs when Poppler is available.
"""

# Import necessary libraries
import fitz  # PyMuPDF for PDF processing
import sys  # System operations
import os  # File operations

# Ensure import path for sibling packages
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import text extraction functions
from services.text_services import extract_text_pdf  # Normal PDF text extraction
from services.ocr_service import extract_text_scanned, is_poppler_installed  # OCR for scanned PDFs


def is_scanned_pdf(path):
    """Return True if the PDF appears to contain little extractable text.

    This is a heuristic used to decide whether to try OCR extraction.
    """
    try:
        doc = fitz.open(path)  # Open PDF with PyMuPDF
        text = "".join([p.get_text() for p in doc])  # Extract all text
        return len(text.strip()) < 100  # If less than 100 chars, likely scanned
    except Exception:  # If error opening
        return False  # Assume not scanned
        

def process_document(path):
    """Extract pages from the file path.

    Returns a list of page dicts: {"page": int, "text": str}.
    Falls back to OCR when the PDF seems scanned.
    """
    try:
        pages = extract_text_pdf(path)  # Try normal text extraction
        text_content = "".join([p.get("text", "") for p in pages])  # Combine all text

        # If extraction yields very little, attempt OCR (scanned PDF)
        if pages and len(text_content) < 100:  # If little text extracted
            if not is_poppler_installed():  # Check if OCR tools available
                return []  # Can't do OCR
            try:
                ocr_pages = extract_text_scanned(path)  # Try OCR extraction
                return ocr_pages or []  # Return OCR results or empty
            except Exception:  # If OCR fails
                return []  # Return empty

        return pages  # Return normal extraction results
    except Exception:  # If any error
        return []  # Return empty list
