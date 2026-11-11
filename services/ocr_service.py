# Import necessary libraries
import base64  # For encoding images
import io  # For byte streams
import sys  # System operations
import os  # File operations
import subprocess  # For running external commands

# Ensure proper imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import AI image processing and model config
from core.ollama_client import generate_with_image  # Function to process images with AI
from config import CHAT_MODEL  # Reuse chat model for image extraction


def is_poppler_installed():
    """Return True when the `pdftotext` tool from Poppler is available.

    The project uses Poppler/pdf2image for scanned PDF -> image conversion.
    """
    try:
        result = subprocess.run(["pdftotext", "-v"], capture_output=True, timeout=2)  # Check if pdftotext is available
        return result.returncode == 0  # Return True if command succeeds
    except Exception:  # If command fails or times out
        return False  # Poppler not installed


def image_to_base64(image):
    """Convert a PIL image to a base64-encoded PNG string."""
    buffer = io.BytesIO()  # Create byte buffer
    image.save(buffer, format="PNG")  # Save image as PNG to buffer
    return base64.b64encode(buffer.getvalue()).decode()  # Encode to base64 and return string


def extract_text_scanned(pdf_path):
    """Convert a scanned PDF to images and extract text using the chat model.

    This function uses `pdf2image` to rasterize pages and then calls the
    configured chat model via `generate_with_image` to extract readable text.
    """
    if not is_poppler_installed():  # Check if required tools are installed
        raise Exception(  # Raise error if not installed
            "Poppler not installed. Install from: https://github.com/oschwartz10612/poppler-windows/releases"
        )

    try:
        from pdf2image import convert_from_path  # Import PDF to image converter

        pages = convert_from_path(pdf_path)  # Convert PDF pages to images
        results = []  # List to store results

        for i, page in enumerate(pages):  # For each page image
            img_base64 = image_to_base64(page)  # Convert image to base64
            prompt = "Extract all readable text from this university document page."  # AI prompt
            text = generate_with_image(CHAT_MODEL, prompt, img_base64)  # Extract text with AI
            results.append({"page": i + 1, "text": text})  # Add to results

        return results  # Return list of page texts
    except ImportError:  # If pdf2image not installed
        raise Exception("pdf2image not installed. Install with: pip install pdf2image")
    except Exception as e:  # Other errors
        raise Exception(f"OCR Error: {str(e)}")


