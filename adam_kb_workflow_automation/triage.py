import pymupdf
'''
This module provides functionality to analyze PDF documents and determine whether they require Optical Character Recognition (OCR) processing. It uses the `pymupdf` library to efficiently inspect the textual content of each page in a PDF and classifies the document based on the proportion of text-based pages.
Classes:
    OcrRequirement (Enum):
        Represents the OCR requirement classification for a PDF:
            - OCR_REQUIRED: The document is scanned or mixed-mode and requires OCR.
            - OCR_NOT_REQUIRED: The document is predominantly text-based and does not require OCR.
            - EMPTY_OR_CORRUPT: The file is empty or cannot be processed.
    PdfTriage:
        Provides methods to analyze and classify a PDF document's OCR requirement.
        - classify(pdf_path: Path) -> OcrRequirement:
            Analyzes the PDF at the given path and returns its OCR requirement classification.
Configuration:
    - MIN_CHARS_FOR_TEXTUAL_PAGE: Minimum number of characters for a page to be considered textual.
    - TEXTUAL_PAGE_PERCENTAGE_THRESHOLD: Proportion of textual pages required to skip OCR (default: 95%).
Logging:
    Uses the standard Python logging module to report processing status, warnings, and errors.
Usage:
    Instantiate `PdfTriage` and call the `classify` method with a PDF file path to determine if OCR is required.

'''
import logging
from pathlib import Path
from enum import Enum, auto

### Module-level Configuration Constants
# These thresholds can be tuned from a central config file in a future version.
MIN_CHARS_FOR_TEXTUAL_PAGE = 150
TEXTUAL_PAGE_PERCENTAGE_THRESHOLD = 0.80 #80%

'''
MIN_CHARS_FOR_TEXTUAL_PAGE (int):
    The minimum number of characters that must be present on a PDF page for it to be considered "textual."
    Pages with a character count above this threshold are classified as containing sufficient text content,
    which helps distinguish between text-based and scanned (image-based) pages.
    Default: 150
TEXTUAL_PAGE_PERCENTAGE_THRESHOLD (float):
    The proportion (between 0 and 1) of textual pages required for a PDF to be classified as not requiring OCR.
    
    If the percentage of textual pages in a document meets or exceeds this threshold, the document is considered predominantly text-based and OCR processing can be skipped.
    
    
    These constants are used by the PdfTriage class to determine whether a PDF document requires OCR processing.
    - Adjust MIN_CHARS_FOR_TEXTUAL_PAGE to fine-tune the sensitivity for what counts as a "textual" page.
    - Adjust TEXTUAL_PAGE_PERCENTAGE_THRESHOLD to control how strict the classification is for skipping OCR.
    
    For example, lowering the threshold will allow more mixed-mode documents to skip OCR, while raising it will require a higher proportion of textual pages to avoid OCR.
'''




logger = logging.getLogger(__name__)

class OcrRequirement(Enum):
    """
    Enum to represent the OCR requirement classification of a PDF.
    """
    OCR_REQUIRED = auto()         # Scanned or mixed-mode PDFs that must be processed.
    OCR_NOT_REQUIRED = auto()     # Predominantly text-based PDFs that can be skipped.
    EMPTY_OR_CORRUPT = auto()     # Files that cannot be processed.

class PdfTriage:
    """
    Analyzes and classifies a PDF to determine if it requires OCR processing.
    """
    def classify(self, pdf_path: Path) -> OcrRequirement:
        """
        Analyzes a PDF page by page to classify its OCR requirement.

        This method checks if a document is predominantly text-based. If not,
        it is flagged as requiring OCR to ensure all content (including from
        scanned pages in mixed-mode documents) is captured.

        Args:
            pdf_path (Path): The file path to the PDF document.

        Returns:
            OcrRequirement: The classification indicating if OCR is needed.
        """
        try:
            doc = pymupdf.open(pdf_path)
        except Exception as e:
            logger.warning(f"TRIAGE: Could not open '{pdf_path.name}': {e}.")
            return OcrRequirement.EMPTY_OR_CORRUPT

        if doc.page_count == 0:
            logger.info(f"TRIAGE: '{pdf_path.name}' is empty.")
            doc.close()
            return OcrRequirement.EMPTY_OR_CORRUPT

        textual_pages_count = 0
        try:
            for page in doc:
                # get_text() is very fast and efficient for this check.
                if len(page.get_text()) > MIN_CHARS_FOR_TEXTUAL_PAGE:
                    textual_pages_count += 1
        except Exception as e:
            logger.error(f"TRIAGE: Failed during page analysis of '{pdf_path.name}': {e}")
            doc.close()
            return OcrRequirement.EMPTY_OR_CORRUPT

        ### Classification Logic ### 
        percentage_textual = textual_pages_count / doc.page_count
        
        doc.close()
        
        # If 95% or more of the pages are textual, we can safely skip OCR.
        # This assumes that for documents with fewer than 5% scanned pages,
        # the loss of that minor content is acceptable for the massive speed gain.
        # If ALL content must be captured, every mixed-mode file needs OCR.

        if percentage_textual >= TEXTUAL_PAGE_PERCENTAGE_THRESHOLD:
            logger.info(f"TRIAGE: '{pdf_path.name}' is predominantly text ({percentage_textual:.0%}). OCR not required.") #:.0% formats
            return OcrRequirement.OCR_NOT_REQUIRED
        else:
            logger.info(f"TRIAGE: '{pdf_path.name}' is not sufficiently text-based ({percentage_textual:.0%}). OCR required.")
            return OcrRequirement.OCR_REQUIRED
        
