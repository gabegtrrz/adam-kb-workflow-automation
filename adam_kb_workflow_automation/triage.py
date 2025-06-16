import pymupdf
import logging
from pathlib import Path
from enum import Enum, auto

### Module-level Configuration
# These thresholds can be tuned from a central config file in a future version.
MIN_CHARS_FOR_TEXTUAL_PAGE = 150
TEXTUAL_PAGE_PERCENTAGE_THRESHOLD = 0.95  # 95%

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
        
        doc.close()

        ### Classification Logic ### 
        percentage_textual = textual_pages_count / doc.page_count
        
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
