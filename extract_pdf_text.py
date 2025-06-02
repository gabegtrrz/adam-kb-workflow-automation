import pymupdf
from PIL import Image
import pytesseract
import io
import argparse
import os

def extract_text_from_pdf(pdf_path, tesseract_cmd_path=None):
    """
    Extracts text from a PDF file.
    This includes text directly embedded in the PDF and text from images within the PDF using OCR.

    Args:
        pdf_path (str): The path to the PDF file.
        tesseract_cmd_path (str, optional): The path to the Tesseract OCR executable.
                                            Defaults to None (assumes Tesseract is in PATH).

    Returns:
        str: The combined extracted text from the PDF.
             Returns an error message string if an issue occurs.
    """
    all_extracted_text = []

    # --- Tesseract OCR Configuration ---
    # Setting the path to the executable.

    # If you don't have tesseract executable in your PATH, include the following:
    # pytesseract.pytesseract.tesseract_cmd = r'C:\Program Fideactles\Tesseract-OCR\tesseract.exe'

    if tesseract_cmd_path:
        if os.path.exists(tesseract_cmd_path):
            # specifies the path to the Tesseract OCR executable
            pytesseract.pytesseract.tesseract_cmd = tesseract_cmd_path

        else:
            print(f"Warning: Provided Tesseract path '{tesseract_cmd_path}' does not exist. Trying system PATH.")


    # --- PDF Processing ---
    try:
        doc = pymupdf.open(pdf_path)
    except Exception as e:
        return f"Error: Could not open or read PDF file '{pdf_path}'.\nDetails: {e}"

    print(f"Processing PDF: {pdf_path} ({doc.page_count} pages)")

    for page_num in range(doc.page_count):
        page = doc.load_page(page_num)
        page_text_content = ""

        # 1. Extract text directly from the page
        try:
            direct_text = page.get_text("text")
            if direct_text.strip():
                page_text_content += f"\n--- Page {page_num + 1}: Direct Text ---\n"
                page_text_content += direct_text
        except Exception as e:
            print(f"Warning: Could not extract direct text from page {page_num + 1}. Details: {e}")


        # 2. Extract text from images on the page using OCR
        image_list = page.get_images(full=True)
        if image_list:
            page_text_content += f"\n--- Page {page_num + 1}: Image OCR Text ---\n"
            for img_index, img_info in enumerate(image_list):
                xref = img_info[0]  # XREF of the image
                try:
                    base_image = doc.extract_image(xref)
                    image_bytes = base_image["image"]
                    # image_ext = base_image["ext"] # e.g., "png", "jpeg"

                    # Open image using Pillow
                    pil_image = Image.open(io.BytesIO(image_bytes))

                    # Perform OCR
                    # You can specify language e.g., lang='eng+fra' for English and French
                    # By default, it uses English ('eng')
                    ocr_text = pytesseract.image_to_string(pil_image)

                    if ocr_text.strip():
                        page_text_content += f"\n[Image {img_index + 1} on Page {page_num + 1}]\n{ocr_text.strip()}\n"
                    else:
                        page_text_content += f"\n[Image {img_index + 1} on Page {page_num + 1}: No text found by OCR or image is empty]\n"

                except pytesseract.TesseractNotFoundError:
                    error_msg = (
                        "TesseractNotFoundError: Tesseract is not installed or not found in your PATH.\n"
                        "Please install Tesseract OCR engine and ensure it's in your system's PATH,\n"
                        "or provide the correct path to tesseract.exe (or tesseract executable) "
                        "using the --tesseract_path argument or by setting "
                        "pytesseract.pytesseract.tesseract_cmd in the script."
                    )
                    print(error_msg)
                    return error_msg # Stop further processing if Tesseract is not found
                except Exception as e:
                    print(f"Warning: Could not process image {img_index + 1} on page {page_num + 1}. Details: {e}")
                    page_text_content += f"\n[Image {img_index + 1} on Page {page_num + 1}: Error during processing - {e}]\n"
        
        if page_text_content.strip(): # Add page content only if there's something
             all_extracted_text.append(page_text_content)


    doc.close()
    
    if not all_extracted_text:
        return "No text content (direct or from images) could be extracted from the PDF."
        
    return "\n".join(all_extracted_text)

def main():
    parser = argparse.ArgumentParser(
        description="Extract text from a PDF file, including text from images using OCR.",
        formatter_class=argparse.RawTextHelpFormatter # For better help text formatting
    )
    parser.add_argument("pdf_file", help="Path to the PDF file to process.")
    parser.add_argument(
        "--tesseract_path",
        help=(
            "Optional: Full path to the Tesseract OCR executable.\n"
            "e.g., C:\\Program Files\\Tesseract-OCR\\tesseract.exe (Windows)\n"
            "or /usr/bin/tesseract (Linux/macOS)\n"
            "Use this if Tesseract is not in your system PATH."
        ),
        default=None
    )
    parser.add_argument(
        "--output_file",
        help="Optional: Path to a file where the extracted text will be saved.",
        default=None
    )

    args = parser.parse_args()

    if not os.path.exists(args.pdf_file):
        print(f"Error: The PDF file '{args.pdf_file}' was not found.")
        return

    print("Starting text extraction process...")
    extracted_text = extract_text_from_pdf(args.pdf_file, args.tesseract_path)

    if args.output_file:
        try:
            with open(args.output_file, "w", encoding="utf-8") as f:
                f.write(extracted_text)
            print(f"\nExtraction complete. Text saved to: {args.output_file}")
        except Exception as e:
            print(f"\nError saving text to file '{args.output_file}'. Details: {e}")
            print("\nExtracted Text:\n--------------------")
            print(extracted_text) # Print to console if saving fails
    else:
        print("\nExtraction complete. Extracted Text:\n--------------------")
        print(extracted_text)

if __name__ == "__main__":
    # main()

    # --- Test Tesseract --- #
    # print(pytesseract.get_languages(config=''))
