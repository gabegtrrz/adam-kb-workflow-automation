import pymupdf
import pathlib
import os
import shutil
import argparse
import logging
import sys

# --- Configuration Constants (tune these thresholds as needed) ---

# - For Category 1: MinimalText_ImageHeavy -

# If total characters in the *PDF* are less than this, it's considered "minimal text".
OVERALL_MIN_CHARS_THRESHOLD = 500
# If the total area of images across all pages is >10% of total PDF area.
MIN_OVERALL_IMAGE_COVERAGE_FOR_CAT1 = 0.10

# -For Category 2: NativeText_SomeImageHeavyPages -

# If >50% of a single page's area is covered by images.
MIN_PAGE_IMAGE_COVERAGE_FOR_CAT2 = 0.50



# - For Category 3: PredominantlyTextual -

# A page is considered "textual" if its character count exceeds this.
MIN_CHARS_FOR_TEXTUAL_PAGE = 50
# If >90% of pages in the PDF are "textual".
TEXTUAL_PAGE_PERCENTAGE_FOR_CAT3 = 0.90



# --- Output Folder Names ---
CAT1_FOLDER_NAME = "1_MinimalText_ImageHeavy"
CAT2_FOLDER_NAME = "2_NativeText_SomeImageHeavyPages"
CAT3_FOLDER_NAME = "3_PredominantlyTextual"
CAT4_OTHER_FOLDER_NAME = "4_Other_Or_Balanced"


# --- Logging Setup ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s]: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)



# worker function

def analyze_pdf(pdf_path: pathlib.Path) -> str:
    """
    Analyzes a PDF file to categorize it based on text and image content.

    Args:
        pdf_path: Path to the PDF file.

    Returns:
        The name of the category folder for this PDF.
    """
    try:
        doc = pymupdf.open(pdf_path)
    except Exception as e:
        logger.warning(f"Could not open or process '{pdf_path.name}': {e}. Moving to '{CAT4_OTHER_FOLDER_NAME}'.")
        return CAT4_OTHER_FOLDER_NAME

    if doc.page_count == 0:
        logger.info(f"'{pdf_path.name}' has 0 pages. Moving to '{CAT4_OTHER_FOLDER_NAME}'.")
        doc.close()
        return CAT4_OTHER_FOLDER_NAME

    total_chars = 0
    total_pages_area = 0
    total_image_area_on_pages = 0  # Sum of displayed image areas
    
    textual_pages_count = 0
    count_pages_over_50_percent_image = 0

    for page_num in range(doc.page_count):
        try:
            page = doc.load_page(page_num)
            page_rect = page.rect
            current_page_area = page_rect.width * page_rect.height
            if current_page_area == 0: # Skip if page area is zero
                continue
            total_pages_area += current_page_area

            # --- Text Analysis ---
            text = page.get_text("text")
            page_chars = len(text)
            total_chars += page_chars
            if page_chars > MIN_CHARS_FOR_TEXTUAL_PAGE:
                textual_pages_count += 1

            # --- Image Analysis (displayed area) ---
            current_page_image_area = 0
            image_list = page.get_images(full=True)
            for img_info in image_list:
                # Get the rectangles where this image is displayed on the page
                rects = page.get_image_rects(img_info)
                for r in rects:
                    if r.is_valid and not r.is_empty:
                        current_page_image_area += r.width * r.height
            
            total_image_area_on_pages += current_page_image_area

            if (current_page_image_area / current_page_area) > MIN_PAGE_IMAGE_COVERAGE_FOR_CAT2:
                count_pages_over_50_percent_image += 1
        
        except Exception as e:
            logger.warning(f"Error processing page {page_num} of '{pdf_path.name}': {e}. Skipping page analysis.")
            continue # Skip this page if there's an error

    

    ### --- Calculate Overall Metrics --- ####

    has_minimal_text = total_chars < OVERALL_MIN_CHARS_THRESHOLD
    
    pdf_overall_image_coverage = 0
    if total_pages_area > 0:
        pdf_overall_image_coverage = total_image_area_on_pages / total_pages_area
    
    percentage_textual_pages = 0
    if doc.page_count > 0:
        percentage_textual_pages = textual_pages_count / doc.page_count

    doc.close()

    ### --- Classification Logic (Hierarchical) --- ###

    # 1. Category 1: MinimalText_ImageHeavy
    if has_minimal_text and pdf_overall_image_coverage > MIN_OVERALL_IMAGE_COVERAGE_FOR_CAT1:
        return CAT1_FOLDER_NAME

    # 2. Category 3: PredominantlyTextual
    #    (Checked before Cat 2 because being overwhelmingly textual is a strong characteristic)
    if percentage_textual_pages > TEXTUAL_PAGE_PERCENTAGE_FOR_CAT3:
        return CAT3_FOLDER_NAME
        
    # 3. Category 2: NativeText_SomeImageHeavyPages
    #    (Implies not minimal text, and not overwhelmingly textual by page count)
    if not has_minimal_text and count_pages_over_50_percent_image > 0:
        return CAT2_FOLDER_NAME

    # 4. Default: Other_Or_Balanced
    return CAT4_OTHER_FOLDER_NAME


def categorize(input_path: pathlib.Path, output_path: pathlib.Path):

    # --- Create Output Directories ---
    category_folders = {
        CAT1_FOLDER_NAME: output_path / CAT1_FOLDER_NAME,
        CAT2_FOLDER_NAME: output_path / CAT2_FOLDER_NAME,
        CAT3_FOLDER_NAME: output_path / CAT3_FOLDER_NAME,
        CAT4_OTHER_FOLDER_NAME: output_path / CAT4_OTHER_FOLDER_NAME,
    }

    for folder in category_folders.values():
        try:
            folder.mkdir(parents=True, exist_ok=True)
        except OSError as e:
            logger.error(f"Error creating directory {folder}: {e}")
            sys.exit(1)
            
    logger.info(f"Output folders will be created in: {output_path.resolve()}")

    # --- Process PDF Files ---
    # pdf_files = list(input_path.glob("*.pdf"))
    pdf_files = [file for file in input_path.iterdir() if file.is_file() and file.suffix.lower() == ".pdf"]
    if not pdf_files:
        logger.info(f"No PDF files found in '{input_path}'.")
        sys.exit(0)

    logger.info(f"Found {len(pdf_files)} PDF files to process.")
    
    summary = {name: 0 for name in category_folders.keys()}

    for pdf_file in pdf_files:
        logger.info(f"Analyzing '{pdf_file.name}'...")
        category = analyze_pdf(pdf_file)
        
        target_folder = category_folders[category]
        destination_path = target_folder / pdf_file.name
        
        try:
            shutil.move(str(pdf_file), str(destination_path))
            logger.info(f"Moved '{pdf_file.name}' to '{category}'.")
            summary[category] += 1
        except Exception as e:
            logger.error(f"Could not move '{pdf_file.name}' to '{destination_path}': {e}")

    # --- Print Summary ---
    logger.info("\n--- Processing Summary ---")
    logger.info(f"Total PDF files processed: {len(pdf_files)}")
    for category_name, count in summary.items():
        logger.info(f"Files moved to '{category_name}': {count}")
    logger.info("Script finished.")

def main():
    parser = argparse.ArgumentParser(
        description="Sorts PDF files into categories based on text and image content."
    )
    parser.add_argument(
        "input_folder", type=str, help="Path to the folder containing PDF files to sort."
    )
    parser.add_argument(
        "output_folder", type=str, help="Path to the parent folder where subfolders will be created."
    )
    args = parser.parse_args()

    input_path = pathlib.Path(args.input_folder)
    output_path = pathlib.Path(args.output_folder)

    if not input_path.is_dir():
        logger.error(f"Error: Input folder '{input_path}' not found or is not a directory.")
        sys.exit(1)

    if not output_path.is_dir():
        logger.error(f"Error: Output folder '{input_path}' not found or is not a directory.")
        sys.exit(1)

    categorize(input_path, output_path)

if __name__ == "__main__":
    main()