import pymupdf
import pathlib
import shutil
import argparse
import logging
import sys


### Configuration Constants ###
# These thresholds can be tuned to adjust the sorting behavior.

# CATEGORY 1: Minimal Text, Image-Heavy
# If a PDF's total character count is below this, it's considered to have "minimal text".
OVERALL_MIN_CHARS_THRESHOLD = 500
# The minimum percentage of total page area that must be covered by images to qualify.
MIN_OVERALL_IMAGE_COVERAGE_FOR_CAT1 = 0.10  # 10%


# CATEGORY 2: Native Text with some Image-Heavy Pages (e.g., textbooks, reports)
# A single page is "image-heavy" if its image area exceeds this percentage.
MIN_PAGE_IMAGE_COVERAGE_FOR_CAT2 = 0.50  # 50%


# CATEGORY 3: Predominantly Textual
# A page is considered "textual" if its character count is above this value.
MIN_CHARS_FOR_TEXTUAL_PAGE = 150
# The minimum percentage of pages that must be "textual" to classify the whole PDF.
TEXTUAL_PAGE_PERCENTAGE_FOR_CAT3 = 0.90  # 90%

### Output Folder Names ###
folder_categories = {
    "CAT1_FOLDER_NAME": "1_MinimalText_ImageHeavy",
    "CAT2_FOLDER_NAME": "2_NativeText_SomeImageHeavyPages",
    "CAT3_FOLDER_NAME": "3_PredominantlyTextual",
    "CAT4_OTHER_FOLDER_NAME": "4_Other_Or_Balanced",
    "CAT5_ERROR_FOLDER_NAME": "5_Errors_Or_Empty"
}

CAT5_ERROR_FOLDER_NAME = folder_categories['CAT5_ERROR_FOLDER_NAME']

category_folders_list = list(folder_categories.values())


### Logging Setup ###
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s]: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)



def analyze_pdf(pdf_path: pathlib.Path) -> str:
    """
    Analyzes a single PDF to determine its category based on text and image content.

    Args:
        pdf_path [Path]: The file path of the PDF to analyze.

    Returns:
        The string name of the category folder for the PDF.
    """

    try:
        doc = pymupdf.open(pdf_path)
    
    except Exception as e:
        logger.warning(f'Could not open or process {pdf_path.name}: {e} \n Moving to "{CAT5_ERROR_FOLDER_NAME}".')
        return CAT5_ERROR_FOLDER_NAME

    if doc.page_count == 0:
        logger.info(f'{pdf_path.name} contains zero pages. Moving to "{CAT5_ERROR_FOLDER_NAME}".')
        return CAT5_ERROR_FOLDER_NAME
    
    ### Initialize values 

    total_chars = 0

    total_pages_area = 0
    total_image_area_on_pages = 0
    
    textual_pages_count = 0
    image_heavy_pages_count = 0


    ### ANALYSIS START

    for page in doc:
        try:
            ### Page and Text Analysis
            page_area = page.rect.width * page.rect.height
            if page_area == 0:
                continue # skip pages w no area

            total_pages_area += page_area
            page_text = page.get_text("text") # "text" argument specifies that you want raw text content—without any formatting, images, or layout preservation.
            
            page_char_count = len(page_text)
            total_chars += page_char_count

            if page_char_count > MIN_CHARS_FOR_TEXTUAL_PAGE:
                textual_pages_count += 1

            ### Image Analysis (calculating displayed area)

            current_page_image_area = 0
            for img in page.get_images(full=True):
                rects = page.get_image_rects(img)
                for r in rects:
                    current_page_image_area += abs(r) # Equivalent to r.width * r.height

            total_image_area_on_pages += current_page_image_area

            ### Check if this specific page is image-heavy
            page_image_coverage = current_page_image_area / page_area if page_area > 0 else 0
            if page_image_coverage > MIN_PAGE_IMAGE_COVERAGE_FOR_CAT2:
                image_heavy_pages_count += 1

        except Exception as e:
            logger.warning(f"Error processing page {page.number} of '{pdf_path.name}': {e}. Skipping page analysis.")
            continue  # Skip this page if there's an error
        
        doc.close()

    ### Calculate OVerall Document Metrics

    effective_page_count = doc.page_count or 1 # fallback value to prevent division by zero error
    effective_total_area = total_pages_area or 1
    

    has_minimal_text = total_chars < OVERALL_MIN_CHARS_THRESHOLD
    overall_image_coverage = total_image_area_on_pages / effective_total_area
    
    percentage_textual_pages = textual_pages_count / effective_page_count


    # is_textual = textual_page_count > (effective_page_count/TEXTUAL_PAGE_PERCENTAGE_FOR_CAT3)


    ### Classification Logic 
    # The order of these checks is important. A PDF is assigned to the first category it qualifies for.

    # CATEGORY 1: Image-Heavy with minimal text
    if has_minimal_text and overall_image_coverage > MIN_OVERALL_IMAGE_COVERAGE_FOR_CAT1:
        return folder_categories['CAT1_FOLDER_NAME']
    
    # !!! Category 3 goes before category 2

    # CATEGORY 3: More than 90% of the pages in the document are "textual".
    # (A single page is considered "textual" if it contains more than 150 characters.)
    # (Checked before Cat 2 because being mostly text is a stronger classifier)
    if percentage_textual_pages > TEXTUAL_PAGE_PERCENTAGE_FOR_CAT3:
        return folder_categories['CAT3_FOLDER_NAME']


    # CATEGORY 2: Native Text with some Image-Heavy Pages
    if not has_minimal_text and (image_heavy_pages_count > 2):
        return folder_categories['CAT2_FOLDER_NAME']
    
    # CATEGORY 4: If the PDF doesn't fit any of the above, it's considered balanced or other.
    return folder_categories['CAT4_OTHER_FOLDER_NAME']



def create_directories(output_path: pathlib.Path, folder_names: list[str] = category_folders_list):
    '''
    Creates the output folders/directories.

    Args:
    1. output_path [pathlib.Path object]: The path where the directories/folders will be created.
    
    2. folder_names [list [str]]:   a list containing the folder/directory names for categorizing.
                                    Defaults to category_folders_list from this 

    '''
    logger.info(f'Ensuring output folder exists...')

    if output_path.is_dir():
        logger.info(f'Creating folders/directories in {output_path}.')
    else:
        logger.info(f'"{output_path}" does not exist. Making parent directory...')
        
        try:
            for folder_name in folder_names:
                (output_path/folder_name).mkdir(parents=True, exist_ok=True)

        except Exception as e:
            logger.error(f"Fatal: could not create directory {output_path}: {e}")
            sys.exit(1)

def process_files (input_path, output_path):
    '''
    Finds, analyzes, and moves all PDF files from the input path
    to the categorized output folders.
    '''
    
    try:
        input_path = pathlib.Path(input_path)
        output_path = pathlib.Path(output_path)

    except Exception as e:
        logger.error(f'Invalid arguments: {e}')
        return

    ### Validate if valid input path
   
    try:
        if input_path.is_dir():
            logger.info(f'Found directory {input_path}.')
        else:
            raise NotADirectoryError(f"{input_path} is not a directory.")
    except NotADirectoryError as e:
        logger.error(f"Error: {e}")
        # initialize pdf_files to empty list
        pdf_files = []
        return


    ### Collect PDF files from input directory

    pdf_files = [file for file in input_path.iterdir() if 
                 file.is_file() and file.suffix.lower() == ".pdf"]
    if not pdf_files:
        logger.info(f"No PDF files found in '{input_path}'. Nothing to do.")
        return
    
    logger.info(f"Found {len(pdf_files)} PDF files to process.")
    

    ### Collect Categories from initialized category names located at beginning of this module.

    category_folders = list(folder_categories.values())
    create_directories(output_path, category_folders)

    # Initialize summary counter
    summary = {category_name: 0 for category_name in category_folders}

    ### Iterate through the PDF files

    for pdf_file in pdf_files:
        logger.info(f'Analyzing "{pdf_file.name}"...')
        category = analyze_pdf(pdf_file)
        summary[category] += 1

        ### Moving mechanism
        destination_folder = output_path/category

        try:
            # The destination path must not already exist.
            shutil.move(pdf_file,destination_folder)
            logger.info(f"--> Moved '{pdf_file.name}' to '{category}'")


        except Exception as e:
            logger.error(f'Could not move "{pdf_file.name}" to "{destination_folder}": {e}. \n\nMoving on to next file.')
            continue
    
    ### Print final summary ###

    logger.info('\n\n\n--- Processing Summary ---\n')
    logger.info(f'Total PDF files processed: {len(pdf_files)}')
    for category_name, count in summary.items():
        logger.info(f'- {category_name} : {count}')
    logger.info('\n--Script Finished--\n')

    def main():
        '''
        Main function to parse arguments and start the triage process.
        '''

        parser = argparse.ArgumentParser(
            description='Sorts PDF files from an input folder into categorized output folders based on their text and image content.'
        )

        parser.add_argument(
            'input_folder',
            type='str',
            help= 'Path to the folder containing PDF files to sort.'
        )

        parser.add_argument(
            'output_folder',
            type='str',
            help= 'Path to the parent folder where categorized subfolders will be created.'
        )

        args = parser.parse_args()

        input_path = args.input_folder
        output_path = args.output_folder

        process_files(input_path, output_path)

if __name__ == '__main__':
    main()


    
