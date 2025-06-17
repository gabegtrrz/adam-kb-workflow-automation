import logging
import sys
from pathlib import Path
from multiprocessing import Pool, cpu_count
import argparse
from enum import Enum, auto

import ocrmypdf
import ocrmypdf.exceptions

from triage import PdfTriage, OcrRequirement
from file_operations import FileMover

logger = logging.getLogger(__name__)

class OcrProcessor:
    '''
    A class to perform OCR on a single PDF file using OCRmyPDF.
    This class encapsulates the logic for processing one file.
    '''
    def __init__(self, force_ocr: bool=False, language: str='eng', skip_text=False, redo_ocr=True, deskew: bool=False):
        '''
        Initializes the OcrProcessor with specific settings.

        Args:
            1. force_ocr (bool): Force OCR on all files.
            2. language (str): Language for OCR.
            3. skip_text (bool): Skip OCR on PDFs that already contain text layers.
            4. redo_ocr (bool): If True, analyzes text and does OCR ONLY on images, preserving native text. Defaults to True.
            5. deskew (bool): Deskew pages before OCR. Defaults to False.
        '''
        
        self.force_ocr = force_ocr
        self.language = language
        self.deskew = deskew
        self.skip_text = skip_text
        self.redo_ocr = redo_ocr


    def process_file(self, input_path, output_path= "") -> dict:
        '''
        Worker function to process a single PDF file.

        Args:
            input_path (str): Path to the input PDF.
            output_path (str): Path to save the output searchable PDF.

        Returns:
            dict: A dictionary containing the processing status and file paths.
        '''

        # Converts path strings into Path objects
        input_path=Path(input_path)
        if output_path != "":
            output_path = Path(output_path)
        else:
            # If no output path is provided, create a default output path
            output_path = input_path.parent / f"[OCR] {input_path.name}"

        try:
            # Ensure the output directory exists right before processing
            output_path.parent.mkdir(parents=True, exist_ok=True)


            ocrmypdf.ocr(
                input_file=input_path,
                output_file=output_path,
                force_ocr=self.force_ocr,
                language=self.language,
                progress_bar=True,
                deskew=self.deskew,
                skip_text = self.skip_text,
                redo_ocr = self.redo_ocr
            )
            return {
                'status': 'success',
                'input_file': str(input_path),
                'output_path': str(output_path)
            }

        except ocrmypdf.exceptions.EncryptedPdfError:
            return {
                'status': 'error',
                'input_file': str(input_path),
                'error': "Encrypted PDF - cannot process."
            }
        except ocrmypdf.exceptions.InputFileError as e:
            return {
                'status': 'error',
                'input_file': str(input_path),
                'error': f"Input file error: {e}"
            }
        except Exception as e:
            return {
                'status': 'error',
                'input_file': str(input_path),
                'error': f"Unexpected error: {e}"
            }

class BatchOCRRunner:
    '''
    Manages batch OCR processing with an intelligent triage step
    to process only the necessary files.
    '''

    CATEGORY_FOLDERS = {
        OcrRequirement.OCR_REQUIRED: 'OCRed_Files',
        OcrRequirement.OCR_NOT_REQUIRED: 'Skipped_No_OCR_Required',
        OcrRequirement.EMPTY_OR_CORRUPT: 'Empty_or_Error'
    }

    CATEGORY_COUNT = {
        OcrRequirement.OCR_REQUIRED: 0,
        OcrRequirement.OCR_NOT_REQUIRED: 0,
        OcrRequirement.EMPTY_OR_CORRUPT: 0
    }

    pdfs_found_count = 0

    def __init__(self, input_folder: str, force_ocr: bool = False, language: str = 'eng', skip_text=False, redo_ocr=True, workers: int = -1, deskew: bool=False):
        '''
        Initializes the batch runner
        ---
        Args:
            1. input_folder (str): Path to the folder containing PDFs.
            2. force_ocr (bool): Force OCR on all files.
            3. language (str): Language for OCR.
            4. skip_text (bool): Skip OCR on PDFs that already contain text layers.
            5. redo_ocr (bool): Analyzes text, does OCR ONLY on images, preserving native text. Defaults to True.
            6. workers (int): Number of parallel processes. Defaults to cpu_count() - 2.
            7. deskew (bool): Deskew pages before OCR.
        '''

        ### Initialize directories
        self.input_folder_path = Path(input_folder)
        self.output_folder_path = self.input_folder_path / "OCRed_PDFs"

        ### Initialize Folder Names for Sorting OcrRequirement

        self.OCR_REQUIRED_FOLDER = 'OCRed_Files'
        self.OCR_NOT_REQUIRED_FOLDER = 'Skipped_No_OCR_Required'
        self.EMPTY_OR_CORRUPT_FOLDER = 'Errors_or_Empty'

        ### Initialize OCR Processor Args

        self.force_ocr = force_ocr
        self.language = language
        self.skip_text = skip_text
        self.redo_ocr = redo_ocr
        self.deskew = deskew
        
        # Instantiate helper classes
        self.triage = PdfTriage()
        self.mover = FileMover(base_output_dir=self.output_folder_path)

        if workers == -1:
            self.num_workers = max(1, cpu_count() - 2)
        elif workers > 0:
            self.num_workers = workers
        else:
            logger.error('Workers argument cannot be less than 1. Setting workers to 1.')
            self.num_workers = 1

    
    ### Helper function to unpack arguments for starmap. ###
    # This does the actual processing. Crucially, it calls `process_file()` function.

    @staticmethod
    def worker_adapter(processor: OcrProcessor, input_p, output_p):
        return processor.process_file(input_p, output_p)


    def _get_pdfs(self) -> list:
        """
        Scans the input folder for PDF files and returns a list of their paths.
        Returns:
            list: List of Path objects for PDF files found in the input folder.
        """
        try:
            if not self.input_folder_path.is_dir():
                logger.error(f"Error: The provided path '{self.input_folder_path}' is not a valid directory.")
                return []
            
            pdf_files = [p for p in self.input_folder_path.iterdir() if p.is_file() and p.suffix.lower() == '.pdf']

            if not pdf_files:
                logger.info("No PDF files found in the input folder.")
                return []
        
        except Exception as e:
            logger.error(f"Exception occurred while scanning for PDFs: {e}")
            return []


        self.pdfs_found_count = len(pdf_files)
        logger.info(f'Found {self.pdfs_found_count} PDF files.')

        return pdf_files
    

    def run_batch(self):
        '''
        Executes the intelligent batch OCR process on the entire folder.
        '''
        
        # OUTPUT: Create the output folder
        output_folder_path = self.input_folder_path / "OCRed_PDFs"
        try:
            output_folder_path.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            logger.error(f'Could not create output folder "{output_folder_path}". Error: {e}')
            logger.error('BatchOCRRunner.run_batch execution halted.')
            return
        
        all_pdf_files = self._get_pdfs()
        sorted_pdfs = self._classify_and_sort_pdfs(pdf_files=all_pdf_files, output_folder_path=output_folder_path)

        ### Task creation

        tasks_for_ocr = self._prepare_tasks(pdf_files=sorted_pdfs, output_folder_path=output_folder_path)
        
    
        logger.info(f"{len(tasks_for_ocr)} files require OCR. Starting parallel processing...")
        logger.info("---------------------")
        logger.info(f"Using {self.num_workers} parallel processes.")
        logger.info(f"OCR Language: '{self.language}'")
        logger.info(f"Force OCR: {self.force_ocr}")
        logger.info(f"Skip Text: {self.skip_text}")
        logger.info(f"Redo OCR: {self.redo_ocr}")
        logger.info(f"Deskew: {self.deskew}")
        logger.info("---------------------")


        ### Begin Worker Multiprocessing

        with Pool(processes=self.num_workers) as pool:
            results = pool.starmap(BatchOCRRunner.worker_adapter, tasks_for_ocr)

        self._log_summary(results, output_folder_path)


    def _classify_and_sort_pdfs(self, pdf_files: list, output_folder_path):
        """
        Classifies a list of PDF files based on their OCR requirements and sorts them into appropriate folders.

        Args:
            pdf_files (list): List of paths to PDF files to be classified.
            output_folder_path (str or Path): Path to the output folder where files will be sorted.
        Returns:
            list: List of PDF file paths that require OCR processing.
        Side Effects:
            - Moves files that do not require OCR or are empty/corrupt to designated folders.
            - Logs the count of files in each classification category.
        Classification Categories:
            - OCR_REQUIRED: Files that require OCR processing (returned in the list).
            - OCR_NOT_REQUIRED: Files that do not require OCR (moved to a specific folder).
            - EMPTY_OR_CORRUPT: Files that are empty or corrupt (moved to a specific folder).
        """
        output_folder_path = Path(output_folder_path)

        ocr_required = []

        ### Iterate through each PDF
        for pdf_path in pdf_files:

            ### Classify PDF if it needs OCR
            ocr_decision = self.triage.classify(pdf_path)

            if ocr_decision == OcrRequirement.OCR_REQUIRED:
                ocr_required.append(pdf_path)

                self.CATEGORY_COUNT[OcrRequirement.OCR_REQUIRED] += 1

            elif ocr_decision == OcrRequirement.OCR_NOT_REQUIRED:
                self.mover.move_file(source_path=pdf_path, destination_folder_name=self.CATEGORY_FOLDERS[OcrRequirement.OCR_NOT_REQUIRED])

                self.CATEGORY_COUNT[OcrRequirement.OCR_NOT_REQUIRED] += 1

            elif ocr_decision == OcrRequirement.EMPTY_OR_CORRUPT:
                self.mover.move_file(source_path=pdf_path, destination_folder_name=self.CATEGORY_FOLDERS[OcrRequirement.EMPTY_OR_CORRUPT])

                self.CATEGORY_COUNT[OcrRequirement.EMPTY_OR_CORRUPT] += 1
        
        self.CATEGORY_COUNT
        
        logger.info('Triage complete')
        logger.info(f'{OcrRequirement.OCR_REQUIRED.name} : {self.CATEGORY_COUNT[OcrRequirement.OCR_REQUIRED]} files')
        logger.info(f'{OcrRequirement.OCR_NOT_REQUIRED.name} : {self.CATEGORY_COUNT[OcrRequirement.OCR_NOT_REQUIRED]} files')
        logger.info(f'{OcrRequirement.EMPTY_OR_CORRUPT.name} : {self.CATEGORY_COUNT[OcrRequirement.EMPTY_OR_CORRUPT]} files')

        return ocr_required



    def _prepare_tasks(self, pdf_files:list, output_folder_path) -> list[tuple[OcrProcessor, Path, Path]]:
        '''
        Prepares a list of tasks using multiprocessing.

        Args:
            pdf_files (list): List of PDF file paths to process.
            output_folder_path (str or Path): Path to the output folder where processed files will be saved.
        Returns:
            list[tuple[OcrProcessor, Path, Path]]: 
                A list of tuples, each containing:
                    - OcrProcessor instance configured for OCR processing,
                    - Path to the input PDF file,
                    - Path to the output PDF file for OCR-required files.
        '''

        tasks = []
        processor = OcrProcessor(
            self.force_ocr,
            self.language,
            skip_text = self.skip_text,
            redo_ocr = self.redo_ocr
            )

        output_folder_path = Path(output_folder_path)

        ### Iterate through each PDF

        for pdf_path in pdf_files:
            output_path = self.output_folder_path / self.CATEGORY_FOLDERS[OcrRequirement.OCR_REQUIRED] / f'[OCR] {pdf_path.name}'
            tasks.append((processor, pdf_path, output_path))
            
        return tasks
        

    def _log_summary(self, results: list, output_folder: Path):
        """Logs a summary of the batch processing results."""
        successful_count = sum(1 for res in results if res['status'] == 'success')
        failed_files = [res for res in results if res['status'] == 'error']

        logger.info("\n\n--- OCR Processing Summary ---\n")
        logger.info(f"Output folder: {output_folder}")
        logger.info(f"Total PDF files found: {self.pdfs_found_count}")
        logger.info(f'{OcrRequirement.OCR_REQUIRED.name} : {self.CATEGORY_COUNT[OcrRequirement.OCR_REQUIRED]} files')
        logger.info(f'{OcrRequirement.OCR_NOT_REQUIRED.name} : {self.CATEGORY_COUNT[OcrRequirement.OCR_NOT_REQUIRED]} files')
        logger.info(f'{OcrRequirement.EMPTY_OR_CORRUPT.name} : {self.CATEGORY_COUNT[OcrRequirement.EMPTY_OR_CORRUPT]} files')
        logger.info(f"Successfully Processed: {successful_count} file(s).")
        logger.info("---------------------")
        logger.info(f"Used {self.num_workers} parallel processes.")
        logger.info(f"OCR Language: '{self.language}'")
        logger.info(f"Force OCR: {self.force_ocr}")
        logger.info(f"Skip Text: {self.skip_text}")
        logger.info(f"Redo OCR: {self.redo_ocr}")
        logger.info(f"Deskew: {self.deskew}")
        logger.info("---------------------")

        if failed_files:
            logger.warning(f"Failed to OCR: {len(failed_files)} file(s). Details below:")
            for info in failed_files:
                logger.warning(f"- File: {Path(info['input_file']).name}, Error: {info['error']}")
        else:
            logger.info("All found PDF files processed successfully!")
        logger.info("Script finished.\n")

def main_cli():
    '''
    Function to handle command-line arguments and run the batch processor.
    '''

    parser = argparse.ArgumentParser(
        description="Batch OCR PDF files in a folder using OCRmyPDF with multicore processing."
    )

    parser.add_argument(
        "input_folder", type=str, help="Path to the folder containing PDF files."
    )

    parser.add_argument(
        "--force_ocr",
        action="store_true",
        help="Force OCR even if the document appears to have text. Default is False."
    )

    parser.add_argument(
        "--skip_text",
        action="store_true",
        help="Skips OCR on PDFs that already contain text layers. Default is False."
    )

    parser.add_argument(
        "--redo_ocr",
        action="store_false",
        help="Add argument to disable. Default = True: Forces OCR to be performed, even if a text layer already exists. This can be useful for correcting errors or improving text quality."
    )

    parser.add_argument(
        "--workers",
        type=int,
        default=-1, # Use -1 to signal using the class default
        help=f"Set the number of parallel worker processes. Default: system cores minus 2."
    )
    parser.add_argument(
        "--language",
        type=str,
        default="eng",
        help="Specify the OCR language (e.g., eng, fil). Default: eng"
    )

    parser.add_argument(
        "--deskew",
        action="store_true",
        help="Automatically deskews (corrects the rotation of) each page before performing OCR.  Improves accuracy, but may increase processing time. Default is False."
    )

    args = parser.parse_args()

    # Configure root logger
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s]: %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    # Instantiate and run the batch processor from the CLI
    runner = BatchOCRRunner(
        input_folder=args.input_folder,
        force_ocr=args.force_ocr,
        language=args.language,
        workers=args.workers,
        skip_text = args.skip_text,
        redo_ocr = args.redo_ocr,
        deskew = args.deskew
    )
    
    runner.run_batch()

if __name__ == "__main__":
    main_cli()