import logging
import sys
from pathlib import Path
from multiprocessing import Pool, cpu_count
import argparse

import ocrmypdf
import ocrmypdf.exceptions

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
            force_ocr (bool): Whether to force OCR even if text is present.
            language (str): Language(s) for OCR.
            deskew (bool): Whether to deskew the image before OCR.
        '''
        
        self.force_ocr = force_ocr
        self.language = language
        self.deskew = deskew
        self.skip_text = skip_text
        self.redo_ocr = redo_ocr


    def process_file(self, input_path_str: str, output_path_str: str = ""):
        '''
        Worker function to process a single PDF file.

        Args:
            input_path (str): Path to the input PDF.
            output_path (str): Path to save the output searchable PDF.

        Returns:
            dict: A dictionary containing the processing status and file paths.
        '''

        # Converts path strings into Path objects
        input_path=Path(input_path_str)
        if output_path_str != "":
            output_path = Path(output_path_str)
        else:
            # If no output path is provided, create a default output path
            output_path = input_path.parent / f"[OCR] {input_path.name}"

        try:
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
    A class to manage batch OCR processing for a folder of PDFs.
    This class handles finding files, setting up multiprocessing, and reporting.
    '''
    
    def __init__(self, input_folder: str, force_ocr: bool = False, language: str = 'eng', skip_text=False, redo_ocr=True, workers: int = -1, deskew: bool=False):
        '''
        Initializes the batch runner.

        Args:
            1.  input_folder (str): Path to the folder containing PDFs.
            2.  force_ocr (bool): Force OCR on all files.
            3.  language (str): Language for OCR.
            workers (int): Number of parallel processes.
                Defaults to = cpu_count() - 2.
        '''
        self.input_folder_path = Path(input_folder)
        self.force_ocr = force_ocr
        self.language = language
        self.skip_text = skip_text
        self.redo_ocr = redo_ocr


        if workers == -1:
            self.num_workers = max(1, cpu_count() - 2)
        elif workers > 0:
            self.num_workers = workers
        else:
            logger.error('Workers argument cannot be less than 1. Setting workers to 1.')
            self.num_workers = 1
        
        
    def _prepare_tasks(self, output_folder_path: Path) -> list[tuple[OcrProcessor, Path, Path]]:
        '''Prepares the list of tasks for multiprocessing.'''

        # output_folder_path = Path(output_folder)

        pdf_files = [p for p in self.input_folder_path.iterdir() if p.is_file() and p.suffix.lower() == '.pdf']

        if not pdf_files:
            logger.info("No PDF files found in the input folder.")
            return []

        logger.info(f'Found {len(pdf_files)} PDF files to process.')

        # We instantiate the processor here to pass its method to the pool
        processor = OcrProcessor(
            self.force_ocr,
            self.language,
            skip_text = self.skip_text,
            redo_ocr = self.redo_ocr
            )


        tasks = []
        for pdf_path in pdf_files:
            ### Name output path for each input pdf
            output_pdf_name = f'[OCR] {pdf_path.name}'
            output_pdf_full_path = output_folder_path / output_pdf_name

            tasks.append((processor, pdf_path, output_pdf_full_path))
            
        return tasks
    
    def run_batch(self):
        '''
        Executes the batch OCR process on the entire folder.
        '''
        if not self.input_folder_path.is_dir():
            logger.error(f"Error: The provided path '{self.input_folder_path}' is not a valid directory.")
            return

        # Create the output folder
        output_folder_path = self.input_folder_path / "OCRed_PDFs"
        try:
            output_folder_path.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            logger.error(f'Could not create output folder "{output_folder_path}". Error: {e}')
            return

        tasks = self._prepare_tasks(output_folder_path)
        if not tasks:
            return

        logger.info(f"Starting OCR processing using {self.num_workers} parallel processes.")
        logger.info(f"OCR Language: '{self.language}', Force OCR: {self.force_ocr}")

        with Pool(processes=self.num_workers) as pool:
            results = pool.starmap(BatchOCRRunner.worker_adapter, tasks)

        self._log_summary(results, output_folder_path)

### Helper function to unpack arguments for starmap. ###
# This does the actual processing. Crucially, it calls `process_file()` function.
    @staticmethod
    def worker_adapter(processor: OcrProcessor, input_p, output_p):
        return processor.process_file(input_p, output_p)

    def _log_summary(self, results: list, output_folder: Path):
        """Logs a summary of the batch processing results."""
        successful_count = sum(1 for res in results if res['status'] == 'success')
        failed_files = [res for res in results if res['status'] == 'error']

        logger.info("\n\n--- OCR Processing Summary ---\n")
        logger.info(f"Output folder: {output_folder}")
        logger.info(f"Total PDF files found: {len(results)}")
        logger.info(f"Successfully Processed: {successful_count} file(s).")

        if failed_files:
            logger.warning(f"Failed to OCR: {len(failed_files)} file(s). Details below:")
            for info in failed_files:
                logger.warning(f"- File: {Path(info['input_file']).name}, Error: {info['error']}")
        else:
            logger.info("All found PDF files processed successfully!")
        logger.info("Script finished.")

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