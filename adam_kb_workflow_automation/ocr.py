import logging
import sys
from pathlib import Path
from multiprocessing import Pool, cpu_count
import argparse
from datetime import datetime

import ocrmypdf
import ocrmypdf.exceptions

from triage import PdfTriage, OcrRequirement
from file_operations import FileOps

from tqdm import tqdm

logger = logging.getLogger(__name__)

class OcrProcessor:
    '''
    A class to perform OCR on a single PDF file using OCRmyPDF.
    This class encapsulates the logic for processing one file.
    '''
    def __init__(self, force_ocr: bool=False, language: str='eng', skip_text=False, redo_ocr=True, deskew: bool=False, copy_files: bool = False, **kwargs):
        '''
        Initializes the OcrProcessor with specific settings.

        Args:
            1. force_ocr (bool): Force OCR on all files.
            2. language (str): Language for OCR.
            3. skip_text (bool): Skip OCR on PDFs that already contain text layers.
            4. redo_ocr (bool): If True, analyzes text and does OCR ONLY on images, preserving native text. Defaults to True.
            5. deskew (bool): Deskew pages before OCR. Defaults to False.
            6. copy_files (bool): If True, copies files instead of moving them. Defaults to False (move).
        '''
        
        self.force_ocr = force_ocr
        self.language = language
        self.deskew = deskew
        self.skip_text = skip_text
        self.redo_ocr = redo_ocr
        self.ocr_kwargs = kwargs


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
                progress_bar=False,
                deskew=self.deskew,
                skip_text = self.skip_text,
                redo_ocr = self.redo_ocr,
                **self.ocr_kwargs

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

    ### Initialize Folder Names for Sorting OcrRequirement
    CATEGORY_FOLDERS = {
        OcrRequirement.OCR_REQUIRED: 'Original PDFs',
        OcrRequirement.OCR_NOT_REQUIRED: 'No_OCR_Required',
        OcrRequirement.EMPTY_OR_CORRUPT: 'Empty_or_Error'
    }

    ### Initialize Category Count
    CATEGORY_COUNT = {
        OcrRequirement.OCR_REQUIRED: 0,
        OcrRequirement.OCR_NOT_REQUIRED: 0,
        OcrRequirement.EMPTY_OR_CORRUPT: 0
    }

    pdfs_found_count = 0

    def __init__(self, input_folder: str, force_ocr: bool = False, language: str = 'eng', skip_text=False, redo_ocr=True, workers: int = -1, deskew: bool=False, move_files: bool = False, **kwargs):
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
            8. move_files (bool): If True, moves files instead of copying them. Defaults to False (copy).
        '''

        # Initialize Timestamp
        self.time_started = datetime.now()

        ### Initialize directories
        self.input_folder_path = Path(input_folder)
        self.output_folder_path = self.input_folder_path / f"OCR_Results_{self.time_started.strftime('%Y-%m-%d_%H-%M-%S')}"
        try:
            self.output_folder_path.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            logger.error(f'Could not create output folder "{self.output_folder_path}". Error: {e}')
            logger.error('BatchOCRRunner.run_batch execution halted.')
            return

        ### Initialize OCR Processor Args
        self.force_ocr = force_ocr
        self.language = language
        self.skip_text = skip_text
        self.redo_ocr = redo_ocr
        self.deskew = deskew
        self.move_files = move_files
        self.ocr_kwargs = kwargs
        
        # Instantiate helper classes
        self.triage = PdfTriage()
        self.file_op = FileOps(base_output_dir=self.input_folder_path)

        if workers == -1:
            self.num_workers = max(1, cpu_count() - 2)
        elif workers > 0:
            self.num_workers = workers
        else:
            logger.error('Workers argument cannot be 0 or negative. Setting workers to 1.')
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
        
        # OUTPUT: Create the results output folder
        output_folder_path = self.output_folder_path
        
        all_pdf_files = self._get_pdfs()
        sorted_pdfs = self._classify_and_sort_pdfs(pdf_files=all_pdf_files, output_folder_path=output_folder_path.parent)

        ### Task creation
        
        tasks_for_ocr = self._prepare_tasks(pdf_files=sorted_pdfs, output_folder_path=output_folder_path)
        
    
        logger.info(f"{len(tasks_for_ocr)} files require OCR. Starting parallel processing...")
            
        summary = (
            "\n"
            "------------------------------\n"
            f"Using {self.num_workers} parallel processes.\n"
            f"OCR Language: '{self.language}'\n"
            f"Force OCR: {self.force_ocr}\n"
            f"Skip Text: {self.skip_text}\n"
            f"Redo OCR: {self.redo_ocr}\n"
            f"Deskew: {self.deskew}\n"
            "------------------------------\n"
        )
        
        logger.info(summary)


        ### Begin Worker Multiprocessing

        with Pool(processes=self.num_workers) as pool:
            # 1. pool.starmap returns an iterator over the results.
            # 2. tqdm wraps that iterator to monitor progress.
            # 3. list() consumes the iterator, pulling each result, which drives the progress bar.
            results = list(tqdm(pool.starmap(BatchOCRRunner.worker_adapter, tasks_for_ocr), total=len(tasks_for_ocr)))

            # results = pool.starmap(BatchOCRRunner.worker_adapter, tasks_for_ocr)

        self._log_summary(results, output_folder_path)


    def _classify_and_sort_pdfs(self, pdf_files: list, output_folder_path, is_move_files:bool = False):
        """
        Classifies a list of PDF files based on their OCR requirements and sorts them into appropriate folders.

        Args:
            pdf_files (list): List of paths to PDF files to be classified.
            output_folder_path (str or Path): Path to the output folder where files will be sorted.
        Returns:
            list: List of PDF file paths that require OCR processing.
        Side Effects:
            - Copies files that do not require OCR or are empty/corrupt to designated folders.
            - Logs the count of files in each classification category.
        Classification Categories:
            - OCR_REQUIRED: Files that require OCR processing (returned in the list).
            - OCR_NOT_REQUIRED: Files that do not require OCR (copied to a specific folder).
            - EMPTY_OR_CORRUPT: Files that are empty or corrupt (copied to a specific folder).
        """
        output_folder_path = Path(output_folder_path)

        ocr_required = []

        if is_move_files:

            ### Iterate through each PDF
            for pdf_path in pdf_files:

                ### Classify PDF if it needs OCR
                ocr_decision = self.triage.classify(pdf_path)

                if ocr_decision == OcrRequirement.OCR_REQUIRED:
                    self.file_op.move_file(source_path=pdf_path, destination_folder_name=self.CATEGORY_FOLDERS[OcrRequirement.OCR_REQUIRED])
                    new_path = self.output_folder_path/self.CATEGORY_FOLDERS[OcrRequirement.OCR_REQUIRED]/pdf_path.name
                    ocr_required.append(new_path)

                    self.CATEGORY_COUNT[OcrRequirement.OCR_REQUIRED] += 1

                elif ocr_decision == OcrRequirement.OCR_NOT_REQUIRED:
                    self.file_op.move_file(source_path=pdf_path, destination_folder_name=self.CATEGORY_FOLDERS[OcrRequirement.OCR_NOT_REQUIRED])

                    self.CATEGORY_COUNT[OcrRequirement.OCR_NOT_REQUIRED] += 1

                elif ocr_decision == OcrRequirement.EMPTY_OR_CORRUPT:
                    self.file_op.move_file(source_path=pdf_path, destination_folder_name=self.CATEGORY_FOLDERS[OcrRequirement.EMPTY_OR_CORRUPT])

                    self.CATEGORY_COUNT[OcrRequirement.EMPTY_OR_CORRUPT] += 1
        
        else: # Copy is default

            ### Iterate through each PDF
            for pdf_path in pdf_files:

                ### Classify PDF if it needs OCR
                ocr_decision = self.triage.classify(pdf_path)

                if ocr_decision == OcrRequirement.OCR_REQUIRED:
                    ocr_required.append(pdf_path)
                    self.CATEGORY_COUNT[OcrRequirement.OCR_REQUIRED] += 1

                elif ocr_decision == OcrRequirement.OCR_NOT_REQUIRED:
                    # self.file_op.copy_file(source_path=pdf_path, destination_folder_name=self.CATEGORY_FOLDERS[OcrRequirement.OCR_NOT_REQUIRED])

                    self.CATEGORY_COUNT[OcrRequirement.OCR_NOT_REQUIRED] += 1

                elif ocr_decision == OcrRequirement.EMPTY_OR_CORRUPT:
                    self.file_op.copy_file(source_path=pdf_path, destination_folder_name=self.CATEGORY_FOLDERS[OcrRequirement.EMPTY_OR_CORRUPT])

                    self.CATEGORY_COUNT[OcrRequirement.EMPTY_OR_CORRUPT] += 1
        
        logger.info('Triage complete')

        summary = (
            "------------------------------\n"
            f"Total PDF files found: {self.pdfs_found_count}\n"
            f"{OcrRequirement.OCR_REQUIRED.name} : {self.CATEGORY_COUNT[OcrRequirement.OCR_REQUIRED]} files\n"
            f"{OcrRequirement.OCR_NOT_REQUIRED.name} : {self.CATEGORY_COUNT[OcrRequirement.OCR_NOT_REQUIRED]} files\n"
            f"{OcrRequirement.EMPTY_OR_CORRUPT.name} : {self.CATEGORY_COUNT[OcrRequirement.EMPTY_OR_CORRUPT]} files\n"
        )

        logger.info(summary)

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
            redo_ocr = self.redo_ocr,
            **self.ocr_kwargs
            )

        output_folder_path = Path(output_folder_path)

        ### Iterate through each PDF

        for pdf_path in pdf_files:
            output_path = self.output_folder_path / f'[OCR] {pdf_path.name}'
            tasks.append((processor, pdf_path, output_path))
            
        return tasks
        

    def _log_summary(self, results: list, output_folder: Path):
        """Logs a summary of the batch processing results."""
        successful_count = sum(1 for res in results if res['status'] == 'success')
        failed_files = [res for res in results if res['status'] == 'error']

        summary = (
            "\n\n--- OCR Processing Summary ---\n"
            f"Output folder: {output_folder}\n"
            "------------------------------\n"
            f"Total PDF files found: {self.pdfs_found_count}\n"
            f"{OcrRequirement.OCR_REQUIRED.name} : {self.CATEGORY_COUNT[OcrRequirement.OCR_REQUIRED]} files\n"
            f"{OcrRequirement.OCR_NOT_REQUIRED.name} : {self.CATEGORY_COUNT[OcrRequirement.OCR_NOT_REQUIRED]} files\n"
            f"{OcrRequirement.EMPTY_OR_CORRUPT.name} : {self.CATEGORY_COUNT[OcrRequirement.EMPTY_OR_CORRUPT]} files\n"
            f"Successfully Processed: {successful_count} file(s)\n"
            "------------------------------\n"
            f"Used {self.num_workers} parallel processes.\n"
            f"OCR Language: '{self.language}'\n"
            f"Force OCR: {self.force_ocr}\n"
            f"Skip Text: {self.skip_text}\n"
            f"Redo OCR: {self.redo_ocr}\n"
            f"Deskew: {self.deskew}\n"
            "------------------------------\n"
        )

        logger.info(summary)
        

        # logger.info("\n\n--- OCR Processing Summary ---\n")
        # logger.info(f"Output folder: {output_folder}")
        # logger.info("---------------------")
        # logger.info(f"Total PDF files found: {self.pdfs_found_count}")
        # logger.info(f'{OcrRequirement.OCR_REQUIRED.name} : {self.CATEGORY_COUNT[OcrRequirement.OCR_REQUIRED]} files')
        # logger.info(f'{OcrRequirement.OCR_NOT_REQUIRED.name} : {self.CATEGORY_COUNT[OcrRequirement.OCR_NOT_REQUIRED]} files')
        # logger.info(f'{OcrRequirement.EMPTY_OR_CORRUPT.name} : {self.CATEGORY_COUNT[OcrRequirement.EMPTY_OR_CORRUPT]} files')
        # logger.info(f"Successfully Processed: {successful_count} file(s)")
        # logger.info("---------------------")
        # logger.info(f"Used {self.num_workers} parallel processes.")
        # logger.info(f"OCR Language: '{self.language}'")
        # logger.info(f"Force OCR: {self.force_ocr}")
        # logger.info(f"Skip Text: {self.skip_text}")
        # logger.info(f"Redo OCR: {self.redo_ocr}")
        # logger.info(f"Deskew: {self.deskew}")
        # logger.info("---------------------")

        if failed_files:
            logger.warning(f"Failed to OCR: {len(failed_files)} file(s). Details below:")
            for info in failed_files:
                logger.warning(f"- File: {Path(info['input_file']).name}, Error: {info['error']}")
        else:
            logger.info("All found PDF files processed successfully!")
        
        time_finished = datetime.now()
        duration = time_finished - self.time_started

        final_timestamps = (
            "\n"
            f"Time Started: {self.time_started.strftime('%Y-%m-%d %H:%M:%S')}\n"
            f"Time Finished: {time_finished.strftime('%Y-%m-%d %H:%M:%S')}\n"
            f"Duration: {duration}\n"
            "Run Finished.\n"
        )

        logger.info(final_timestamps)
        

        # logger.info(f"Time Started: {self.time_started.strftime('%Y-%m-%d %H:%M:%S')}")
        # logger.info(f"Time Finished: {time_finished.strftime('%Y-%m-%d %H:%M:%S')}\n")
        # logger.info(f"Duration: {duration}\n")

def main_cli():
    '''
    Function to handle command-line arguments and run the batch processor.
    '''

    parser = argparse.ArgumentParser(
        description="Batch OCR PDF files in a folder using OCRmyPDF with multicore processing."
    )

    parser.add_argument(
        '-i', '--input_pdf',
        type=str,
        required=True,
        dest='input_folder',
        help="Path to the folder containing PDF files."
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
        default="eng+fil",
        help="Specify the OCR language (e.g., eng, fil). Default: eng"
    )

    parser.add_argument(
        "--deskew",
        action="store_true",
        help="Automatically deskews (corrects the rotation of) each page before performing OCR.  Improves accuracy, but may increase processing time. Default is False."
    )

    parser.add_argument(
        "--move",
        action="store_true",
        help="Move files instead of copying them. Default behavior is to copy."
    )

    parser.add_argument(
        '--output_type',
        type=str,
        default='pdf',
        help='Specify the output PDF type (e.g., pdf, pdfa, pdfa-1, pdfa-2, pdfa-3). Default is "pdf".'
    )

    parser.add_argument (
        '--max_workers',
        action='store_true',
        help= 'Enables tool to utilize max number of threads for processing'
    )

    args, unknown_args = parser.parse_known_args()
    # parse_known_args() intelligently separates the arguments it knows from the ones it doesn't.
    # It returns two things:
    #    args: An object containing the arguments you defined with parser.add_argument().
    #    unknown: A list of strings containing all the arguments it did not recognize.

    ocr_kwargs = {}
    
    ocr_kwargs['force_ocr'] = args.force_ocr
    ocr_kwargs['language'] = args.language
    ocr_kwargs['skip_text'] = args.skip_text
    ocr_kwargs['redo_ocr'] = args.redo_ocr
    ocr_kwargs['deskew'] = args.deskew
    ocr_kwargs['output_type'] = args.output_type

    ### include unlisted/unknown args
    if unknown_args:
        logger.info(f"Passing extra arguments to OCRmyPDF: {unknown_args}")
        try:
            # convert ['--key', 'value'] into {'key': 'value'}
            for i in range(0, len(unknown_args), 2):
                key = unknown_args[i].lstrip('-').replace('-', '_')
                value = unknown_args[i+1] # the value is next to index
                ocr_kwargs[key] = value
        except IndexError:
            logger.info("Error: Unrecognized arguments must be in key-value pairs.")
            logger.info("Example: --output-type pdfa --title 'My Document'")
            return



    # Configure root logger
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s]: %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    # Instantiate and run the batch processor from the CLI
    runner = BatchOCRRunner(
        input_folder=args.input_folder, # separate for clarity, code safety, and separation of concerns
        workers=args.workers,
        move_files=args.move,
        **ocr_kwargs
    )
    
    runner.run_batch()

if __name__ == "__main__":
    main_cli()