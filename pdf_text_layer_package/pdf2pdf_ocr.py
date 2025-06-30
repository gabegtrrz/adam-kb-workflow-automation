
# Batch OCR Processing for PDF Files using OCRmyPDF
#
# This module provides classes and a CLI for batch-processing PDF files with OCR (Optical Character Recognition).
# It includes intelligent triage to skip files that do not require OCR, multiprocessing for speed, and optional
# sidecar text extraction. The main classes are:
#   - OcrProcessor: Handles OCR for a single PDF file.
#   - BatchOCRRunner: Manages batch processing, triage, and parallelization.
#
# Usage:
#   python pdf2pdf_ocr.py -i <input_folder> [options]
#
# Dependencies:
#   - ocrmypdf
#   - pymupdf
#   - tqdm
#   - triage (local)
#   - file_operations (local)

import logging
import sys
from pathlib import Path
from multiprocessing import Pool, cpu_count
import argparse
from datetime import datetime

import ocrmypdf
import ocrmypdf.exceptions

import pymupdf as pypdf

from triage import PdfTriage, OcrRequirement
from file_operations import FileOps

from tqdm import tqdm

logger = logging.getLogger(__name__)

class OcrProcessor:
    '''
    Handles OCR processing for a single PDF file using OCRmyPDF.

    This class encapsulates the logic for running OCR on one PDF file, including
    options for language, deskewing, skipping files with text, and generating a sidecar text file.
    '''
    def __init__(self, force_ocr: bool=False, language: str='eng', skip_text=False, redo_ocr=True, deskew: bool=False, sidecar: bool = False, **kwargs):
        '''
        Initialize the OcrProcessor with OCR settings.

        Args:
            force_ocr (bool): Force OCR on all files, even if text is detected.
            language (str): Language(s) for OCR (e.g., 'eng', 'eng+fil').
            skip_text (bool): Skip OCR on PDFs that already contain text layers.
            redo_ocr (bool): If True, OCR is performed only on image pages, preserving native text.
            deskew (bool): Deskew pages before OCR.
            sidecar (bool): If True, creates a .txt sidecar with extracted text.
            **kwargs: Additional keyword arguments passed to ocrmypdf.ocr().
        '''
        self.force_ocr  = force_ocr
        self.language   = language
        self.deskew     = deskew
        self.skip_text  = skip_text
        self.redo_ocr   = redo_ocr
        self.ocr_kwargs = kwargs
        self.sidecar    = sidecar


    def process_file(self, input_path, output_path= "") -> dict:
        '''
        Process a single PDF file with OCRmyPDF.

        Args:
            input_path (str or Path): Path to the input PDF file.
            output_path (str or Path, optional): Path to save the output searchable PDF. If not provided, a default is used.

        Returns:
            dict: Dictionary with status, input/output file paths, and error info if any.
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

            if self.sidecar:
                sidecar_path = output_path.with_suffix('.txt')
                try:
                    with pypdf.open(output_path) as doc:
                        all_text = ""
                        for page in doc:
                            all_text += page.get_text()

                    sidecar_path.write_text(all_text, encoding='utf-8')

                except Exception as e:
                    # Log or handle the text extraction error
                    logger.error(f"Could not create sidecar for {output_path.name}. Reason: {e}")
           
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
    Manages batch OCR processing for a folder of PDF files.

    This class performs triage to determine which files require OCR, sorts/copies/moves files
    into appropriate folders, and runs OCR in parallel using multiprocessing.
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

    def __init__(self, input_folder: str, force_ocr: bool = False, language: str = 'eng', skip_text=False, redo_ocr=True, workers: int = -1, deskew: bool=False, move_files: bool = False, is_max_workers = False, sidecar: bool = False, **kwargs):
        '''
        Initialize the batch OCR runner for a folder of PDFs.

        Args:
            input_folder (str): Path to the folder containing PDF files.
            force_ocr (bool): Force OCR on all files.
            language (str): Language(s) for OCR.
            skip_text (bool): Skip OCR on PDFs that already contain text layers.
            redo_ocr (bool): OCR only image pages, preserve native text.
            workers (int): Number of parallel processes (default: cpu_count() - 2).
            is_max_workers (bool): Use all available CPU cores if True.
            deskew (bool): Deskew pages before OCR.
            move_files (bool): Move files to sorted subdirectories (default: copy).
            sidecar (bool): Create a text file sidecar for each OCR'd PDF.
            **kwargs: Additional keyword arguments for OCRmyPDF.
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

        ### Initialize Module Args
        self.move_files = move_files
        self.sidecar = sidecar
        self.ocr_kwargs = kwargs
        
        # Instantiate helper classes
        self.triage = PdfTriage()
        self.file_op = FileOps(base_output_dir=self.input_folder_path)
        
        if not is_max_workers:
            if workers == -1:
                self.num_workers = max(1, cpu_count() - 2)
            elif workers > 0:
                self.num_workers = workers
            else:
                logger.error('Workers argument cannot be 0 or negative. Setting workers to 1.')
                self.num_workers = 1
        else:
            self.num_workers = cpu_count()



    
    
    ### Helper function to unpack arguments for starmap. ###
    # This does the actual processing. Crucially, it calls `process_file()` function.

    @staticmethod
    def worker_adapter(processor: OcrProcessor, input_p, output_p):
        return processor.process_file(input_p, output_p)


    def _get_pdfs(self) -> list:
        """
        Scan the input folder for PDF files.

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
        Run the batch OCR process on all PDF files in the input folder.
        Performs triage, sorts/copies/moves files, and runs OCR in parallel.
        '''
        
        # OUTPUT: Create the results output folder
        output_folder_path = self.output_folder_path
        
        all_pdf_files = self._get_pdfs()
        sorted_pdfs = self._classify_and_sort_pdfs(pdf_files=all_pdf_files, output_folder_path=output_folder_path.parent,is_move_files=self.move_files)

        ### Task creation
        
        tasks_for_ocr = self._prepare_tasks(pdf_files=sorted_pdfs, output_folder_path=output_folder_path)
        
        if not tasks_for_ocr:
            logger.info("No files requiring OCR were found. Exiting.")
            self._log_summary([], output_folder_path)
            return

        logger.info(f"\n{len(tasks_for_ocr)} files require OCR. Starting parallel processing...")
            
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

            # Rollback when removing TQDM:
            # results = pool.starmap(BatchOCRRunner.worker_adapter, tasks_for_ocr)

        self._log_summary(results, output_folder_path)


    def _classify_and_sort_pdfs(self, pdf_files: list, output_folder_path, is_move_files:bool = False):
        """
        Classify PDF files by OCR requirement and sort/copy/move them into folders.

        Args:
            pdf_files (list): List of PDF file paths to classify.
            output_folder_path (str or Path): Path to the output folder for sorted files.
            is_move_files (bool): If True, move files; otherwise, copy files.

        Returns:
            list: List of PDF file paths that require OCR processing.

        Side Effects:
            - Moves or copies files to subfolders based on classification.
            - Logs the count of files in each category.
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
        Prepare a list of OCR tasks for multiprocessing.

        Args:
            pdf_files (list): List of PDF file paths to process.
            output_folder_path (str or Path): Path to the output folder for processed files.

        Returns:
            list[tuple[OcrProcessor, Path, Path]]: List of (OcrProcessor, input_path, output_path) tuples.
        '''
        output_folder_path = Path(output_folder_path)
        tasks = []

        processor = OcrProcessor(
            self.force_ocr,
            self.language,
            skip_text = self.skip_text,
            redo_ocr = self.redo_ocr,
            sidecar=self.sidecar,
            **self.ocr_kwargs
            )
        
        for pdf_path in pdf_files:
            output_path = output_folder_path / f'[OCR] {pdf_path.name}'
            tasks.append((processor, pdf_path, output_path))
    


        
            
        return tasks
        

    def _log_summary(self, results: list, output_folder: Path):
        """
        Log a summary of the batch OCR processing results.

        Args:
            results (list): List of result dicts from OCR processing.
            output_folder (Path): Path to the output folder.
        """
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
        
        if failed_files:
            logger.warning(f"Failed to OCR: {len(failed_files)} file(s). Details below:")
            for info in failed_files:
                logger.warning(f"- File: {Path(info['input_file']).name}, Error: {info['error']}")
        else:
            logger.info("All found PDF files requiring OCR processed successfully!")
        
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
        

def main_cli():
    '''
    Command-line interface for batch OCR processing.

    Parses arguments, configures logging, and runs the BatchOCRRunner.
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
        dest='redo_ocr', # Explicitly set dest
        help="Add argument to disable. Default = True: OCRs pages ignoring native text."
    )
    parser.set_defaults(redo_ocr=True) # Set the default to True

    parser.add_argument(
        "--workers",
        type=int,
        default=-1, # Use -1 to signal using the class default
        help=f"Set the number of parallel worker processes. Default: system cores minus 2."
    )
    
    parser.add_argument(
        "--max_workers",
        action="store_true",
        dest='is_max_workers',
        help="Use the maximum number of available CPU cores. Overrides --workers."
    )

    parser.add_argument(
        "--language",
        type=str,
        default="eng+fil",
        help="Specify the OCR language (e.g., eng, fil). Default: eng+fil"
    )

    parser.add_argument(
        "--deskew",
        action="store_true",
        help="Automatically deskews (corrects the rotation of) each page before performing OCR.  Improves accuracy, but may increase processing time. Default is False."
    )

    parser.add_argument(
        "--move",
        action="store_true",
        help="Move original files to sorted subdirectories instead of copying them. Default behavior is to copy."
    )

    parser.add_argument(
        '--output_type',
        type=str,
        default='pdf',
        help='Specify the output PDF type (e.g., pdf, pdfa, pdfa-1, pdfa-2, pdfa-3). Default is "pdf".'
    )

    parser.add_argument(
        "--sidecar",
        action="store_true",
        help="Create a text file sidecar for each OCR'd PDF."
    )


    args, unknown_args = parser.parse_known_args()

    ocr_kwargs = {}
    # ocr_kwargs['force_ocr'] = args.force_ocr
    # ocr_kwargs['language'] = args.language
    # ocr_kwargs['skip_text'] = args.skip_text
    # ocr_kwargs['redo_ocr'] = args.redo_ocr
    # ocr_kwargs['deskew'] = args.deskew
    # ocr_kwargs['output_type'] = args.output_type

    if unknown_args:
        logger.info(f"Passing extra arguments to OCRmyPDF: {unknown_args}")
        try:
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
        **vars(args),
        **ocr_kwargs
    )
    
    runner.run_batch()

if __name__ == "__main__":
    main_cli()
