# **Overview**

This module serves as a powerful and intelligent command-line tool for performing Optical Character Recognition (OCR) on a batch of PDF files. It leverages the robust `ocrmypdf` library, but enhances it with two key features:

1. **Intelligent Triage:** It first analyzes each PDF to determine if OCR is actually necessary. Files that are already text-searchable are skipped, saving immense amounts of processing time.
2. **Parallel Processing:** It uses Python's multiprocessing to run OCR on multiple files simultaneously, maximizing the use of available CPU cores to speed up the entire workflow.

The ultimate goal is to take a folder of mixed PDFs and efficiently convert only the non-searchable ones into fully searchable documents, which are then sorted into organized output folders.

# **Installation**

1. **Clone the Repository:**
    
    ```bash
    git clone <https://github.com/gabegtrrz/data-extraction-goeden.git>
    cd data-extraction-goeden
    
    ```
    
    ```bash
    git clone <https://github.com/gabegtrrz/data-extraction-goeden.git>
    cd data-extraction-goeden
    
    ```
    
2. **Install Python Libraries:** It's highly recommended to use a virtual environment.
    
    ```bash
    python -m venv venv
    .\\venv\\Scripts\\activate  # On Windows
    source venv/bin/activate # On macOS/Linux
    pip install -r requirements.txt
    
    ```
    
    ```bash
    python -m venv venv
    .\\venv\\Scripts\\activate  # On Windows
    source venv/bin/activate # On macOS/Linux
    pip install -r requirements.txt
    
    ```
    

# **Prerequisites**

To run this script, you'll need the following installed on your system:

- **Python:** Version 3.8 or higher.
- **Tesseract OCR Engine:** A 64-bit installation is required.
- **Ghostscript:** A 64-bit version is necessary for PDF processing.

# **Dependencies**

This script relies on external Python libraries and internal modules.

- **External Libraries**:
    - `ocrmypdf`: The core engine used for performing OCR.
    - `pymupdf`: Used by the triage module to analyze PDF content.
- **Internal Modules**:
    - `triage.py`: Contains the `PdfTriage` class for determining if a PDF needs OCR.
    - `file_operations.py`: Contains the `FileOps` class for organizing files into output folders.

# **Command-Line Usage**

The script is executed from your terminal. The basic command requires specifying the input folder.

** Input path must be a folder/directory*

```bash
python ocr.py -i "C:/path/to/your/pdf_folder"
```

## **Core Arguments**

These are the fundamental arguments for controlling the script's behavior.

| **Argument** | **Purpose** | **Example / Common Use** |
| --- | --- | --- |
| `--input_pdf` or `-i` | **Required.** Specifies the source folder to process. | `-i "C:/path/to/pdfs"` |
| `-l <LANG>`, `--language LANG` | Specifies the language(s) for Tesseract OCR. Accuracy depends heavily on this. | `-l eng` (English)&lt;br>`--language eng+fil` (English + Filipino) |
| `--workers N` | Sets the number of parallel worker processes. | `--workers 4` (Default: CPU cores minus 2) |
| `--move` | Moves the original files to the categorized output folders instead of copying them. | Add `--move` to enable. (Default: copy) |

## **Advanced Usage & Arguments**

For a complete list of `ocrmypdf` options, run `ocrmypdf --help`.

### **OCR Workflow Control**

These arguments control *if* and *how* OCR is applied to pages.

### **Image Processing and Quality**

These arguments manipulate images *before* they are sent to the OCR engine, which can dramatically improve accuracy.

### **PDF Output and Optimization**

These arguments control the properties and format of the final output PDF.

| **Argument** | **Purpose** | **Example** |
| --- | --- | --- |
| `--output-type TYPE` | Sets the output format. `pdfa` creates a PDF/A-2b compliant file (for archiving). `pdf` creates a standard PDF.

Default is set to `pdf` | `--output-type pdfa` |
| `--title "My Title"` | Sets the document's metadata (Title, Author, Subject, Keywords). | `--author "Corporate Archives"` |
| `--sidecar FILE` | Creates a "sidecar" text file containing all the text extracted from the PDF. | `--sidecar text_content.txt` |

# **Understanding the Output**

After the script finishes, go back to your input folder. You will find a new folder named `OCR_Results_[Date_Time]`.

- **Main Output:** Inside this folder, you will find your new, text-searchable PDFs. They will be prefixed with `[OCR]`.
    - Example: `[OCR] old_document_name.pdf`
- **Sidecar Files (Optional):** If you used the `-sidecar` option, you will also see `.txt` files containing the raw extracted text.
- **Sorted Originals (Optional):** If you used the `-move` option, the script will also create folders and sort your original files into them:
    - `Original PDFs`: The original files that required and underwent OCR.
    - `No_OCR_Required`: Original files that already had text and were skipped.
    - `Empty_or_Error`: Files that were empty, corrupt, or could not be opened.

# **Core Components**

The script is built around two main classes that divide the responsibilities of the workflow.

### **1. `OcrProcessor`**

This class is a focused "worker" designed to handle the OCR process for a **single file**.

- **Purpose**: To act as a configurable wrapper around the `ocrmypdf.ocr()` function. An instance of this class holds the specific OCR settings (language, force OCR, etc.) that will be used for a file.
- **Key Method: `process_file()`**:
    - Takes an input file path and an output file path.
    - Executes `ocrmypdf.ocr()` with the configured settings.
    - Includes robust error handling to catch common issues like encrypted or corrupt PDFs, ensuring that the failure of one file does not stop the entire batch.
    - Returns a dictionary containing the status (`'success'` or `'error'`) and relevant file paths.

### **2. `BatchOCRRunner`**

This is the main orchestrator that manages the entire end-to-end batch process.

- **Purpose**: To manage the workflow of finding, triaging, processing, and organizing a folder full of PDFs.
- **Responsibilities**:
    - **Initialization**: Takes user-defined settings from the command line (input folder, OCR language, number of workers, etc.) and sets up the environment.
    - **Triage & Sorting**:
        - Scans the input folder for all PDF files.
        - Uses the `PdfTriage` helper class to classify each PDF (`OCR_REQUIRED`, `OCR_NOT_REQUIRED`, `EMPTY_OR_CORRUPT`).
        - Uses the `FileOps` helper class to immediately sort non-required or corrupt files into appropriate sub-folders (`No_OCR_Required`, `Empty_or_Error`), removing them from the processing queue.
    - **Parallel Execution**:
        - Prepares a list of OCR tasks for the files that need processing.
        - Creates a multiprocessing pool of `OcrProcessor` workers.
        - Distributes the tasks to the workers for parallel processing.
    - **Logging & Summary**: Monitors the progress and logs a final summary detailing how many files were processed, skipped, or failed.

# **Workflow: How It Works**

The script follows a logical, multi-stage process:

1. **Initialization**: The `BatchOCRRunner` is started from the command line, pointing to an input folder. It creates a main output folder named `OCR_Results_{timestamp}` inside the input directory.
2. **Triage and File Sorting**: The runner scans all PDFs in the input folder. Each file is analyzed:
    - If a PDF is determined to be already text-searchable, it is copied to the `No_OCR_Required` sub-folder.
    - If a PDF is empty or corrupt, it is copied to the `Empty_or_Error` sub-folder.
    - Only files that are image-based or not sufficiently textual remain for the next step.
    - If the `-move` flag is used, the original files are moved to the respective sub-folders instead of being copied. Files needing OCR are moved to an `Original PDFs` sub-folder to await processing.
3. **Task Preparation**: A list of tasks is created for the remaining PDFs. Each task bundles an `OcrProcessor` instance with the input and final output paths.
4. **Parallel OCR Processing**: The script creates a pool of worker processes (by default, the number of CPU cores minus two). These workers run in parallel, with each one taking a task from the list and executing OCR. The newly created searchable PDFs are saved to the main `OCR_Results_{timestamp}` folder.
5. **Completion**: Once all tasks are complete, a summary of the entire operation is printed to the console.

| **Argument** | **Purpose** |
| --- | --- |
| `--force-ocr` | Forces OCR to be performed on pages that already have text. Useful for "re-doing" a bad OCR job. |
| `--redo-ocr` | This is enabled by default. It re-does OCR on pages with text by placing the new text layer over the image content. Use the `--redo-ocr` flag to disable this and keep the original text layer. |
| `--skip-text` | Skips OCR on any pages that already contain text. This is the default behavior if `--force-ocr` or `--redo-ocr` are not specified. |

| **Argument** | **Purpose** |
| --- | --- |
| `--clean` | Uses the `unpaper` tool to remove speckles, noise, and small black borders from scanned pages. |
| `--deskew` | Finds pages that were scanned at a slight angle and straightens them before OCR. Improves accuracy on crooked scans. |
| `--rotate-pages` | Rotates pages by a specified angle if the orientation is consistently wrong. |
| `--image-dpi N` | Overrides the DPI (dots per inch) that is assumed for images. Tesseract works best with ~300 DPI. |

# **Important Considerations & Limitations**

- **Image Quality is Key**: The accuracy of the OCR engine is highly dependent on the quality of the input images. Clear, high-resolution scans yield significantly better results.
- **Resource Intensive**: OCR is a computationally demanding task. The script utilizes multiprocessing but a noticeable decrease in system performance for other concurrent tasks is to be expected during execution.
- **Loss of Vector Text**: When processing PDFs that already contain native text, the script may rasterize the document, converting crisp vector text into an image layer. This can result in reduced text crispness and degradation of graphics in the final output.
- Logging: Ghostscript copyright showing in console is not an error.
- **Accuracy Limitations**:
    - The OCR engine may have difficulty with special characters, complex tables, and text embedded in artistic elements or logos.
    - Complex column layouts might result in text being displaced or misaligned in the OCR output.
    - **Tabular Data:** Transforming tables into structured, searchable tabular data is challenging for OCR and this script cannot accurately preserve table structures, leading to unstructured text output.