import ocrmypdf
import ocrmypdf.exceptions
from multiprocessing import Process
import argparse
import sys

def extract_text_ocrmypdf(input_pdf: str, output_pdf: str, language:str ='eng'):
    '''
    args:
        input_pdf (str): Path
        output_pdf (str): Path

    Returns:
        str: success message/exception
    '''

    try:
        # ocrmypdf.ocr(pdfa_image_compression='jpeg',input_file= input_pdf, output_file=output_pdf, force_ocr=True)
        ocrmypdf.ocr(language= language,input_file= input_pdf, output_file= =output_pdf,)
        return output_pdf
    
    
    except ocrmypdf.exceptions.InputFileError as e:
        print(f"Error: Input file problem. {e}")

    except ocrmypdf.exceptions.ExitCodeException as e:
        raise Exception(f"OCRmyPDF failed. Error: {e}")

    except Exception as e:
        print(f"An unexpected error occurred: {e}")
    

'''
def run_ocr():
    p = Process(target=extract_text_ocrmypdf)
    p.start
    p.join
'''
    
def main():
    parser = argparse.ArgumentParser(
        description="Run OCR applying scanned text as a layer on PDF."
    )
    parser.add_argument('input_pdf', help='Path to input file.')
    parser.add_argument('output_pdf', help='Path for the output PDF file.')
    parser.add_argument('-l', '--language', default='eng+fil',
                        help='Language used int the document. Use "+" sign for multi-lang. Default is eng \n(e.g. "-l eng+fil") ')
    
    args = parser.parse_args()

    try:
        processed_pdf_path = extract_text_ocrmypdf(args.input_pdf, args.output_pdf, args.language)
        print(processed_pdf_path)
    except Exception as e:
        print(f'Error: {e}', file=sys.stderr)
        sys.exit(1)



    

if __name__ == "__main__":
    main()

