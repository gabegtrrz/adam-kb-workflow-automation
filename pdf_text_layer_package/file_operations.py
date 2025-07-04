import shutil
import logging
from pathlib import Path

# Set up a logger for this module
logger = logging.getLogger(__name__)

class FileOps:
    """
    Handles file operations such as moving and copying files into categorized sub-directories
    within a main output folder. This is useful for organizing files based on processing results
    (e.g., separating skipped, error, or processed files).
    """

    def __init__(self, base_output_dir):
        """
        Initialize the FileOps object with a base directory for all output.

        The constructor ensures the base output directory exists, creating it if necessary.
        If the directory cannot be created, an exception is raised and logged.

        Args:
            base_output_dir (Path or str): The root folder where all categorized
                sub-folders will be created and files will be organized.
        """
        self.base_output_dir = Path(base_output_dir)
        try:
            self.base_output_dir.mkdir(parents=True, exist_ok=True)
            logger.info(f"FileOps initialized. Output will be saved in: {self.base_output_dir}")
        except Exception as e:
            logger.error(f"FATAL: Could not create base output directory at '{self.base_output_dir}'. Error: {e}")
            # Re-raise the exception because if folder can't be created, nothing else can proceed.
            raise

    def move_file(self, source_path, destination_folder_name: str):
        """
        Move a file into a named sub-directory within the base output directory.

        This method will create the destination sub-directory if it does not already exist.
        It is robust to missing source files and will log a warning if the source does not exist or is not a file.

        Args:
            source_path (Path or str): The full path of the file to be moved.
            destination_folder_name (str): The name of the sub-folder to move the file into
                (e.g., 'Skipped_NoOCRequired', 'Errors_Or_Empty').

        Returns:
            None
        """
        source_path = Path(source_path)

        if not source_path.is_file():
            logger.warning(f"Cannot move file. Source not found or is not a file: {source_path}")
            return

        try:
            # Construct the full path for the destination folder (e.g., .../output/Skipped_NoOCRequired)
            destination_dir = self.base_output_dir / destination_folder_name

            # Create the specific sub-folder if it doesn't exist.
            destination_dir.mkdir(exist_ok=True)

            # Construct the final destination path for the file itself.
            final_destination_path = destination_dir / source_path.name

            # Move the file. shutil.move is great because it handles moving across different drives.
            shutil.move(str(source_path), str(final_destination_path))

            logger.info(f"--> Moved '{source_path.name}' \n to '{destination_folder_name}' folder.")

        except Exception as e:
            logger.error(f"Failed to move '{source_path.name}'. Error: {e}")

    def copy_file(self, source_path: Path, destination_folder_name: str):
        """
        Copy a file to a subfolder within the base output directory.

        This method will create the destination sub-directory if it does not already exist.
        It uses shutil.copy2 to preserve file metadata such as timestamps.

        Args:
            source_path (Path or str): The full path of the file to be copied.
            destination_folder_name (str): The name of the sub-folder to copy the file into.

        Returns:
            None
        """
        destination_dir = self.base_output_dir / destination_folder_name
        destination_dir.mkdir(parents=True, exist_ok=True)
        try:
            # Use shutil.copy2 to preserve metadata like timestamps
            shutil.copy2(str(source_path), str(destination_dir / Path(source_path).name))
            logger.debug(f"Copied '{Path(source_path).name}' to '{destination_dir}'.")
        except Exception as e:
            logger.error(f"Failed to copy '{Path(source_path).name}': {e}")



