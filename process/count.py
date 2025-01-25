import os
import csv
import sys

def process_metadata_csv(file_path):
    """
    Process a single metadata CSV file to count total rows and rows with duration_seconds > 5.

    This function reads a CSV file, counts the total number of data rows (excluding headers),
    counts how many rows have 'duration_seconds' greater than 5, and retrieves the last row.

    Parameters:
        file_path (str): Path to the metadata.csv file.

    Returns:
        tuple:
            total_rows (int): Total number of data rows in the CSV.
            duration_over_5 (int): Number of rows where 'duration_seconds' > 5.
            last_row (dict or None): The last row of the CSV as a dictionary, or None if no data rows exist.
    """
    total_rows = 0
    duration_over_5 = 0
    last_row = None

    try:
        # Open the CSV file for reading
        with open(file_path, mode='r', newline='', encoding='utf-8') as csvfile:
            reader = csv.DictReader(csvfile)
            # Iterate through each row in the CSV
            for row in reader:
                total_rows += 1
                last_row = row  # Update last_row to the current row

                try:
                    # Attempt to convert 'duration_seconds' to float and check if it's greater than 5
                    if float(row.get('duration_seconds', 0)) > 5:
                        duration_over_5 += 1
                except ValueError:
                    # Handle cases where 'duration_seconds' cannot be converted to float
                    print(f"Warning: Could not convert duration_seconds value '{row.get('duration_seconds')}' to float in file {file_path}.")
    except Exception as e:
        # Handle any unexpected errors during file processing
        print(f"Error processing {file_path}: {e}")

    return total_rows, duration_over_5, last_row

def scan_folder(folder_path):
    """
    Traverse the given folder to find and process all metadata.csv files.

    This function walks through the directory tree starting at 'folder_path',
    identifies all files named 'metadata.csv', and processes each using the
    'process_metadata_csv' function. It accumulates the total counts across all files.

    Parameters:
        folder_path (str): Path to the root directory to scan.

    Returns:
        tuple:
            grand_total_rows (int): Cumulative total number of data rows across all CSV files.
            grand_duration_over_5 (int): Cumulative number of rows with 'duration_seconds' > 5 across all CSV files.
    """
    grand_total_rows = 0
    grand_duration_over_5 = 0

    # Walk through the directory tree
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file == "metadata.csv":
                file_path = os.path.join(root, file)
                # Process the current metadata.csv file
                total_rows, duration_over_5, last_row = process_metadata_csv(file_path)
                grand_total_rows += total_rows
                grand_duration_over_5 += duration_over_5

                # Display the last row of the current file
                if last_row is not None:
                    print(f"\nProcessed {file_path}:")
                    print("Last row:", last_row)
                else:
                    print(f"\nProcessed {file_path}: No data rows found in this file.")

    return grand_total_rows, grand_duration_over_5

def main(folder_path):
    """
    Entry point of the script.

    This function parses command-line arguments, validates the input directory,
    and initiates the scanning and processing of metadata CSV files. It
    finally prints a summary of the processed data.
    """

    print(f"Scanning folder: {folder_path}")

    # Scan the folder and process all metadata.csv files found
    total_rows, duration_over_5 = scan_folder(folder_path)

    # Print a summary of the results
    print("\nSummary:")
    print(f"Total data rows (excluding headers): {total_rows}")
    print(f"Rows with duration_seconds > 5: {duration_over_5}")

if __name__ == "__main__":
    """
    Example usage:
        python count.py <folder_path>

    Parameters:
        <folder_path> : Path to the outputted The_Face_Of_Parkinsons or The_Face_Of_Parkinsons_CN folders.
    """
    # Check if the correct number of command-line arguments is provided
    if len(sys.argv) < 2:
        print("Usage: python count.py <folder_path>")
        sys.exit(1)

    # Retrieve the folder path from command-line arguments
    folder_path = sys.argv[1]

    # Validate that the provided path is a directory
    if not os.path.isdir(folder_path):
        print(f"Error: {folder_path} is not a valid directory.")
        sys.exit(1)

    main(folder_path)
