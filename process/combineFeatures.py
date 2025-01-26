import os
import sys
import pandas as pd
import shutil

def compile_csvs(base_path):
    """
    Recursively searches for all 'features.csv' files within the base directory,
    compiles them into a single DataFrame, and saves the consolidated CSV in the base directory.
    
    Parameters:
        base_path (str): 
            The root directory path where the search for 'features.csv' files begins.
    
    Returns:
        None
    """
    all_files = []  # List to store paths of all found 'features.csv' files
    
    # ---------------------------------------------------
    # 1) Traverse the directory tree to locate 'features.csv'
    # ---------------------------------------------------
    for root, _, files in os.walk(base_path):
        for file in files:
            if file.lower() == 'features.csv':  # Case-insensitive match
                file_path = os.path.join(root, file)
                all_files.append(file_path)
                print(f"Found 'features.csv': {file_path}")
    
    # ---------------------------------------------------
    # 2) Handle case where no 'features.csv' files are found
    # ---------------------------------------------------
    if not all_files:
        print("No 'features.csv' files found in the specified directory.")
        return  # Exit the function as there's nothing to compile
    
    # ---------------------------------------------------
    # 3) Read and concatenate all found CSV files
    # ---------------------------------------------------
    dataframes = []
    for file in all_files:
        try:
            df = pd.read_csv(file)
            dataframes.append(df)
            print(f"Successfully read '{file}' with {len(df)} records.")
        except Exception as e:
            print(f"Error reading '{file}': {e}")
    
    if not dataframes:
        print("No valid 'features.csv' files were read successfully.")
        return
    
    # Concatenate all DataFrames into one, resetting the index
    combined_df = pd.concat(dataframes, ignore_index=True)
    print(f"Combined DataFrame created with {len(combined_df)} total records.")
    
    # ---------------------------------------------------
    # 4) Save the merged DataFrame to a new CSV file
    # ---------------------------------------------------
    output_path = os.path.join(base_path, "compiled_features.csv")
    try:
        combined_df.to_csv(output_path, index=False)
        print(f"Compiled CSV successfully saved at: '{output_path}'")
    except Exception as e:
        print(f"Error saving compiled CSV to '{output_path}': {e}")

def copy_bounding_boxes(base_path):
    """
    Searches for all 'bounding_boxes' directories within the base directory,
    copies them into a new directory named 'compiled_bounding_box',
    and renames each copied directory to include its parent folder name to prevent conflicts.
    
    Parameters:
        base_path (str): 
            The root directory path where the search for 'bounding_boxes' folders begins.
    
    Returns:
        None
    """
    compiled_dir = os.path.join(base_path, "compiled_bounding_box")  # Destination directory
    os.makedirs(compiled_dir, exist_ok=True)  # Create destination directory if it doesn't exist
    print(f"Compiled bounding boxes will be stored in: '{compiled_dir}'")
    
    # ---------------------------------------------------
    # 1) Traverse the directory tree to locate 'bounding_boxes' folders
    # ---------------------------------------------------
    for root, dirs, _ in os.walk(base_path):
        for dir_name in dirs:
            if dir_name.lower() == "bounding_boxes":  # Case-insensitive match
                src = os.path.join(root, dir_name)  # Source directory path
                parent_folder = os.path.basename(root)  # Parent folder name
                new_folder_name = f"bounding_boxes_{parent_folder}"  # New unique folder name
                dest = os.path.join(compiled_dir, new_folder_name)  # Destination directory path
                
                # ---------------------------------------------------
                # 2) Copy the 'bounding_boxes' folder if it doesn't already exist
                # ---------------------------------------------------
                if os.path.exists(dest):
                    print(f"Skipping copy. Destination '{dest}' already exists.")
                else:
                    try:
                        shutil.copytree(src, dest)  # Recursively copy the directory
                        print(f"Copied '{src}' to '{dest}'.")
                    except Exception as e:
                        print(f"Error copying '{src}' to '{dest}': {e}")

def main():
    """
    Example usage:
        python combineFeatures.py <base_folder_path>

    Parameters:
        <base_folder_path>: Path to The_Face_Of_Parkinsons datset or the The_Face_Of_Parkinsons_CN dataset. 
    """
    # ---------------------------------------------------
    # 1) Parse Command-Line Arguments
    # ---------------------------------------------------
    if len(sys.argv) != 2:
        print("Usage: python combineFeatures.py <base_folder_path>")
        sys.exit(1)  # Exit the script if incorrect number of arguments are provided
    
    base_folder_path = sys.argv[1]  # The base directory provided by the user
    
    # ---------------------------------------------------
    # 2) Validate the Base Directory Path
    # ---------------------------------------------------
    if not os.path.isdir(base_folder_path):
        print(f"Error: The path '{base_folder_path}' is not a valid directory.")
        sys.exit(1)  # Exit the script if the path is invalid
    
    print(f"Starting compilation and copying processes in base directory: '{base_folder_path}'\n")
    
    # ---------------------------------------------------
    # 3) Execute the Compilation and Copying Functions
    # ---------------------------------------------------
    compile_csvs(base_folder_path)  # Compile all 'features.csv' files
    copy_bounding_boxes(base_folder_path)  # Copy and rename all 'bounding_boxes' folders
    
    print("\nAll processes completed successfully.")

if __name__ == "__main__":
    main()
