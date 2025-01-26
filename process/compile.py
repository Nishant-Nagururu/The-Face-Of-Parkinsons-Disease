import os
import sys
import cv2
import ffmpeg
import shutil
import pandas as pd
import numpy as np

def get_video_duration(video_path):
    """
    Retrieve the duration of a video using ffmpeg.

    Parameters:
        video_path (str): Path to the video file.

    Returns:
        float or None: Duration of the video in seconds, or None if an error occurs.
    """
    try:
        # Probe the video file to extract metadata
        probe = ffmpeg.probe(video_path)
        return float(probe["format"]["duration"])
    except Exception as e:
        # If an error occurs (e.g., file not found or corrupted), return None
        print(f"Error retrieving duration for {video_path}: {e}")
        return None

def main(base_folder_path, output_path, subfolder_prefix):
    """
    Process videos from the base folder by copying and renaming them while creating corresponding metadata.

    The function performs the following steps:
        1. Creates the output directory if it doesn't exist.
        2. Iterates through subfolders in the base directory that match the specified prefix.
        3. For each eligible subfolder:
            a. Cleans and prepares the corresponding output subfolder.
            b. Processes each random ID folder within the subfolder to locate and handle videos.
            c. Extracts metadata such as video name, years from diagnosis, and video duration.
            d. Copies and renames video files to the output directory.
            e. Copies the target face image to the target_faces folder with a new name.
            f. Compiles and saves the metadata to a CSV file.

    Parameters:
        base_folder_path (str): Root directory containing the subfolders with videos and metadata.
        output_path (str): Destination directory for processed videos and metadata.
        subfolder_prefix (str): Prefix of the subfolders to process (e.g., "pd_", "cn_").
    """
    # Ensure the output directory exists
    os.makedirs(output_path, exist_ok=True)

    # Check if the base folder exists; exit early if it doesn't
    if not os.path.exists(base_folder_path):
        print(f"Base folder does not exist: {base_folder_path}")
        return

    # Iterate over each subfolder in the base directory
    for subfolder in os.listdir(base_folder_path):
        # Process only subfolders that start with the specified prefix
        if subfolder.startswith(subfolder_prefix):
            subfolder_path = os.path.join(base_folder_path, subfolder)
            output_subfolder_path = os.path.join(output_path, subfolder)

            # Remove the existing output subfolder to ensure a clean state
            if os.path.exists(output_subfolder_path):
                shutil.rmtree(output_subfolder_path)
            # Create the output subfolder
            os.makedirs(output_subfolder_path, exist_ok=True)

            # Create the target_faces directory within the output subfolder
            target_faces_path = os.path.join(output_subfolder_path, "target_faces")
            os.makedirs(target_faces_path, exist_ok=True)

            # Path to the metadata CSV file within the output subfolder
            metadata_csv_path = os.path.join(output_subfolder_path, "metadata.csv")
            metadata_rows = []  # List to store metadata for each video
            counter = 1  # Counter to rename videos sequentially

            # Iterate over each random ID folder within the current subfolder
            for random_id_folder in os.listdir(subfolder_path):
                random_id_path = os.path.join(subfolder_path, random_id_folder)
                # Skip if the path is not a directory
                if not os.path.isdir(random_id_path):
                    continue

                # Path to the target face image
                target_face_source_path = os.path.join(random_id_path, "target_face.png")
                # Path to the directory containing processed videos
                processed_videos_path = os.path.join(random_id_path, "processed_videos")

                # Skip if the processed_videos directory does not exist
                if not os.path.isdir(processed_videos_path):
                    continue

                # Initialize 'years_val' to store years from diagnosis
                years_val = None
                # Path to the speakers_info CSV file
                info_csv_path = os.path.join(random_id_path, "speakers_info.csv")
                if os.path.exists(info_csv_path):
                    try:
                        # Load the speakers_info CSV into a DataFrame
                        speakers_df = pd.read_csv(info_csv_path)
                        # Select the row where the status is 'target'
                        target_row = speakers_df[speakers_df["status"] == "target"].iloc[0]
                        years_val = target_row["years_from_diagnosis"]

                        # Handle special cases for 'years_val'
                        if years_val == "same":
                            years_val = 0.0
                        elif years_val != 0.0:
                            before_after_val = target_row["before_after_diagnosis"]
                            if isinstance(before_after_val, str):
                                # Assign negative value if 'before', positive if 'after'
                                years_val = -abs(years_val) if before_after_val.lower() == "before" else abs(years_val)
                            else:
                                years_val = None
                    except Exception as e:
                        print(f"Error processing {info_csv_path}: {e}")
                        years_val = None

                # Walk through the processed_videos directory to find .mp4 files
                for root, _, files in os.walk(processed_videos_path):
                    for file in files:
                        if file.endswith(".mp4"):
                            old_video_path = os.path.join(root, file)
                            # Create a new filename with the subfolder prefix and a counter
                            new_filename = f"{subfolder}_{counter}.mp4"
                            new_video_path = os.path.join(output_subfolder_path, new_filename)

                            try:
                                # Copy the video to the output directory with the new name
                                shutil.copy2(old_video_path, new_video_path)
                                # Retrieve the duration of the video
                                video_duration = get_video_duration(old_video_path)

                                # Define the new name for the target face image (without extension)
                                target_face_new_name = os.path.splitext(new_filename)[0] + ".png"
                                target_face_destination_path = os.path.join(target_faces_path, target_face_new_name)

                                # Copy and rename the target face image to the target_faces folder
                                if os.path.exists(target_face_source_path):
                                    shutil.copy2(target_face_source_path, target_face_destination_path)
                                else:
                                    print(f"Target face image not found: {target_face_source_path}")

                                # Append the metadata for this video
                                metadata_rows.append({
                                    "video_name": new_filename,
                                    "years_val": years_val,
                                    "duration_seconds": video_duration
                                })
                                counter += 1  # Increment the counter for the next video
                            except Exception as e:
                                print(f"Error copying {old_video_path} to {new_video_path}: {e}")

            # After processing all videos, save the metadata to a CSV file if any videos were processed
            if metadata_rows:
                try:
                    pd.DataFrame(metadata_rows).to_csv(metadata_csv_path, index=False)
                    print(f"Saved metadata to {metadata_csv_path}")
                except Exception as e:
                    print(f"Error saving metadata to {metadata_csv_path}: {e}")
            else:
                print(f"No videos processed for subfolder: {subfolder_path}")

if __name__ == "__main__":
    """
    Example usage:
        python compile.py <base_folder_path> <output_path> <subfolder_prefix>

    Parameters:
        <base_folder_path>   : Path to the PD or CN folder in the ParkCeleb dataset. 
        <output_path>        : Destination directory for the compiled videos and metadata. This should be a path to the The_Face_Of_Parkinsons or The_Face_Of_Parkinsons_CN folders.
        <subfolder_prefix>   : Prefix of the subfolders to process (e.g., "pd_", "cn_"). Must correspond to the base_folder_path.
    """
    # Validate the number of command-line arguments
    if len(sys.argv) != 4:
        print("Usage: python compile.py <base_folder_path> <output_path> <subfolder_prefix>")
        sys.exit(1)

    # Parse command-line arguments
    base_folder_path = sys.argv[1]
    output_path = sys.argv[2]
    subfolder_prefix = sys.argv[3]

    # Execute the main processing function
    main(base_folder_path, output_path, subfolder_prefix)
