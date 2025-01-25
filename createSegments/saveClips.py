import os
import pandas as pd
import ffmpeg
import sys

def extract_speaker_segments_ffmpeg(video_path, speaker_info_path, random_id_csv_path, output_folder, prefix):
    """
    Extracts video segments corresponding to the target speaker from the provided video file using ffmpeg.

    This function performs the following steps:
        1. Reads speaker information to identify the target speaker.
        2. Parses segment details to determine which parts of the video to extract.
        3. Uses ffmpeg to extract the specified segments without re-encoding.
        4. Saves the extracted segments to the designated output folder with a standardized naming convention.

    Parameters:
        video_path (str): Path to the original video file.
        speaker_info_path (str): Path to the 'speakers_info.csv' file containing speaker details.
        random_id_csv_path (str): Path to the CSV file containing segment start and end times.
        output_folder (str): Directory where the extracted video segments will be saved.
        prefix (str): Prefix to be used for naming the output video files (e.g., "PD", "CN").

    Returns:
        bool or None: Returns True if segments are successfully extracted, otherwise None.
    """
    try:
        # Read speaker information from the CSV file
        speaker_info = pd.read_csv(speaker_info_path)

        # Identify the row corresponding to the target speaker
        target_row = speaker_info[speaker_info['status'] == 'target']
        if target_row.empty:
            print(f"No target speaker found in {speaker_info_path}")
            return None

        # Extract the target speaker's identifier
        target_speaker = target_row.iloc[0]['speakers']
        print("TARGET SPEAKER:", target_speaker)

        # Read segment details from the provided CSV file
        random_id_df = pd.read_csv(random_id_csv_path)

        # Filter segments that belong to the target speaker
        target_segments = random_id_df[random_id_df['speaker'] == target_speaker]

        if target_segments.empty:
            print(f"No matching segments found for speaker {target_speaker} in {random_id_csv_path}")
            return None

        # Ensure the output directory exists; create it if it doesn't
        os.makedirs(output_folder, exist_ok=True)

        segment_index = 0  # Initialize counter for naming segments

        # Iterate through each segment and extract it using ffmpeg
        for _, segment in target_segments.iterrows():
            start_time = segment['start']  # Start time of the segment in seconds
            end_time = segment['end']      # End time of the segment in seconds

            # Skip segments shorter than 3 seconds to avoid very short clips
            if end_time - start_time < 3:
                print(f"Skipping short segment {segment_index} (Duration: {end_time - start_time} seconds)")
                continue

            # Define the output filename with the specified prefix and segment index
            output_filename = f"{prefix}_{segment_index}.mp4"
            output_filepath = os.path.join(output_folder, output_filename)

            try:
                # Use ffmpeg to extract the segment without re-encoding
                (
                    ffmpeg
                    .input(video_path, ss=start_time, to=end_time)  # Specify start and end times
                    .output(output_filepath, c="copy")              # Use 'copy' codec to avoid re-encoding
                    .run(overwrite_output=True)                      # Overwrite output file if it exists
                )
                print(f"Saved segment to {output_filepath}")
                segment_index += 1  # Increment the segment counter
            except ffmpeg.Error as e:
                # Handle ffmpeg-specific errors
                print(f"Error processing segment {segment_index}: {e.stderr.decode()}")
                continue

        return True

    except Exception as e:
        # Handle any unexpected errors during the extraction process
        print(f"Error extracting speaker segments: {e}")
        return None

def main(base_folder_path):
    """
    Processes all video files within the specified base folder, extracting segments for target speakers.

    The function performs the following operations:
        1. Iterates through the 'PD' and 'CN' categories within the base folder.
        2. For each relevant subfolder, locates video files and associated metadata.
        3. Extracts segments corresponding to the target speaker using ffmpeg.
        4. Handles errors gracefully, ensuring the processing of other files continues uninterrupted.

    Parameters:
        base_folder_path (str): Path to the root directory containing 'PD' and 'CN' subfolders.
    """
    # Define the categories to process
    for folder_type in ["PD", "CN"]:
        folder_path = os.path.join(base_folder_path, folder_type)
        if not os.path.exists(folder_path):
            print(f"Folder {folder_path} does not exist. Skipping.")
            continue

        # Iterate through subfolders matching the pattern (e.g., "pd_1", "cn_2")
        for subfolder in os.listdir(folder_path):
            if subfolder.startswith(folder_type.lower() + "_"):
                subfolder_path = os.path.join(folder_path, subfolder)

                # Iterate through folders with random IDs within the current subfolder
                for random_id_folder in os.listdir(subfolder_path):
                    random_id_path = os.path.join(subfolder_path, random_id_folder)

                    if not os.path.isdir(random_id_path):
                        print("Skipping non-directory:", random_id_path)
                        continue

                    # Initialize variables to store file paths
                    video_file = None
                    speaker_info_path = None
                    random_id_csv_path = None

                    # Locate necessary files within the random ID directory
                    for file in os.listdir(random_id_path):
                        if file.endswith(".mp4"):  # Identify the main video file
                            video_file = file
                        elif file == "speakers_info.csv":  # Identify the speaker info CSV
                            speaker_info_path = os.path.join(random_id_path, file)
                        elif file.endswith(".csv") and file != "speakers_info.csv":  # Identify the segments CSV
                            random_id_csv_path = os.path.join(random_id_path, file)

                    # Proceed only if all required files are found
                    if video_file and speaker_info_path and random_id_csv_path:
                        video_path = os.path.join(random_id_path, video_file)
                        print(f"Processing video: {video_path}")

                        try:
                            # Extract speaker segments using the defined function
                            success = extract_speaker_segments_ffmpeg(
                                video_path=video_path,
                                speaker_info_path=speaker_info_path,
                                random_id_csv_path=random_id_csv_path,
                                output_folder=random_id_path,
                                prefix=folder_type
                            )

                            if not success:
                                print(f"Failed to extract segments for video: {video_path}")

                        except Exception as e:
                            # Handle any errors that occur during the extraction process
                            print(f"Error processing {random_id_path}: {e}")
                    else:
                        # Inform the user about missing required files
                        missing_files = []
                        if not video_file:
                            missing_files.append("video file (*.mp4)")
                        if not speaker_info_path:
                            missing_files.append("speakers_info.csv")
                        if not random_id_csv_path:
                            missing_files.append("segments CSV (*.csv)")
                        print(f"Missing required files in {random_id_path}: {', '.join(missing_files)}")

if __name__ == "__main__":
    """
    Example usage:
        python saveClips.py <base_folder_path>

    Parameters:
        <base_folder_path> : Path to the base folder of the ParkCeleb dataset containing 'PD' and 'CN' subfolders.
    """
    # Validate the number of command-line arguments
    if len(sys.argv) != 2:
        print("Usage: python script.py <base_folder_path>")
        sys.exit(1)

    # Parse the base folder path from command-line arguments
    base_folder_path = sys.argv[1]

    # Execute the main processing function
    main(base_folder_path)
