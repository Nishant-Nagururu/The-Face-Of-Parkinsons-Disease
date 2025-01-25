import os
import pandas as pd
import ffmpeg
import sys

def extract_speaker_segments_ffmpeg(video_path, speaker_info_path, random_id_csv_path, output_folder, prefix):
    try:
        # Read speaker_info.csv
        speaker_info = pd.read_csv(speaker_info_path)

        # Find the 'target' row
        target_row = speaker_info[speaker_info['status'] == 'target']
        if target_row.empty:
            print(f"No target speaker found in {speaker_info_path}")
            return None

        target_speaker = target_row.iloc[0]['speakers']
        print("TARGET SPEAKER", target_speaker)

        # Read the random_id.csv
        random_id_df = pd.read_csv(random_id_csv_path)

        # Find all rows where 'speaker' matches the target speaker
        target_segments = random_id_df[random_id_df['speaker'] == target_speaker]

        if target_segments.empty:
            print(f"No matching segments found for speaker {target_speaker} in {random_id_csv_path}")
            return None

        # Ensure output folder exists
        os.makedirs(output_folder, exist_ok=True)

        segment_index = 0

        # For each segment, extract and save
        for _, segment in target_segments.iterrows():
            start_time = segment['start']
            end_time = segment['end']

            if end_time - start_time < 3:
                continue

            # Use ffmpeg-python to trim the video segment
            output_filename = f"{prefix}_{segment_index}.mp4"
            output_filepath = os.path.join(output_folder, output_filename)

            try:
                (
                    ffmpeg
                    .input(video_path, ss=start_time, to=end_time)  # Input video with start and end time
                    .output(output_filepath, c="copy")             # Copy codec to avoid re-encoding
                    .run(overwrite_output=True)                    # Overwrite if file exists
                )
                print(f"Saved segment to {output_filepath}")
                segment_index += 1
            except ffmpeg.Error as e:
                print(f"Error processing segment {segment_index}: {e}")
        
        return True

    except Exception as e:
        print(f"Error extracting speaker segments: {e}")
        return None

def main(base_folder_path):
    for folder_type in ["PD", "CN"]:
        folder_path = os.path.join(base_folder_path, folder_type)
        if not os.path.exists(folder_path):
            print(f"Folder {folder_path} does not exist. Skipping.")
            continue

        # Iterate through folders like "pd_" or "cn_"
        for subfolder in os.listdir(folder_path):
            if subfolder.startswith(folder_type.lower() + "_"):
                subfolder_path = os.path.join(folder_path, subfolder)

                # Iterate through folders with random IDs
                for random_id_folder in os.listdir(subfolder_path):
                    random_id_path = os.path.join(subfolder_path, random_id_folder)

                    if not os.path.isdir(random_id_path):
                        print("Skipping non-directory:", random_id_path)
                        continue

                    # Locate the video and CSV files
                    video_file = None
                    speaker_info_path = None
                    random_id_csv_path = None

                    for file in os.listdir(random_id_path):
                        if file.endswith((".mp4")):
                            video_file = file
                        elif file == "speakers_info.csv":
                            speaker_info_path = os.path.join(random_id_path, file)
                        elif file.endswith(".csv") and file != "speakers_info.csv":
                            random_id_csv_path = os.path.join(random_id_path, file)

                    if video_file and speaker_info_path and random_id_csv_path:
                        video_path = os.path.join(random_id_path, video_file)
                        print(f"Processing video: {video_path}")

                        try:
                            # Extract the speaker segments using ffmpeg-python
                            extract_speaker_segments_ffmpeg(
                                video_path, speaker_info_path, random_id_csv_path, random_id_path, folder_type
                            )

                        except Exception as e:
                            print(f"Error processing {random_id_path}: {e}")
                    else:
                        print(f"Missing required files in {random_id_path}")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python script.py <base_folder_path>")
        sys.exit(1)
    base_folder_path = sys.argv[1]
    main(base_folder_path)