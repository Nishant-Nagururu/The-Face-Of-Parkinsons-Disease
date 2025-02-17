import os
import sys
import subprocess
import pandas as pd
import json
from urllib.parse import urlparse, parse_qs

def extract_video_id(youtube_url):
    parsed_url = urlparse(youtube_url)
    video_id = parse_qs(parsed_url.query).get('v')
    if video_id:
        return video_id[0]
    return None

# Function to download video from YouTube
def download_youtube_content(base_output_path, video_id, youtube_url):
    # Define the output directory for this video ID
    output_dir = os.path.join(base_output_path, video_id)
    os.makedirs(output_dir, exist_ok=True)

    # Added to avoid downloading videos over 20 minutes - comment out to download videos over 20 minutes.
    try:
        result = subprocess.run(
            ['yt-dlp', '--dump-json', youtube_url],
            capture_output=True, text=True, check=True
        )
        video_info = json.loads(result.stdout)
        duration = video_info.get('duration', 0)
        if duration > 1200:
            print(f"Skipping video {youtube_url} (Duration: {duration} seconds exceeds 20 minutes)")
            return
    except subprocess.CalledProcessError as e:
        print(f"Failed to retrieve metadata for {youtube_url}: {e}")
        return
    except json.JSONDecodeError as e:
        print(f"Failed to parse JSON metadata for {youtube_url}: {e}")
        return

    # New yt-dlp command to download video
    yt_dlp_command = [
        'yt-dlp', '--retries', '5', '--no-check-certificate',
        '--ffmpeg-location', '/apps/ffmpeg/4.3.1/bin/ffmpeg',  # Replace with your actual path
        '--output-na-placeholder', 'not_available',
        '-o', os.path.join(output_dir, '%(id)s.%(ext)s'),
        '-f', 'bestvideo[fps>=60]/bestvideo+bestaudio/best',
        '--merge-output-format', 'mp4',
        youtube_url
    ]

    # Run the command
    try:
        subprocess.run(yt_dlp_command, check=True)
        print(f"Downloaded video: {youtube_url}")
    except subprocess.CalledProcessError as e:
        print(f"Failed to download {youtube_url}: {e}")

# Function to process metadata files in a given directory
def process_metadata_files(directory):
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith('metadata.xlsx'):
                metadata_file_path = os.path.join(root, file)
                print(f"Processing metadata file: {metadata_file_path}")

                # Determine the speaker_id as the part of the path before 'metadata.xlsx'
                parts = metadata_file_path.split(os.sep)
                metadata_index = parts.index('metadata.xlsx')
                speaker_id = parts[metadata_index - 1]
                print(f"Speaker ID: {speaker_id}")

                # Define base output path for this speaker
                base_output_path = os.path.join(directory, speaker_id)

                df = pd.read_excel(metadata_file_path)
                links = df['link'].tolist()

                for link in links:
                    video_id = extract_video_id(link)
                    if video_id:
                        # Define the output directory for this video ID
                        output_dir = os.path.join(base_output_path, video_id)
                        print(f"Output Directory: {output_dir}")
                        if not os.path.exists(output_dir):
                            os.makedirs(output_dir)
                        download_youtube_content(base_output_path, video_id, link)
                    else:
                        print(f"Video ID could not be extracted from {link}")

# Main script execution
if __name__ == "__main__":
    """
    Example Usage:
        python download_videos.py <base_folder_path>

    Parameters:
        <base_folder_path> : Path to the base folder of the ParkCeleb dataset containing 'PD'/'CN' subfolders.
    """

    # root directory refers to the base folder of the ParkCeleb dataset
    if len(sys.argv) != 2:
        print("Usage: python download_videos.py <base_folder_path>")
        sys.exit(1)
        
    base_folder_path = sys.argv[1] 

    # Directories to traverse
    subdirectories = ['PD','CN']
    for subdir in subdirectories:
        subdir_path = os.path.join(base_folder_path, subdir)
        if os.path.exists(subdir_path):
            process_metadata_files(subdir_path)
        else:
            print(f"Directory does not exist: {subdir_path}")

    print("Processing completed.")