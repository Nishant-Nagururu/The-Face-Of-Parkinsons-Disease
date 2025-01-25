import os
import sys
import cv2
import ffmpeg
import math
import numpy as np
import csv
import glob
import mediapipe as mp
import concurrent.futures
from deepface import DeepFace
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# Ensure TensorFlow allows GPU memory growth to prevent potential memory issues
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"

def initialize_landmarker(model_path):
    """
    Initialize and return a Face Landmarker model from MediaPipe.
    
    This function sets up the MediaPipe Face Landmarker with the specified model,
    enabling facial landmark detection in images.
    
    Parameters:
        model_path (str): Path to the Face Landmarker .task model.
    
    Returns:
        vision.FaceLandmarker: Initialized Face Landmarker model.
    """
    # Import necessary classes from MediaPipe
    BaseOptions = python.BaseOptions
    FaceLandmarkerOptions = vision.FaceLandmarkerOptions
    VisionRunningMode = vision.RunningMode

    # Configure the base options with the provided model path
    base_options = BaseOptions(model_asset_path=model_path)
    
    # Set up FaceLandmarkerOptions with desired parameters
    options = FaceLandmarkerOptions(
        base_options=base_options,
        output_facial_transformation_matrixes=True,
        running_mode=VisionRunningMode.IMAGE
    )
    
    # Create and return the FaceLandmarker instance
    return vision.FaceLandmarker.create_from_options(options)

def process_single_video(video_paths, target_face_path, random_id_folder_path, model_path, gap_threshold):
    """
    Process a single video by initializing the landmarker and extracting relevant face segments.
    
    This function initializes the MediaPipe Face Landmarker and delegates the processing
    of the video to the `process_video_with_ffmpeg` function.
    
    Parameters:
        video_paths (list): List of video file paths to process.
        target_face_path (str): Path to the target face image for verification.
        random_id_folder_path (str): Path to the folder containing video segments.
        model_path (str): Path to the Face Landmarker .task model.
        gap_threshold (int): Threshold for gaps between detected segments in frames.
    
    Returns:
        list: List of processed video paths.
    """
    # Initialize the MediaPipe Face Landmarker
    landmarker = initialize_landmarker(model_path)
    
    # Process the video with ffmpeg and MediaPipe
    process_video_with_ffmpeg(video_paths, target_face_path, random_id_folder_path, landmarker, gap_threshold)
    
    return video_paths

def calculate_head_pose(rotation_matrix):
    """
    Compute Pitch, Yaw, and Roll angles from a rotation matrix.
    
    This function decomposes the rotation matrix to extract the head's orientation
    in terms of pitch, yaw, and roll angles.
    
    Parameters:
        rotation_matrix (np.array): 3x3 rotation matrix obtained from MediaPipe.
    
    Returns:
        tuple: (pitch, yaw, roll) angles in degrees.
    """
    # Calculate the sine of the pitch angle
    sy = math.sqrt(rotation_matrix[0, 0] ** 2 + rotation_matrix[1, 0] ** 2)
    singular = sy < 1e-6  # Check for gimbal lock

    if not singular:
        # Compute angles when the system is not in gimbal lock
        x = math.atan2(rotation_matrix[2, 1], rotation_matrix[2, 2])
        y = math.atan2(-rotation_matrix[2, 0], sy)
        z = math.atan2(rotation_matrix[1, 0], rotation_matrix[0, 0])
    else:
        # Compute angles when the system is in gimbal lock
        x = math.atan2(-rotation_matrix[1, 2], rotation_matrix[1, 1])
        y = math.atan2(-rotation_matrix[2, 0], sy)
        z = 0

    # Convert radians to degrees
    return np.degrees(x), np.degrees(y), np.degrees(z)

def estimate_head_pose(image_bgr, landmarker):
    """
    Estimate head pose from an image using MediaPipe's Face Landmarker.
    
    This function detects facial landmarks in the given image and computes the
    head's pitch, yaw, and roll angles based on the facial transformation matrix.
    
    Parameters:
        image_bgr (np.array): Input image in BGR format.
        landmarker (vision.FaceLandmarker): Initialized Face Landmarker model.
    
    Returns:
        dict or None: Dictionary containing 'pitch', 'yaw', and 'roll' angles if a face is detected,
                      otherwise None.
    """
    # Convert the image from BGR to RGB as MediaPipe expects RGB images
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    
    # Create a MediaPipe Image object
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)
    
    # Detect facial landmarks using the Face Landmarker
    results = landmarker.detect(mp_image)

    if results.face_landmarks:
        # Extract the facial transformation matrix for the first detected face
        rotation_matrix = results.facial_transformation_matrixes[0]
        
        # Calculate pitch, yaw, and roll angles from the rotation matrix
        pitch, yaw, roll = calculate_head_pose(rotation_matrix)
        
        return {"pitch": pitch, "yaw": yaw, "roll": roll}

    # Return None if no face is detected
    return None

def process_video_with_ffmpeg(
    video_paths,
    target_face_path,
    random_id_folder_path,
    landmarker,
    gap_threshold=10
):
    """
    Process videos to detect segments where the target face is present 
    and has an acceptable head pose, then extract and save these segments.
    
    This function performs the following steps:
        1. Loads the target face image for verification.
        2. Iterates through each video, detecting frames where the target face appears
           with a head pose within specified thresholds.
        3. Identifies continuous segments based on detected frames and gap thresholds.
        4. Extracts the identified segments using ffmpeg and saves them to the output directory.
        5. Records the segment details in a CSV file.
    
    Parameters:
        video_paths (list): List of video file paths to process.
        target_face_path (str): Path to the target face image for verification.
        random_id_folder_path (str): Path to the folder where processed segments will be saved.
        landmarker (vision.FaceLandmarker): Initialized Face Landmarker model.
        gap_threshold (int, optional): Threshold in frames to determine gaps between segments. Defaults to 10.
    """
    # Define the expansion ratio for the bounding box around detected faces
    EXPAND_RATIO = 1.3  # The factor by which the bounding box is enlarged

    print("VIDEO_PATHS:", video_paths)

    # Load the target face image using OpenCV
    target_face = cv2.imread(target_face_path)
    if target_face is None:
        print(f"Could not read the target face image from {target_face_path}.")
        return

    # Create the 'processed_videos' directory if it doesn't exist
    processed_videos_dir = os.path.join(random_id_folder_path, "processed_videos")
    os.makedirs(processed_videos_dir, exist_ok=True)

    # Iterate through each video to process
    for video_path in video_paths:
        video_name = os.path.basename(video_path)
        print("PROCESSING VIDEO:", video_name)
        video_basename, _ = os.path.splitext(video_name)

        # Create a directory for the current video inside 'processed_videos'
        video_output_dir = os.path.join(processed_videos_dir, video_basename)
        os.makedirs(video_output_dir, exist_ok=True)

        # Open the video file using OpenCV
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Could not open video: {video_path}")
            continue

        # Retrieve video properties: frames per second (fps) and total frame count
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_timestamps = np.arange(0, frame_count) / fps  # Array of timestamps in seconds

        frame_number = 0  # Initialize frame counter
        active_frames = []  # List to store timestamps of frames where the target face is detected

        # Iterate through each frame in the video
        while True:
            ret, frame = cap.read()
            if not ret:
                break  # Exit loop if no frame is returned
            frame_number += 1

            # Detect faces in the current frame using DeepFace
            try:
                faces = DeepFace.extract_faces(img_path=frame, enforce_detection=False)
            except Exception as e:
                print(f"Error processing frame {frame_number} of {video_path}: {e}")
                continue

            # Iterate through each detected face in the frame
            for face_info in faces:
                # Retrieve the facial area bounding box
                facial_area = face_info.get("facial_area", None)
                if facial_area is None:
                    continue  # Skip if facial area is not available

                # Verify the detected face against the target face using DeepFace
                try:
                    # DeepFace.extract_faces() returns faces as float arrays [0..1]
                    result = DeepFace.verify(
                        target_face, 
                        face_info["face"],  # Face image for verification
                        enforce_detection=False
                    )
                    if not result["verified"]:
                        continue  # Skip if the face does not match the target
                except Exception:
                    # Skip if verification or processing fails
                    continue

                # ---------------------------------------------------------------
                # 1) Expand the bounding box by the defined ratio and clamp to frame boundaries
                # ---------------------------------------------------------------
                fx = facial_area["x"]
                fy = facial_area["y"]
                fw = facial_area["w"]
                fh = facial_area["h"]

                # Calculate the center of the bounding box
                cx = fx + fw / 2.0
                cy = fy + fh / 2.0
                new_w = fw * EXPAND_RATIO
                new_h = fh * EXPAND_RATIO

                # Calculate the new bounding box coordinates
                nx1 = int(cx - new_w / 2.0)
                ny1 = int(cy - new_h / 2.0)
                nx2 = int(nx1 + new_w)
                ny2 = int(ny1 + new_h)

                # Clamp the bounding box to the frame dimensions
                frame_height, frame_width = frame.shape[:2]
                nx1 = max(0, nx1)
                ny1 = max(0, ny1)
                nx2 = min(frame_width, nx2)
                ny2 = min(frame_height, ny2)

                # Validate the expanded bounding box dimensions
                expanded_w = nx2 - nx1
                expanded_h = ny2 - ny1
                if expanded_w < 1 or expanded_h < 1:
                    continue  # Skip if the bounding box is too small

                # ---------------------------------------------------------------
                # 2) Crop the expanded bounding box from the original frame
                # ---------------------------------------------------------------
                face_crop_bgr = frame[ny1:ny2, nx1:nx2]
                if face_crop_bgr.size == 0:
                    continue  # Skip if the cropped face is empty

                # Convert the cropped face image from BGR to RGB
                face_crop_rgb = cv2.cvtColor(face_crop_bgr, cv2.COLOR_BGR2RGB)

                # ---------------------------------------------------------------
                # 3) Pass the cropped face to the head-pose estimator
                # ---------------------------------------------------------------
                pose = estimate_head_pose(face_crop_rgb, landmarker)
                if pose is None:
                    continue  # Skip if head pose estimation fails

                # Example criterion: check if yaw angle is less than 20 degrees
                yaw = pose.get("yaw", 9999)
                if abs(yaw) < 20:
                    # Record the timestamp of the frame if it meets the criterion
                    active_frames.append(frame_timestamps[frame_number - 1])

                # Since only one face match is needed per frame, break after processing
                break

        # Release the video capture object
        cap.release()

        # ---------------------------------------------------------------
        # Derive continuous segments based on detected active frames
        # ---------------------------------------------------------------
        segments = []
        if active_frames:
            start_time = active_frames[0]
            for i in range(1, len(active_frames)):
                # Determine if the gap between consecutive frames exceeds the threshold
                if (active_frames[i] - active_frames[i - 1]) > (gap_threshold / fps):
                    end_time = active_frames[i - 1] - (gap_threshold / fps)
                    if end_time < start_time:
                        end_time = start_time
                    segments.append((start_time, end_time))
                    start_time = active_frames[i]
            # Append the last segment
            end_time = active_frames[-1]
            if end_time > start_time:
                segments.append((start_time, end_time))

        # Filter out segments that are shorter than 1 second
        segments = [(s, e) for (s, e) in segments if (e - s) >= 1.0]
        print(f"Segments for {video_path}: {segments}")

        # ---------------------------------------------------------------
        # Extract and save the identified segments using ffmpeg
        # ---------------------------------------------------------------
        for idx, (start, end) in enumerate(segments):
            output_filename = f"{video_basename}_segment_{idx + 1}.mp4"
            output_filepath = os.path.join(video_output_dir, output_filename)
            try:
                (
                    ffmpeg
                    .input(video_path, ss=start, to=end)
                    .output(output_filepath, c="copy")
                    .run(overwrite_output=True)
                )
                print(f"Saved segment {idx + 1} to {output_filepath}")
            except ffmpeg.Error as e:
                print(f"Error extracting segment {idx + 1} from {video_path}: {e.stderr.decode()}")

        # ---------------------------------------------------------------
        # Save the segment details to a CSV file for reference
        # ---------------------------------------------------------------
        csv_filename = os.path.join(video_output_dir, "segments.csv")
        try:
            with open(csv_filename, mode="w", newline="") as csvfile:
                csv_writer = csv.writer(csvfile)
                csv_writer.writerow(["Segment Name", "Start Time (s)", "End Time (s)"])  # Header
                for idx, (start, end) in enumerate(segments):
                    output_filename = f"{video_basename}_segment_{idx + 1}.mp4"
                    csv_writer.writerow([output_filename, round(start, 2), round(end, 2)])
            print(f"Processed videos summary saved in: {video_output_dir}")
        except Exception as e:
            print(f"Error saving segments CSV for {video_path}: {e}")

    def main(base_folder_path, model_path, prefix, start_num, end_num, num_workers):
        """
        Main function to process a range of subfolders and extract relevant video segments.
        
        This function performs the following steps:
            1. Iterates through a specified range of subfolders based on the prefix and indices.
            2. Identifies video files and associated metadata within each subfolder.
            3. Prepares tasks for concurrent processing of videos.
            4. Utilizes a ProcessPoolExecutor to process multiple videos in parallel.
            5. Handles any exceptions that occur during the processing of videos.
        
        Parameters:
            base_folder_path (str): Base directory containing 'CN'/'PD' subfolders.
            model_path (str): Path to the Face Landmarker .task model.
            prefix (str): Prefix for subfolder naming (e.g., "PD", "CN").
            start_num (int): Starting index for processing subfolders.
            end_num (int): Ending index for processing subfolders.
            num_workers (int): Number of parallel worker processes to use.
        """
        # Iterate through the specified range of subfolders
        for i in range(start_num, end_num + 1):
            # Construct the subfolder name with leading zeros (e.g., "pd_01")
            subfolder_name = f"{prefix.lower()}_{i:02d}"
            folder_path = os.path.join(base_folder_path, prefix, subfolder_name)
            if not os.path.exists(folder_path):
                print(f"Subfolder does not exist: {folder_path}. Skipping.")
                continue

            tasks = []  # List to hold tasks for concurrent processing

            # Iterate through each random ID folder within the current subfolder
            for random_id_folder in os.listdir(folder_path):
                random_id_path = os.path.join(folder_path, random_id_folder)
                if not os.path.isdir(random_id_path):
                    print("Skipping non-directory:", random_id_path)
                    continue

                target_face_path = None  # Path to the target face image
                video_paths = []         # List to hold paths of video files

                # Locate necessary files within the random ID directory
                for file in os.listdir(random_id_path):
                    if file.endswith(".mp4") and file.startswith(prefix):
                        video_paths.append(os.path.join(random_id_path, file))
                    elif file.lower() == "target_face.png":
                        target_face_path = os.path.join(random_id_path, file)

                # Proceed only if both video files and target face image are found
                if video_paths and target_face_path:
                    tasks.append((video_paths, target_face_path, random_id_path, model_path, 5))
                else:
                    # Inform the user about missing required files
                    missing_files = []
                    if not video_paths:
                        missing_files.append("video files (*.mp4)")
                    if not target_face_path:
                        missing_files.append("target_face.png")
                    print(f"Missing required files in {random_id_path}: {', '.join(missing_files)}")

            if not tasks:
                print(f"No valid tasks found in subfolder: {folder_path}")
                continue

            # Use ProcessPoolExecutor for parallel processing of videos
            with concurrent.futures.ProcessPoolExecutor(max_workers=num_workers) as executor:
                # Submit all tasks to the executor
                futures = [
                    executor.submit(
                        process_single_video, 
                        video_paths, 
                        target_face_path, 
                        random_id_folder_path, 
                        model_path, 
                        5
                    )
                    for video_paths, target_face_path, random_id_folder_path, _, _ in tasks
                ]

                # Iterate through completed futures to handle results and exceptions
                for future in concurrent.futures.as_completed(futures):
                    try:
                        # Retrieve the result of the future
                        result = future.result()
                        print(f"Completed processing for videos: {result}")
                    except Exception as exc:
                        # Handle any exceptions that occurred during processing
                        print(f"An error occurred during video processing: {exc}")

    if __name__ == "__main__":
        import multiprocessing as multi
        # Set the start method for multiprocessing to 'spawn' for compatibility
        multi.set_start_method('spawn')

        """
        Example usage:
            python segmentClips.py <base_folder_path> <model_path> <prefix> <start_num> <end_num> [num_workers]
        
        Parameters:
            <base_folder_path> : Path to the base folder of the ParkCeleb dataset containing 'PD'/'CN' subfolders.
            <model_path>       : Path to the MediaPipe Face Landmarker .task model.
            <prefix>           : Prefix for subfolder naming (e.g., "PD", "CN").
            <start_num>        : Starting index of subfolders to process.
            <end_num>          : Ending index of subfolders to process.
            [num_workers]      : (Optional) Number of parallel worker processes to use. Defaults to CPU count.
        """
        # Check if the correct number of command-line arguments is provided
        if len(sys.argv) == 6 or len(sys.argv) == 7:
            print("Usage: python script.py <base_folder_path> <model_path> <prefix> <start_num> <end_num> [num_workers]")
            sys.exit(1)

        # Parse command-line arguments
        base_folder_path = sys.argv[1]
        model_path = sys.argv[2]
        prefix = sys.argv[3]
        try:
            start_num = int(sys.argv[4])
            end_num = int(sys.argv[5])
        except ValueError:
            print("Error: start_num and end_num must be integers.")
            sys.exit(1)
        
        # Determine the number of worker processes
        if len(sys.argv) == 7:
            try:
                num_workers = int(sys.argv[6])
            except ValueError:
                print("Error: num_workers must be an integer.")
                sys.exit(1)
        else:
            num_workers = multi.cpu_count()  # Default to the number of CPU cores

        # Execute the main processing function
        main(base_folder_path, model_path, prefix, start_num, end_num, num_workers)
