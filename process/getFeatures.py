import os
import sys
import cv2
import subprocess
import numpy as np
import pandas as pd
import json
import math
import mediapipe as mp
import concurrent.futures
from deepface import DeepFace
from scipy.ndimage import median_filter

# Allow TensorFlow to dynamically grow the GPU memory if used by DeepFace
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"

def get_ffmpeg_video_metadata(video_path):
    """
    Retrieves video metadata using ffprobe.
    
    This function extracts the frame rate, total number of frames, and duration of the video
    by invoking ffprobe through subprocess. It first retrieves general format information,
    then specifically extracts video stream details.

    Parameters:
        video_path (str): Path to the video file.

    Returns:
        tuple:
            frame_rate (float or None): Frames per second of the video.
            total_frames (int or None): Total number of frames in the video.
            duration (float): Duration of the video in seconds.
    """
    # Command to retrieve video metadata in JSON format
    cmd = f'ffprobe -v quiet -print_format json -show_format -show_streams "{video_path}"'
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    metadata = json.loads(result.stdout)
    
    # Extract duration from format information
    duration = float(metadata["format"]["duration"])
    frame_rate = None
    total_frames = None
    
    # Iterate through streams to find video stream details
    for stream in metadata["streams"]:
        if stream["codec_type"] == "video":
            frame_rate_str = stream.get("avg_frame_rate", "0/0")
            try:
                num, den = map(int, frame_rate_str.split('/'))
                frame_rate = num / den if den != 0 else None
            except ValueError:
                frame_rate = None
            break
    
    # Alternative method to get total frames directly from ffprobe
    frame_cmd = (
        f'ffprobe -v error -select_streams v:0 -count_frames '
        f'-show_entries stream=nb_read_frames -of default=nokey=1:noprint_wrappers=1 "{video_path}"'
    )
    frame_result = subprocess.run(frame_cmd, shell=True, capture_output=True, text=True)
    
    # Parse total frames from ffprobe output
    if frame_result.stdout.strip().isdigit():
        total_frames = int(frame_result.stdout.strip())
    elif frame_rate:
        total_frames = int(duration * frame_rate)  # Fallback calculation
    
    return frame_rate, total_frames, duration

def detect_features_with_deepface(
    video_path, 
    target_face_path, 
    years_val, 
    landmarker, 
    close_threshold=0.5
):
    """
    Processes a single video to extract facial features and metrics.
    
    This function performs the following steps:
        1. Retrieves video metadata (frame rate, total frames, duration) via ffprobe.
        2. Reads frames using OpenCV and detects faces using DeepFace.
        3. Verifies detected faces against a target face if multiple faces are present.
        4. Expands the bounding box around detected faces and clamps it to frame boundaries.
        5. Uses MediaPipe's Face Landmarker to detect facial landmarks and blendshapes.
        6. Counts blinks and calculates mouth movement metrics.
        7. Aggregates and returns statistics for the video.

    Parameters:
        video_path (str): Path to the video file.
        target_face_path (str or None): Path to the target face image for verification.
        years_val (float or int): Additional metadata value (e.g., years from diagnosis).
        landmarker: MediaPipe FaceLandmarker instance.
        close_threshold (float): Threshold to determine if eyes are closed.

    Returns:
        dict or None: Aggregated statistics for the video or None if processing failed.
    """
    # Constants for bounding-box expansion
    EXPAND_RATIO = 1.3

    # 1) Get metadata using ffprobe
    frame_rate, total_frames, duration_sec = get_ffmpeg_video_metadata(video_path)
    if total_frames is None:
        print(f"Warning: Could not determine total_frames for {video_path}. Using fallback.")
        total_frames = 99999999  # Fallback sentinel

    # Open the video capture to read frames
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Unable to open video {video_path}")
        return None

    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_area = frame_width * frame_height

    # Load target face if provided
    target_face = None
    if target_face_path:
        try:
            target_face = DeepFace.extract_faces(
                img_path=target_face_path, 
                enforce_detection=False
            )[0]["face"]
        except Exception as e:
            print(f"ERROR reading target face {target_face_path}: {e}")
            return None

    # Initialize blink counts and previous eye states
    blink_count = 0
    prev_left_eye_closed = False
    prev_right_eye_closed = False

    # Lists for metrics
    mouth_movement_dists = []
    frame_data = []
    for fidx in range(1, total_frames + 1):
        frame_data.append({
            "frame_number": fidx,
            "bounding_box_x": None,
            "bounding_box_y": None,
            "bounding_box_w": None,
            "bounding_box_h": None,
            "landmarks": None,  # Stored as JSON
            "mouth_movement_dist": None
        })

    # Landmark indices based on MediaPipe's face mesh
    LEFT_EYE_MEDIAL_CORNER_IDX = 133
    RIGHT_EYE_MEDIAL_CORNER_IDX = 362
    JAW_LOWEST_IDX = 152
    MIDDLE_UPPER_LIP_IDX = 0

    def get_xy(landmarks, idx):
        """
        Retrieves the normalized x and y coordinates of a specific landmark.
        
        Parameters:
            landmarks (list): List of facial landmarks.
            idx (int): Index of the desired landmark.
        
        Returns:
            tuple: (x, y) coordinates.
        """
        lm = landmarks[idx]
        return lm.x, lm.y  # Normalized [0,1]

    frame_number = 0
    processed_frame_count = 0
    while True:
        ret, frame_bgr = cap.read()
        if not ret or frame_bgr is None:
            break

        frame_number += 1
        if frame_number > total_frames:
            break  # Safety check

        processed_frame_count = frame_number

        # Initialize mouth distance for this frame
        mouth_movement_dists.append(np.nan)

        # Convert the BGR frame to RGB for DeepFace
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        try:
            faces = DeepFace.extract_faces(
                img_path=frame_rgb, 
                enforce_detection=False
            )
        except Exception as e:
            print(f"Error processing frame {frame_number} of {video_path}: {e}")
            continue

        if not faces:
            continue  # No faces detected in this frame

        verified_face_found = False
        verified_facial_area = None

        # If exactly 1 face, treat as verified automatically
        if len(faces) == 1:
            verified_face_found = True
            verified_facial_area = faces[0].get("facial_area", None)
        elif target_face is not None:
            # Multiple faces: verify against target_face
            for face_dict in faces:
                try:
                    candidate_face = face_dict["face"]
                    result = DeepFace.verify(
                        target_face, 
                        candidate_face,
                        enforce_detection=False
                    )
                    if result["verified"]:
                        verified_face_found = True
                        verified_facial_area = face_dict.get("facial_area", None)
                        break  # Process only the first verified face
                except Exception as e:
                    print(f"[Frame {frame_number}] Error verifying face: {e}")
                    continue
        else:
            # Multiple faces but no target_face provided
            continue

        if not verified_face_found or not verified_facial_area:
            continue  # No verified face found

        # Extract bounding box coordinates
        fx = verified_facial_area["x"]
        fy = verified_facial_area["y"]
        fw = verified_facial_area["w"]
        fh = verified_facial_area["h"]

        # Expand bounding box by EXPAND_RATIO (1.3x) around center
        cx = fx + fw / 2.0
        cy = fy + fh / 2.0
        new_w = fw * EXPAND_RATIO
        new_h = fh * EXPAND_RATIO
        nx1 = int(cx - new_w / 2.0)
        ny1 = int(cy - new_h / 2.0)
        nx2 = int(nx1 + new_w)
        ny2 = int(ny1 + new_h)

        # Clamp coordinates to frame boundaries
        nx1 = max(0, nx1)
        ny1 = max(0, ny1)
        nx2 = min(frame_width, nx2)
        ny2 = min(frame_height, ny2)

        expanded_w = nx2 - nx1
        expanded_h = ny2 - ny1
        if expanded_w < 1 or expanded_h < 1:
            continue  # Invalid bounding box after expansion

        # Save expanded bounding box to frame_data
        frame_data[frame_number - 1]["bounding_box_x"] = nx1
        frame_data[frame_number - 1]["bounding_box_y"] = ny1
        frame_data[frame_number - 1]["bounding_box_w"] = expanded_w
        frame_data[frame_number - 1]["bounding_box_h"] = expanded_h

        # Crop the expanded bounding box from the original frame
        face_crop_bgr = frame_bgr[ny1:ny2, nx1:nx2]
        if face_crop_bgr.size == 0:
            continue  # Empty crop due to clamping

        # Convert to RGB for MediaPipe
        face_crop_rgb = cv2.cvtColor(face_crop_bgr, cv2.COLOR_BGR2RGB)

        # Create a MediaPipe Image object
        mp_frame = mp.Image(image_format=mp.ImageFormat.SRGB, data=face_crop_rgb)
        try:
            detection_result = landmarker.detect(mp_frame)
        except Exception as e:
            print(f"[Frame {frame_number}] MediaPipe detection error: {e}")
            continue

        if not detection_result.face_landmarks:
            continue  # No landmarks detected

        # Blink detection using blendshapes
        if detection_result.face_blendshapes:
            blendshapes = detection_result.face_blendshapes[0]
            left_blink_score = 0.0
            right_blink_score = 0.0

            for bshape in blendshapes:
                cat = bshape.category_name.lower()
                if cat == 'eyeblinkleft':
                    left_blink_score = bshape.score
                elif cat == 'eyeblinkright':
                    right_blink_score = bshape.score

            left_eye_closed = (left_blink_score > close_threshold)
            right_eye_closed = (right_blink_score > close_threshold)

            if left_eye_closed and not prev_left_eye_closed:
                blink_count += 1
            if right_eye_closed and not prev_right_eye_closed:
                blink_count += 1

            prev_left_eye_closed = left_eye_closed
            prev_right_eye_closed = right_eye_closed

        # Landmark-based metrics
        face_landmarks = detection_result.face_landmarks[0]

        # Store all landmarks (x, y, z) as JSON
        all_landmarks = []
        for lm in face_landmarks:
            # These x, y coordinates are normalized relative to the cropped image
            all_landmarks.append([lm.x, lm.y, lm.z])
        frame_data[frame_number - 1]["landmarks"] = json.dumps(all_landmarks)

        # Compute mouth movement distance
        lx, ly = get_xy(face_landmarks, LEFT_EYE_MEDIAL_CORNER_IDX)
        rx, ry = get_xy(face_landmarks, RIGHT_EYE_MEDIAL_CORNER_IDX)
        inter_eye_dist = math.sqrt((rx - lx)**2 + (ry - ly)**2)
        if inter_eye_dist < 1e-7:
            continue  # Prevent division by zero

        top_lip_x, top_lip_y = get_xy(face_landmarks, MIDDLE_UPPER_LIP_IDX)
        chin_x, chin_y = get_xy(face_landmarks, JAW_LOWEST_IDX)
        mm_dist = math.sqrt((top_lip_x - chin_x)**2 + (top_lip_y - chin_y)**2)
        norm_mm_dist = mm_dist / inter_eye_dist

        mouth_movement_dists[-1] = norm_mm_dist
        frame_data[frame_number - 1]["mouth_movement_dist"] = norm_mm_dist

    cap.release()

    # Truncate frame_data to the actual processed_frame_count
    frame_data = frame_data[:processed_frame_count]
    mouth_movement_dists = mouth_movement_dists[:processed_frame_count]

    # Save bounding box & landmarks info to bounding_boxes/<video_name>.csv
    bounding_box_folder = os.path.join(os.path.dirname(video_path), "bounding_boxes")
    os.makedirs(bounding_box_folder, exist_ok=True)
    video_basename = os.path.splitext(os.path.basename(video_path))[0]
    bb_csv_path = os.path.join(bounding_box_folder, f"{video_basename}.csv")

    df_frames = pd.DataFrame(frame_data)
    df_frames.to_csv(bb_csv_path, index=False)

    # Final blink stats
    final_blink_count = blink_count // 2  # Each blink counts for both eyes
    blinks_per_second = final_blink_count / duration_sec if duration_sec > 0 else np.nan

    # Mouth movement feature (smoothed and averaged)
    mouth_movement_feature = 0.0
    if mouth_movement_dists:
        mm_series = pd.Series(mouth_movement_dists, dtype=float)
        mm_interp = mm_series.interpolate()  # Fill NaNs via interpolation
        mm_arr = mm_interp.to_numpy()
        if len(mm_arr) > 0:
            filtered = median_filter(mm_arr, size=5)  # Apply median filter for smoothing
            mouth_movement_feature = float(np.sum(filtered) / len(filtered))
        else:
            mouth_movement_feature = 0.0

    # Print summary of the processed video
    print(f"\nFinished processing {video_path}. Stats from FFprobe:")
    if frame_rate:
        print(f"  Frame Rate (ffmpeg):    {frame_rate:.2f} FPS")
    else:
        print(f"  Frame Rate (ffmpeg):    None")
    print(f"  Total Frames (ffmpeg):  {total_frames}")
    print(f"  Duration (sec):         {duration_sec:.2f}")
    print(f"  Processed frames:       {processed_frame_count}")
    print(f"  Blinks (estimated):     {final_blink_count}")
    print(f"  Blinks/sec:             {blinks_per_second:.4f}")
    print(f"  Mouth Movement Feature: {mouth_movement_feature:.4f}")
    print(f"  Per-frame data saved to {bb_csv_path}")

    # Return aggregated statistics as a dictionary
    return {
        "video_name": os.path.basename(video_path),
        "frame_rate": frame_rate,
        "processed_frames": processed_frame_count,
        "blinks_per_second": blinks_per_second,
        "mouth_movement_feature": mouth_movement_feature,
        "raw_mouth_movement_dists": json.dumps(mouth_movement_dists),
        "years_from_diagnosis": years_val
    }

def process_one_video(row, subfolder_path, model_path):
    """
    Worker function for ProcessPoolExecutor.
    
    This function initializes the MediaPipe FaceLandmarker locally to avoid pickling issues
    and processes a single video to extract facial features and metrics.
    
    Parameters:
        row (pd.Series): Row from metadata.csv containing video information.
        subfolder_path (str): Path to the subfolder containing the video and metadata.
        model_path (str): Path to the MediaPipe FaceLandmarker .task file.
    
    Returns:
        dict or None: Aggregated statistics for the video or None if processing failed.
    """
    try:
        import mediapipe as mp

        # Set up the MediaPipe FaceLandmarker with the provided model_path
        BaseOptions = mp.tasks.BaseOptions
        FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
        VisionRunningMode = mp.tasks.vision.RunningMode

        landmarker_options = FaceLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=model_path),
            running_mode=VisionRunningMode.IMAGE,
            output_face_blendshapes=True,
            num_faces=1,
            output_facial_transformation_matrixes=False
        )

        # Initialize the FaceLandmarker instance
        with mp.tasks.vision.FaceLandmarker.create_from_options(landmarker_options) as landmarker:
            # Retrieve video details from the row
            video_name = row.get("video_name", None)
            if not video_name:
                print("Row has no video_name, skipping.")
                return None

            years_val = row.get("years_val", np.nan)
            duration_seconds = row.get("duration_seconds", 0)

            # Skip videos shorter than 5 seconds
            if float(duration_seconds) < 5:
                print(f"Video {video_name} is shorter than 5 seconds; skipping.")
                return None

            video_path = os.path.join(subfolder_path, video_name)
            video_name_no_ext = os.path.splitext(video_name)[0]

            target_face_path = os.path.join(subfolder_path, "target_faces", video_name_no_ext + ".png")
            if not os.path.isfile(video_path):
                print(f"Video not found: {video_path}")
                return None

            print(f"\n--- Processing video: {video_path} ---")
            result = detect_features_with_deepface(
                video_path=video_path,
                target_face_path=target_face_path,
                years_val=years_val,
                landmarker=landmarker,
                close_threshold=0.5
            )
            return result
    except Exception as e:
        print(f"Unexpected error processing video in {subfolder_path}: {e}")
        return None

def main(data_path, model_path, lower_bound, upper_bound, num_workers):
    """
    Main function to process multiple videos across subfolders in parallel.
    
    This function performs the following steps:
        1. Iterates through subfolders within the data_path that fall within the specified index range.
        2. For each eligible subfolder, reads the metadata.csv file containing video information.
        3. Uses ProcessPoolExecutor to process each video in parallel, leveraging multiple CPU cores.
        4. Collects and aggregates the results, saving them to features.csv within each subfolder.
    """
    # Iterate through each subfolder in the output directory
    for subfolder in os.listdir(data_path):
        try:
            # Extract the index from the subfolder name (e.g., "pd_01" -> 1)
            folder_index = int(subfolder.split("_")[-1])
            if folder_index < lower_bound or folder_index > upper_bound:
                continue  # Skip subfolders outside the specified range
        except ValueError:
            # Skip folders that don't end with an integer index
            continue

        subfolder_path = os.path.join(data_path, subfolder)
        if not os.path.isdir(subfolder_path):
            continue  # Ensure the path is a directory

        metadata_csv_path = os.path.join(subfolder_path, "metadata.csv")
        if not os.path.exists(metadata_csv_path):
            print(f"metadata.csv not found in {subfolder_path}, skipping.")
            continue

        df_metadata = pd.read_csv(metadata_csv_path)
        if df_metadata.empty:
            print(f"metadata.csv is empty in {subfolder_path}, skipping.")
            continue

        all_results = []
        with concurrent.futures.ProcessPoolExecutor(max_workers=num_workers) as executor:
            future_to_video = {}
            for idx, row in df_metadata.iterrows():
                # Submit each video processing task to the executor
                future = executor.submit(
                    process_one_video, 
                    row, 
                    subfolder_path, 
                    model_path  # Pass only model_path to avoid pickling issues
                )
                future_to_video[future] = row.get("video_name", "UNKNOWN_VIDEO")

            # Collect the results as they complete
            for future in concurrent.futures.as_completed(future_to_video):
                video_name = future_to_video[future]
                try:
                    result = future.result()
                    if result is not None:
                        all_results.append(result)
                except Exception as exc:
                    print(f"Error in future for video {video_name}: {exc}")

        if all_results:
            # Save the aggregated features to features.csv within the subfolder
            out_csv_path = os.path.join(subfolder_path, "features.csv")
            df_out = pd.DataFrame(all_results)
            df_out.to_csv(out_csv_path, index=False)
            print(f"\nSaved features to {out_csv_path}")
        else:
            print(f"No videos processed or no results for subfolder: {subfolder_path}")

if __name__ == "__main__":
    """
    Example usage:
        python getFeatures.py <data_path> <model_path> <lower_bound> <upper_bound> <num_workers>
    
    Parameters:
          <data_path>  : Directory to The_Face_Of_Parkinsons or The_Face_Of_Parkinsons_CN folders created by compile.py.
          <model_path> : Path to the MediaPipe FaceLandmarker With Blendshapes .task file.
          <lower_bound>: Integer representing the lower index bound of subfolders to process.
          <upper_bound>: Integer representing the upper index bound of subfolders to process.
          <num_workers>: (Optional) Number of parallel worker processes. Defaults to the number of CPU cores.
    """
    import multiprocessing as multi
    try:
        multi.set_start_method('spawn', force=True)
    except RuntimeError:
        pass  # 'spawn' can only be set once

    # Validate the number of command-line arguments
    if len(sys.argv) < 5 or len(sys.argv) > 6:
        print("Usage: python getFeatures.py <data_path> <model_path> <lower_bound> <upper_bound> [num_workers]")
        sys.exit(1)

    # Parse command-line arguments
    data_path = sys.argv[1]
    model_path = sys.argv[2]
    lower_bound = int(sys.argv[3])
    upper_bound = int(sys.argv[4])

    # Determine the number of worker processes
    if len(sys.argv) == 6:
        try:
            num_workers = int(sys.argv[5])
            if num_workers < 1:
                raise ValueError("num_workers must be a positive integer.")
        except ValueError as ve:
            print(f"Invalid num_workers value: {ve}")
            sys.exit(1)
    else:
        num_workers = multi.cpu_count()  # Default to the number of CPU cores

    main(data_path, model_path, lower_bound, upper_bound, num_workers)
