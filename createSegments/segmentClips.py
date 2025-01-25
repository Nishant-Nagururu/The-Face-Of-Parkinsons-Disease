import cv2
from deepface import DeepFace
import numpy as np
import os
import sys
import glob
import ffmpeg
import math
import mediapipe as mp
import concurrent.futures
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

def initialize_landmarker(model_path):
    from mediapipe.tasks import python
    from mediapipe.tasks.python import vision

    BaseOptions = python.BaseOptions
    FaceLandmarkerOptions = vision.FaceLandmarkerOptions
    VisionRunningMode = vision.RunningMode

    base_options = BaseOptions(model_asset_path=model_path)
    options = FaceLandmarkerOptions(
        base_options=base_options,
        output_facial_transformation_matrixes=True,
        running_mode=VisionRunningMode.IMAGE  # IMAGE for single-frame detection
    )
    return vision.FaceLandmarker.create_from_options(options)

def calculate_head_pose(rotation_matrix):
    sy = math.sqrt(rotation_matrix[0, 0] ** 2 + rotation_matrix[1, 0] ** 2)
    singular = sy < 1e-6

    if not singular:
        # Pitch (x), Yaw (y), Roll (z)
        x = math.atan2(rotation_matrix[2, 1], rotation_matrix[2, 2])  
        y = math.atan2(-rotation_matrix[2, 0], sy)                   
        z = math.atan2(rotation_matrix[1, 0], rotation_matrix[0, 0]) 
    else:
        x = math.atan2(-rotation_matrix[1, 2], rotation_matrix[1, 1])
        y = math.atan2(-rotation_matrix[2, 0], sy)
        z = 0

    # Convert radians to degrees
    pitch_deg = np.degrees(x)
    yaw_deg   = np.degrees(y)
    roll_deg  = np.degrees(z)

    return pitch_deg, yaw_deg, roll_deg

def estimate_head_pose(image_bgr, landmarker):
    """
    Estimate head pose from a single face in image_bgr using
    MediaPipe Face Landmarker transformation matrix.
    Returns a dict with {yaw, pitch, roll} or None if no face found.
    """
    # Convert to RGB
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)

    # Detect face(s)
    results = landmarker.detect(mp_image)

    # If at least one face was detected
    if results.face_landmarks:
        # Extract the first face's transformation matrix
        rotation_matrix = results.facial_transformation_matrixes[0]

        # Calculate pitch, yaw, roll
        pitch, yaw, roll = calculate_head_pose(rotation_matrix)

        return {
            "pitch": pitch,
            "yaw":   yaw,
            "roll":  roll
        }

    return None

# --------------------------------------------
# The rest of your script: DeepFace checks, folder traversal, etc.
# --------------------------------------------
def process_video_with_ffmpeg(video_root, target_face_path, random_id_folder_path, landmarker, gap_threshold=5):
    # Load the target face image
    target_face = cv2.imread(target_face_path)
    if target_face is None:
        print(f"Could not read the target face image from {target_face_path}.")
        return

    # Create the processed_videos directory if it doesn't exist
    processed_videos_dir = os.path.join(random_id_folder_path, "processed_videos")
    os.makedirs(processed_videos_dir, exist_ok=True)

    # Find .mp4 videos in random_id_folder_path that start with video_root
    video_paths = glob.glob(os.path.join(random_id_folder_path, f"{video_root}*.mp4"))
    if not video_paths:
        print(f"No videos found starting with {video_root} in {random_id_folder_path}.")
        return

    # For each matching video
    for video_path in video_paths:
        video_name = os.path.basename(video_path)
        
        print("PROCESSING VIDEO", video_name)
        
        video_basename, _ = os.path.splitext(video_name)

        # Create a directory for this video inside processed_videos
        video_output_dir = os.path.join(processed_videos_dir, video_basename)
        os.makedirs(video_output_dir, exist_ok=True)

        # Open the video file to detect faces
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Could not open video: {video_path}")
            continue

        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_timestamps = np.arange(0, frame_count) / fps  # Array of frame timestamps in seconds

        frame_number = 0
        active_frames = []  # Store frame timestamps where the target face is detected

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame_number += 1

            # Detect faces in the frame using DeepFace
            try:
                faces = DeepFace.extract_faces(img_path=frame, enforce_detection=False)
            except Exception as e:
                print(f"Error processing frame {frame_number} of {video_path}: {e}")
                continue

            # For each detected face, verify against the target
            for face in faces:
                face_img = face["face"]

                # Compare the detected face with the target face
                try:
                    result = DeepFace.verify(target_face, face_img, enforce_detection=False)
                    if result["verified"]:
                        # -------------------------------
                        # 1. Calculate the face area % from DeepFace bounding box
                        facial_area = face["facial_area"]
                        face_area = facial_area["w"] * facial_area["h"]
                        image_area = frame.shape[1] * frame.shape[0]
                        face_percentage = (face_area / image_area) * 100

                        # -------------------------------
                        # 2. If we pass area threshold, check head pose using MediaPipe
                        if face_percentage >= 0.0:
                            # a) Convert face_img to uint8 so OpenCV can process it
                            face_img_uint8 = (face_img * 255).astype(np.uint8)

                            # b) Convert face image to RGB
                            rgb_face_img = cv2.cvtColor(face_img_uint8, cv2.COLOR_BGR2RGB)

                            # c) Use our new Face Landmarker approach
                            pose = estimate_head_pose(rgb_face_img, landmarker)
                            if pose is None:
                                continue

                            yaw = pose["yaw"]

                            # If absolute yaw < 60 degrees, consider face "straight enough"
                            if abs(yaw) < 100:
                                active_frames.append(frame_timestamps[frame_number - 1])
                                
                        # Break out if we see the target face regardless of wheter it qualifies as an active frame
                        break

                except Exception as e:
                    # Skip if verification or processing fails
                    continue

        cap.release()

        # Process active_frames into time-based segments
        if not active_frames:
            print(f"No segments found in video: {video_path}")
            continue

        # Determine segments based on gap_threshold
        segments = []
        start_time = active_frames[0]
        for i in range(1, len(active_frames)):
            # Check if the gap between frames is larger than the allowed threshold
            if active_frames[i] - active_frames[i - 1] > gap_threshold / fps:
                # End this segment 5 frames earlier to account for buffer
                end_time = active_frames[i - 1] - (gap_threshold / fps)

                # Ensure we don't get a negative or inverted segment
                if end_time < start_time:
                    end_time = start_time

                segments.append((start_time, end_time))

                # Start the next segment at the current frame's timestamp
                start_time = active_frames[i]

        # Add the last segment if there's any leftover
        end_time = active_frames[-1]
        if end_time > start_time:
            segments.append((start_time, end_time))

        # Filter out very short segments (< 1 sec)
        segments = [(s, e) for (s, e) in segments if (e - s) >= 1.0]

        print(f"Segments for {video_path}: {segments}")

        # Extract segments using ffmpeg
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
                print(f"Error extracting segment {idx + 1} from {video_path}: {e}")

        import csv

        # After extracting segments using ffmpeg, save the details to a CSV file
        csv_filename = os.path.join(video_output_dir, "segments.csv")
        with open(csv_filename, mode="w", newline="") as csvfile:
            csv_writer = csv.writer(csvfile)
            csv_writer.writerow(["Segment Name", "Start Time (s)", "End Time (s)"])  # Header row

            for idx, (start, end) in enumerate(segments):
                output_filename = f"{video_basename}_segment_{idx + 1}.mp4"
                csv_writer.writerow([output_filename, round(start, 2), round(end, 2)])  # Write segment details

        print(f"Proceessed videos summary saved in: {video_output_dir}")
    
    print(f"Processed video segments saved in: {processed_videos_dir}")


def main(base_folder_path, landmarker):
    for folder_type in ["PD"]:
        folder_path = os.path.join(base_folder_path, folder_type)
        if not os.path.exists(folder_path):
            print(f"Folder {folder_path} does not exist. Skipping.")
            continue

        for subfolder in os.listdir(folder_path):
            subfolder_path = os.path.join(folder_path, subfolder)

            # Iterate through folders with random IDs
            for random_id_folder in os.listdir(subfolder_path):

                random_id_path = os.path.join(subfolder_path, random_id_folder)

                # skips metadata.xlsx files
                if not os.path.isdir(random_id_path):
                    print("skipping", random_id_path)
                    continue

                # only processes folders that have a target_face.png and have the speaker segments from running saveClips.py
                split_videos = False
                target_face_path = None

                for file in os.listdir(random_id_path):
                    if file.endswith(".mp4") and file.startswith(folder_type):
                        split_videos = True
                    elif file == "target_face.png":
                        target_face_path = os.path.join(random_id_path, file)

                if split_videos and target_face_path:
                    try:
                        process_video_with_ffmpeg(folder_type, target_face_path, random_id_path, landmarker)
                    except Exception as e:
                        print(f"Error processing {random_id_path}: {e}")
                else:
                    print(f"Missing required files in {random_id_path}")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python script.py <base_folder_path> <face_landmarker_v2_with_blendshapes.task path>")
        sys.exit(1)
    base_folder_path = sys.argv[1]
    model_path = sys.argv[2]
    landmarker = initialize_landmarker(model_path)
    main(base_folder_path, landmarker)
