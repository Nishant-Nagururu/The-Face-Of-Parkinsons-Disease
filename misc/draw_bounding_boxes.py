import os
import sys
import cv2
import pandas as pd

def draw_bounding_boxes_on_video(
    video_path, 
    bounding_box_csv_path, 
    output_video_path, 
    box_color=(0, 255, 0), 
    box_thickness=2
):
    """
    Draws bounding boxes on each frame of the input video based on the bounding box CSV.
    
    This function processes an input video frame by frame, overlays bounding boxes as specified
    in a corresponding CSV file, and saves the annotated video to a new file.
    
    Parameters:
        video_path (str): 
            Path to the original input video file.
        
        bounding_box_csv_path (str): 
            Path to the CSV file containing bounding box information. The CSV should have the following columns:
                - bounding_box_x: X-coordinate of the top-left corner of the bounding box.
                - bounding_box_y: Y-coordinate of the top-left corner of the bounding box.
                - bounding_box_w: Width of the bounding box.
                - bounding_box_h: Height of the bounding box.
        
        output_video_path (str): 
            Path where the output video with bounding boxes will be saved.
        
        box_color (tuple, optional): 
            Color of the bounding box in BGR format. Default is green (0, 255, 0).
        
        box_thickness (int, optional): 
            Thickness of the bounding box lines. Default is 2 pixels.
    
    Returns:
        None
    """
    # ---------------------------------------------------
    # 1) Validate Input Paths
    # ---------------------------------------------------
    if not os.path.isfile(video_path):
        print(f"Error: Video file not found at '{video_path}'.")
        sys.exit(1)
    if not os.path.isfile(bounding_box_csv_path):
        print(f"Error: Bounding box CSV file not found at '{bounding_box_csv_path}'.")
        sys.exit(1)
    
    # ---------------------------------------------------
    # 2) Read Bounding Box Data from CSV
    # ---------------------------------------------------
    try:
        df_bb = pd.read_csv(bounding_box_csv_path)
    except Exception as e:
        print(f"Error: Failed to read CSV file '{bounding_box_csv_path}'. Exception: {e}")
        sys.exit(1)
    
    # Validate required columns in the CSV
    required_columns = {"bounding_box_x", "bounding_box_y", "bounding_box_w", "bounding_box_h"}
    if not required_columns.issubset(df_bb.columns):
        print(f"Error: CSV file '{bounding_box_csv_path}' is missing required columns: {required_columns}")
        sys.exit(1)
    
    # ---------------------------------------------------
    # 3) Initialize Video Capture
    # ---------------------------------------------------
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Unable to open video file '{video_path}'.")
        sys.exit(1)
    
    # Retrieve video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # ---------------------------------------------------
    # 4) Initialize Video Writer
    # ---------------------------------------------------
    # Define the codec for the output video. 'mp4v' is widely supported.
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))
    
    # ---------------------------------------------------
    # 5) Processing Loop: Read, Annotate, and Write Frames
    # ---------------------------------------------------
    frame_idx = 0  # Current frame index
    total_bb_entries = len(df_bb)  # Total number of bounding box entries
    print(f"Starting processing of video: '{video_path}'")
    print(f"Total frames in video: {total_frames}")
    print(f"Total bounding box entries: {total_bb_entries}")
    print(f"Output video will be saved to: '{output_video_path}'\n")
    
    while True:
        ret, frame = cap.read()  # Read the next frame
        if not ret:
            break  # Exit loop if no frame is returned (end of video)
    
        if frame_idx >= total_bb_entries:
            # No more bounding box data available; write the frame as-is
            out.write(frame)
            frame_idx += 1
            continue
    
        # Retrieve bounding box data for the current frame
        bb_entry = df_bb.iloc[frame_idx]
    
        # Extract bounding box coordinates
        x = bb_entry.get("bounding_box_x", None)
        y = bb_entry.get("bounding_box_y", None)
        w = bb_entry.get("bounding_box_w", None)
        h = bb_entry.get("bounding_box_h", None)
    
        # Check if all bounding box coordinates are present and valid
        if pd.notna(x) and pd.notna(y) and pd.notna(w) and pd.notna(h):
            try:
                # Convert coordinates to integers
                x = int(round(x))
                y = int(round(y))
                w = int(round(w))
                h = int(round(h))
    
                # Define the top-left and bottom-right points of the bounding box
                top_left = (x, y)
                bottom_right = (x + w, y + h)
    
                # Draw the bounding box on the frame
                cv2.rectangle(frame, top_left, bottom_right, box_color, box_thickness)
            except Exception as e:
                print(f"Warning: Failed to draw bounding box on frame {frame_idx + 1}. Exception: {e}")
    
        # Optional: Add frame number annotation on the video
        # Uncomment the following lines if frame numbers are desired
        # cv2.putText(
        #     frame, 
        #     f"Frame: {frame_idx + 1}", 
        #     (10, 30), 
        #     cv2.FONT_HERSHEY_SIMPLEX, 
        #     1, 
        #     (0, 0, 255), 
        #     2, 
        #     cv2.LINE_AA
        # )
    
        # Write the annotated frame to the output video
        out.write(frame)
        frame_idx += 1
    
        # Progress update every 100 frames
        if frame_idx % 100 == 0:
            print(f"Processed {frame_idx} / {total_frames} frames...")
    
    # ---------------------------------------------------
    # 6) Release Resources
    # ---------------------------------------------------
    cap.release()  # Close the video capture
    out.release()  # Close the video writer
    print("\nProcessing complete. Output video saved successfully.")

def main():
    """
    Example usage:
        python draw_bounding_boxes.py <video_path> <bounding_box_csv_path> <output_video_path>

    Parameters:
        <video_path>             : Path to video in The Face of Parkinsons Disease dataset. 
        <bounding_box_csv_path>  : Path to the CSV file containing bounding box data.
        <output_video_path>      : Path to where you would like to save the output video.
    """
    # ---------------------------------------------------
    # 1) Parse Command-Line Arguments
    # ---------------------------------------------------
    if len(sys.argv) != 4:
        print("Usage: python script_draw_bboxes.py <video_path> <bounding_box_csv_path> <output_video_path>")
        sys.exit(1)
    
    video_path = sys.argv[1]
    bounding_box_csv_path = sys.argv[2]
    output_video_path = sys.argv[3]
    
    # ---------------------------------------------------
    # 2) Invoke the Bounding Box Drawing Function
    # ---------------------------------------------------
    draw_bounding_boxes_on_video(
        video_path=video_path, 
        bounding_box_csv_path=bounding_box_csv_path, 
        output_video_path=output_video_path
    )

if __name__ == "__main__":
    main()
