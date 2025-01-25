import os
import sys
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Allows writing PNG files without a display environment
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import math
from deepface import DeepFace
from scipy.ndimage import median_filter

###########################################################
# 1) Define 3-Year Buckets
###########################################################
def generate_3yr_buckets(start=-42, end=39):
    """
    Generate 3-year buckets from `start` to `end` inclusive,
    with the exception of the year 0, which is placed in its own bin.
    
    Example bins:
      -42 to -40, -39 to -37, ..., -3 to -1, 0, 1 to 3, 4 to 6, ..., 37 to 39
    
    Parameters:
        start (int): Starting year (negative for years before diagnosis).
        end (int): Ending year (positive for years after diagnosis).
    
    Returns:
        list of tuples: Each tuple contains (low, high, label) defining a bucket.
    """
    buckets = []
    # Negative side in steps of 3
    for bin_start in range(start, 0, 3):
        bin_end = bin_start + 2
        if bin_end >= 0:
            break  # Ensure no overlap with the 0 bin
        label = f"{bin_start} to {bin_end}"
        buckets.append((bin_start, bin_end, label))
    
    # The single bin for year 0
    buckets.append((0, 0, "0"))
    
    # Positive side in steps of 3
    for bin_start in range(1, end + 1, 3):
        bin_end = bin_start + 2
        if bin_end > end:
            bin_end = end  # Adjust the last bin to not exceed the end year
        label = f"{bin_start} to {bin_end}"
        buckets.append((bin_start, bin_end, label))
    
    return buckets

# Generate the 3-year buckets
YEAR_BUCKETS = generate_3yr_buckets(-42, 39)

def get_3yr_bucket_label(year_val, year_buckets=YEAR_BUCKETS):
    """
    Determine the 3-year bucket label for a given year value.
    
    Parameters:
        year_val (int or float): The year value relative to diagnosis.
        year_buckets (list of tuples): The predefined 3-year buckets.
    
    Returns:
        str or None: The label of the bucket that `year_val` falls into, or None if out of range.
    """
    for low, high, label in year_buckets:
        if low <= year_val <= high:
            return label
    return None  # If out of all defined ranges

###########################################################
# 2) Helpers to Create Histograms
###########################################################
def create_and_save_histograms(df, folder_path, year_buckets=YEAR_BUCKETS):
    """
    Create and save two histograms based on video data:
        1) Histogram for all videos.
        2) Histogram for videos with duration >= 5 seconds.
    
    The histograms are saved as PNG files in the specified folder.
    
    Parameters:
        df (pd.DataFrame): DataFrame containing at least 'years_val' and 'duration_seconds' columns.
        folder_path (str): Directory where the histogram images will be saved.
        year_buckets (list of tuples): The predefined 3-year buckets.
    
    Notes:
        - Overwrites existing files with the same names.
    """
    # Ensure required columns are present
    if not {"years_val", "duration_seconds"}.issubset(df.columns):
        print(f"Warning: DataFrame is missing required columns. Skipping histogram creation for {folder_path}.")
        return
    
    # Create a new column for bucket labels based on 'years_val'
    df["bucket_label"] = df["years_val"].apply(lambda y: get_3yr_bucket_label(y, year_buckets))
    
    # ---------------------------
    # 1) Histogram of ALL videos
    # ---------------------------
    bucket_counts_all = df["bucket_label"].value_counts()
    ordered_labels = [b[2] for b in year_buckets]  # Ensure consistent order
    bucket_counts_all = bucket_counts_all.reindex(ordered_labels, fill_value=0)  # Fill missing buckets with 0
    
    plt.figure(figsize=(8, 6))
    bucket_counts_all.plot(kind="bar", color="skyblue", edgecolor="black")
    plt.title(f"Video Distribution For {os.path.basename(folder_path)}")
    plt.xlabel("3-Year Buckets (relative to diagnosis)")
    plt.ylabel("Number of Videos")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    hist_path_all = os.path.join(folder_path, "hist_all_videos.png")
    plt.savefig(hist_path_all, dpi=200)
    plt.close()
    
    # --------------------------------
    # 2) Histogram of videos >=5 secs
    # --------------------------------
    df_5sec = df[df["duration_seconds"] >= 5].copy()
    bucket_counts_5sec = df_5sec["bucket_label"].value_counts()
    bucket_counts_5sec = bucket_counts_5sec.reindex(ordered_labels, fill_value=0)  # Ensure consistent order
    
    plt.figure(figsize=(8, 6))
    bucket_counts_5sec.plot(kind="bar", color="lightgreen", edgecolor="black")
    plt.title(f"Video Distribution (≥5s) For {os.path.basename(folder_path)}")
    plt.xlabel("3-Year Buckets (relative to diagnosis)")
    plt.ylabel("Number of Videos (≥5s)")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    hist_path_5sec = os.path.join(folder_path, "hist_videos_5sec_or_more.png")
    plt.savefig(hist_path_5sec, dpi=200)
    plt.close()

def create_and_save_global_histogram(df_all, output_path, year_buckets=YEAR_BUCKETS):
    """
    Create a single global histogram showing the distribution of all videos from all patients
    across the defined 3-year buckets.
    
    Parameters:
        df_all (pd.DataFrame): Combined DataFrame of all patients' video data.
        output_path (str): File path where the global histogram will be saved.
        year_buckets (list of tuples): The predefined 3-year buckets.
    """
    # Create bucket labels for the combined DataFrame
    df_all["bucket_label"] = df_all["years_val"].apply(lambda y: get_3yr_bucket_label(y, year_buckets))
    
    # Count the number of videos in each bucket
    bucket_counts_global = df_all["bucket_label"].value_counts()
    ordered_labels = [b[2] for b in year_buckets]  # Ensure consistent order
    bucket_counts_global = bucket_counts_global.reindex(ordered_labels, fill_value=0)  # Fill missing buckets with 0

    plt.figure(figsize=(10, 6))
    bucket_counts_global.plot(kind="bar", color="orange", edgecolor="black")
    plt.title("Distribution of ALL Videos Across All Patients")
    plt.xlabel("3-Year Buckets (relative to diagnosis)")
    plt.ylabel("Number of Videos (All Patients)")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()

###########################################################
# 3) Building Patient-Level Data for the Grid (Counts)
###########################################################
def add_counts_for_all_videos(df, patient_name, patient_dict):
    """
    Increment the count of videos in each 3-year bucket for a patient.
    
    Parameters:
        df (pd.DataFrame): DataFrame containing video data for the patient.
        patient_name (str): Identifier for the patient.
        patient_dict (dict): Dictionary mapping patient names to their bucket counts.
    """
    for _, row in df.iterrows():
        years_val = row["years_val"]
        bucket_label = get_3yr_bucket_label(years_val)
        if bucket_label is None:
            continue  # Skip if the year_val is out of defined buckets
        if patient_name not in patient_dict:
            patient_dict[patient_name] = {}
        patient_dict[patient_name][bucket_label] = patient_dict[patient_name].get(bucket_label, 0) + 1

def add_counts_for_5s_videos(df, patient_name, patient_dict_5s):
    """
    Increment the count of videos with duration >= 5 seconds in each 3-year bucket for a patient.
    
    Parameters:
        df (pd.DataFrame): DataFrame containing video data for the patient.
        patient_name (str): Identifier for the patient.
        patient_dict_5s (dict): Dictionary mapping patient names to their bucket counts for videos >=5s.
    """
    df_5 = df[df["duration_seconds"] >= 5]
    for _, row in df_5.iterrows():
        years_val = row["years_val"]
        bucket_label = get_3yr_bucket_label(years_val)
        if bucket_label is None:
            continue  # Skip if the year_val is out of defined buckets
        if patient_name not in patient_dict_5s:
            patient_dict_5s[patient_name] = {}
        patient_dict_5s[patient_name][bucket_label] = patient_dict_5s[patient_name].get(bucket_label, 0) + 1

###########################################################
# 4) Create the Grid for All Patients, Displaying Counts
###########################################################
def create_year_grid_image_counts(
    patient_data,
    output_img_path,
    year_buckets=YEAR_BUCKETS,
    include_totals=False
):
    """
    Create a table-like grid image for all patients showing the number of videos in each 3-year bucket.
    
    Parameters:
        patient_data (dict): Dictionary mapping patient names to their bucket counts.
        output_img_path (str): File path where the grid image will be saved.
        year_buckets (list of tuples): The predefined 3-year buckets.
        include_totals (bool): Whether to include a row summing counts across all patients.
    
    Notes:
        - Uses larger fonts and ensures proper layout.
        - Overwrites any existing file with the same name.
    """
    import math
    from matplotlib.lines import Line2D
    
    # Sort the bucket labels according to year_buckets
    all_possible_buckets = [b[2] for b in year_buckets]  # Extract labels from buckets
    bucket_label_to_index = {lab: i for i, lab in enumerate(all_possible_buckets)}  # Map labels to indices
    
    # Sort patients by name for consistent ordering
    sorted_patients = sorted(patient_data.keys())
    num_patients = len(sorted_patients)
    
    # Compute totals for each bucket if needed
    total_counts = {bucket: 0 for bucket in all_possible_buckets}
    if include_totals:
        for patient in sorted_patients:
            for bucket, count in patient_data[patient].items():
                total_counts[bucket] += count
    
    num_buckets = len(all_possible_buckets)
    total_rows = num_patients + (1 if include_totals else 0)  # Add an extra row for totals if needed
    
    # Define figure size based on the number of buckets and patients
    fig_width = max(14, num_buckets * 1.0)  # Width scales with the number of buckets
    fig_height = max(10, total_rows * 1.0)  # Height scales with the number of patients
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    
    # Define the grid boundaries
    ax.set_xlim(-0.5, num_buckets - 0.5)
    ax.set_ylim(-0.5, total_rows - 0.5)
    
    # Invert y-axis so that the first patient is at the top
    ax.invert_yaxis()
    ax.xaxis.tick_top()
    ax.xaxis.set_label_position("top")
    
    # Show only the top spine for a cleaner look
    ax.spines["top"].set_visible(True)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    
    # Draw horizontal lines for row boundaries
    for i in range(total_rows + 1):
        ax.axhline(y=i - 0.5, color='black', linewidth=0.25)
    # Draw vertical lines for column boundaries
    for col_idx in range(num_buckets + 1):
        line = Line2D(
            xdata=[col_idx - 0.5, col_idx - 0.5],
            ydata=[-0.5, total_rows - 0.5],
            color='black',
            linewidth=0.25
        )
        ax.add_line(line)
    
    # Set axis tick labels
    label_fontsize = 16
    ax.set_xticks(range(num_buckets))
    ax.set_xticklabels(all_possible_buckets, fontsize=label_fontsize, rotation=90)
    ax.set_yticks(range(total_rows))
    ytick_labels = sorted_patients + (["Total"] if include_totals else [])  # Append "Total" if required
    ax.set_yticklabels(ytick_labels, fontsize=label_fontsize)
    
    # Set axis titles with increased font size and padding
    ax.set_xlabel("Years from Diagnosis (3-year bins)", fontsize=18, labelpad=20)
    ax.set_ylabel("Patients", fontsize=18, labelpad=20)
    
    # --------------------
    # Fill text in cells
    # --------------------
    cell_fontsize = 30
    top_margin = 0.4  # Shift text slightly downward within the cell
    
    # 1) Per-patient rows
    for row_idx, patient in enumerate(sorted_patients):
        patient_buckets = patient_data.get(patient, {})
        for bucket_label in all_possible_buckets:
            count_value = patient_buckets.get(bucket_label, 0)
            if count_value != 0:
                col_idx = bucket_label_to_index[bucket_label]
                # Calculate y-coordinate for text placement
                y_text = (row_idx - 0.5) + top_margin
                ax.text(
                    col_idx, y_text,
                    str(count_value),
                    fontsize=cell_fontsize,
                    ha='center',
                    va='top'
                )
    
    # 2) Totals row (if requested)
    if include_totals:
        row_idx = num_patients  # "Total" row is the last row
        for bucket_label in all_possible_buckets:
            count_value = total_counts.get(bucket_label, 0)
            if count_value != 0:
                col_idx = bucket_label_to_index[bucket_label]
                y_text = (row_idx - 0.5) + top_margin
                ax.text(
                    col_idx, y_text,
                    str(count_value),
                    fontsize=cell_fontsize,
                    fontweight='bold',  # Highlight totals
                    ha='center',
                    va='top'
                )
    
    # Adjust layout and save the figure
    plt.tight_layout()
    plt.savefig(output_img_path, dpi=200)
    plt.close()

###########################################################
# 5) Main Entry Point
###########################################################
def main(output_dir):
    """
    Main function to process all patient subfolders, generate histograms,
    and create grid images summarizing video distributions.
    
    This function performs the following steps:
        1. Iterates through each subfolder in the output directory.
        2. For each subfolder, reads the 'metadata.csv' file and generates histograms.
        3. Accumulates counts of videos in each 3-year bucket for all patients.
        4. Creates grid images showing the distribution of videos across buckets.
        5. Generates a global histogram summarizing all videos from all patients.
    
    Parameters:
        output_dir (str): Directory containing patient subfolders with 'metadata.csv' and videos.
    """
    # Dictionaries to hold video counts for all videos and for videos >=5 seconds
    all_videos_dict = {}
    five_sec_dict = {}
    
    # List to accumulate DataFrames for creating a combined histogram
    df_list_for_global = []
    
    # Iterate through each subfolder (e.g., "pd_01", "cn_02") in the output directory
    for folder_name in os.listdir(output_dir):
        folder_path = os.path.join(output_dir, folder_name)
        if not os.path.isdir(folder_path):
            continue  # Skip if not a directory
        
        metadata_csv_path = os.path.join(folder_path, "metadata.csv")
        if not os.path.exists(metadata_csv_path):
            print(f"metadata.csv not found in {folder_path}, skipping.")
            continue  # Skip if 'metadata.csv' does not exist
        
        # Read the metadata CSV into a DataFrame
        df = pd.read_csv(metadata_csv_path)
        if "years_val" not in df.columns or "duration_seconds" not in df.columns:
            print(f"metadata.csv in {folder_path} is missing required columns, skipping.")
            continue  # Skip if necessary columns are missing
        
        # Create histograms for the current patient
        create_and_save_histograms(df, folder_path, YEAR_BUCKETS)
        
        # Add counts to patient dictionaries
        patient_name = folder_name  # e.g., "pd_01"
        add_counts_for_all_videos(df, patient_name, all_videos_dict)
        add_counts_for_5s_videos(df, patient_name, five_sec_dict)
        
        # Add the DataFrame to the global list for combined histogram
        df_list_for_global.append(df)
    
    # Create the grid images showing video distributions across all patients
    grid_all_path = os.path.join(output_dir, "all_videos_distribution.png")
    create_year_grid_image_counts(
        all_videos_dict,
        grid_all_path,
        YEAR_BUCKETS,
        include_totals=True  # Include a total row at the bottom
    )
    
    grid_5s_path = os.path.join(output_dir, "all_videos_distribution_5s.png")
    create_year_grid_image_counts(
        five_sec_dict,
        grid_5s_path,
        YEAR_BUCKETS,
        include_totals=True  # Include a total row for videos >=5 seconds
    )
    
    # Create a global histogram for ALL patients combined
    if df_list_for_global:
        df_all_patients = pd.concat(df_list_for_global, ignore_index=True)
        global_hist_path = os.path.join(output_dir, "hist_all_patients.png")
        create_and_save_global_histogram(df_all_patients, global_hist_path, YEAR_BUCKETS)
        print(f"Global histogram saved at: {global_hist_path}")
    
    # Final summary message
    print(f"\nDone! Grid images saved at:\n  {grid_all_path}\n  {grid_5s_path}\n")

###########################################################
# Entry Point
###########################################################
if __name__ == "__main__":
    """
    Script Execution.
    
    Example usage:
        python analyze_3yr_buckets.py <output_directory>
    
    Where:
        - <output_directory>: Directory containing patient subfolders with 'metadata.csv' and videos.
    
    Notes:
        - The script generates histograms and grid images summarizing the distribution of videos
          across predefined 3-year buckets relative to diagnosis.
        - Outputs are saved within each patient subfolder and the main output directory.
    """
    # Validate the number of command-line arguments
    if len(sys.argv) != 2:
        print("Usage: python analyze_3yr_buckets.py <output_directory>")
        sys.exit(1)
    
    # Parse the output directory from command-line arguments
    output_dir = sys.argv[1]
    
    # Validate that the provided path exists and is a directory
    if not os.path.exists(output_dir):
        print(f"Error: {output_dir} does not exist.")
        sys.exit(1)
    if not os.path.isdir(output_dir):
        print(f"Error: {output_dir} is not a directory.")
        sys.exit(1)
    
    # Execute the main processing function
    main(output_dir)
