# Analyzing Progression of Parkinson's Disease via Longitudinal Analysis of Celebrity Hypomimia Symptoms

## Overview 

This study leverages the **ParkCeleb** dataset, a novel longitudinal collection of annotated YouTube videos featuring 40 celebrities diagnosed with Parkinson's Disease (PD) and 60 control subjects. Spanning forty years before to thirty years after diagnosis, the dataset offers a comprehensive view of evolving speech and facial symptoms associated with PD, enabling robust longitudinal analysis of the disease's progression.

To focus specifically on hypomimia (reduced facial expressiveness), we have adapted **ParkCeleb** into the **The Face of Parkinson's Disease** database. Since **ParkCeleb** was initially compiled for speech analysis, many annotated segments lack clear facial views or proper angles. To address this, we utilized **DeepFace** and **Mediapipe** to isolate celebrities' faces, calculate face angles and sizes, and extract relevant segments. The resulting dataset is optimized for hypomimia analysis.

**ParkCeleb** can be requested and accessed via the [Zenodo repository](https://zenodo.org/uploads/13954768). Note that this repository does not include actual audio recordings but provides metadata files with links to YouTube videos, speaker information, and relevant timestamps.

### The Face of Parkinson's Disease

To obtain the **The Face of Parkinson's Disease** dataset, download it here (coming soon). This dataset includes all video segments along with metadata detailing the locations of 478 facial landmarks in each detectable frame and the bounding box coordinates surrounding the target subject's face.

## The Face of Parkinson's Disease Pipeline 

### 1. Clone the Repository

```bash
git clone https://github.com/Nishant-Nagururu/The-Face-Of-Parkinsons-Disease.git
cd The-Face-Of-Parkinsons-Disease
```

### 2. Install Dependencies

#### a. Create a Virtual Environment and Install Required Packages

```bash
pip install -r requirements.txt
```

Alternatively, you can use the `environment.yml` to create a Conda environment:

```bash
conda env create -f environment.yml
conda activate deepface
```

*Note:* The `environment.yml` file was exported from our Linux HPC and may not work on all machines. We recommend using the `pip` approach for most users.

#### b. Download Mediapipe Face Landmarker Model

Download the `face_landmarker_v2_with_blendshapes.task` file from the [Mediapipe Documentation](https://ai.google.dev/edge/mediapipe/solutions/vision/face_landmarker) and place it in the project directory.

### 3. Obtain The Face of Parkinson's Videos

#### a. Download ParkCeleb Repository

After requesting and downloading the ParkCeleb metadata from Zenodo, move it to the top level of this repository. Then, download the video files by running:

```bash
python download/download_videos.py <base_folder_path>
```
**Parameters:**
- `<base_folder_path>`: Path to the ParkCeleb dataset's base folder containing `PD` and `CN` subfolders.

#### b. Extract Speaking Segments

Extract segments where the target subject is speaking using:

```bash
python saveClips.py <base_folder_path>
```

**Parameters:**
- `<base_folder_path>`: Path to the ParkCeleb dataset's base folder containing `PD` and `CN` subfolders.

#### c. Segment Clips Based on Face Visibility

Segment the clips into contiguous frames where the subject is present and facing the camera. Run the script twice—once for control (CN) videos and once for PD videos:

```bash
python segmentClips.py <base_folder_path> <model_path> <prefix> <start_num> <end_num> [num_workers]
```

**Parameters:**
- `<base_folder_path>`: Path to the ParkCeleb dataset's base folder containing `PD`/`CN` subfolders.
- `<model_path>`: Path to the Mediapipe Face Landmarker `.task` model.
- `<prefix>`: Prefix for subfolder naming (e.g., `PD`, `CN`).
- `<start_num>`: Starting index of subfolders to process.
- `<end_num>`: Ending index of subfolders to process.
- `[num_workers]`: (Optional) Number of parallel worker processes. Defaults to CPU count.

#### d. Compile Video Segments

Compile the video segments into the **The Face of Parkinson's Disease** datasets by running the script twice—once for PD and once for CN videos:

```bash
python compile.py <base_folder_path> <output_path> <subfolder_prefix>
```

**Parameters:**
- `<base_folder_path>`: Path to the `PD` or `CN` folder in the ParkCeleb dataset.
- `<output_path>`: Destination directory for the compiled videos and metadata (e.g., `The_Face_Of_Parkinsons` or `The_Face_Of_Parkinsons_CN`).
- `<subfolder_prefix>`: Prefix of the subfolders to process (e.g., `pd_`, `cn_`).

#### e. Manual Data Review

Manually review the data to ensure high-quality segments. Remove videos for reasons such as:
- Subjects wearing sunglasses
- Obstructions covering the face
- The algorithm locking onto a still image
- Misidentification due to large groups of people

#### f. Generate Data Statistics

Run the following scripts to gather statistics on the dataset:

- **Count Videos:**

  ```bash
  python count.py <folder_path>
  ```

  **Parameters:**
  - `<folder_path>`: Path to the `The_Face_Of_Parkinsons` or `The_Face_Of_Parkinsons_CN` folders.

- **Create Histograms:**

  ```bash
  python graph.py <folder_path>
  ```

  **Parameters:**
  - `<folder_path>`: Path to the `The_Face_Of_Parkinsons` or `The_Face_Of_Parkinsons_CN` folders.

*Note:* These scripts generate histograms for each subject's video distribution over the years from diagnosis and overall patient distribution. They also differentiate between all videos and those longer than 5 seconds. To adjust thresholds (e.g., 3 or 4 seconds), modify the scripts accordingly.

#### g. Extract Facial Features

Extract features such as blinks per second and mouth movement using:

   ```bash
   python getFeatures.py <data_path> <model_path> <lower_bound> <upper_bound> <num_workers>
   ```

   **Parameters:**
   - `<data_path>`: Directory containing `The_Face_Of_Parkinsons` or `The_Face_Of_Parkinsons_CN` folders.
   - `<model_path>`: Path to the Mediapipe FaceLandmarker with Blendshapes `.task` file.
   - `<lower_bound>`: Lower index bound of subfolders to process.
   - `<upper_bound>`: Upper index bound of subfolders to process.
   - `<num_workers>`: (Optional) Number of parallel worker processes. Defaults to CPU cores.

*Note:* Currently, the script extracts blinks per second and mouth movement. We recommend expanding it to include additional features as needed. Also, this script saves the location of all 478 face mesh landmarks and the location of the bounding box for the subject's face for each frame of each video. This data is in corresponding csv within the bounding_boxes folder of each of the patient folders.

#### h. Combine Feature Files

If you would like, you can combine all of the features.csv outputted by "getFeatures.csv" into one large csv of the features for all the videos. It also combines the bounding_boxes folders into one large folder that stores all the metadata for all the videos.

   ```bash
   python combineFeatures.py <base_folder_path>
   ```

   **Parameters:**
   - `<base_folder_path>`: Path to The_Face_Of_Parkinsons datset or The_Face_Of_Parkinsons_CN dataset. 


## Miscellaneous Scripts

### 1. Draw Bounding Box on Video

If you want to see a video with the bounding box drawn on you can use the following script:

   ```bash
   python draw_bounding_boxes.py <video_path> <bounding_box_csv_path> <output_video_path>
   ```

   **Parameters:**
   - `<video_path>`: Path to video in The Face of Parkinsons Disease dataset. 
   - `<bounding_box_csv_path>`: Path to the CSV file containing bounding box data.
   - `<output_video_path>`: Path to where you would like to save the output video.

## Important Note on Disk Space Requirements 💾

The complete dataset, including all video segments, target face images, and metadata, will occupy approximately **100 GB** when it is released. We recommend using a high-performance computing (HPC) system or a robust PC for handling this dataset efficiently.

## Citing The Face of Parkinson's 📖

If you utilize **The Face of Parkinson's Disease** in your research, please cite the following publication:

```bibtex
@article{nagururu2024face,
  title={The Face of Parkinson’s Disease: A Longitudinal Dataset Capturing the Impact of Parkinson’s on Facial Expressivity},
  author={Nagururu, Nishant and Wong, Alvin and Liu, Shuyu and Guarin, Diego},
  year={2025},
}
```

Additionally, cite **ParkCeleb** as follows:

```bibtex
@article{favaro2024unveiling,
  title={Unveiling Early Signs of Parkinson’s Disease via a Longitudinal Analysis of Celebrity Speech Recordings},
  author={Favaro, Anna and Butala, Ankur and Thebaud, Thomas and Villalba, Jes{\'u}s and Dehak, Najim and Moro-Vel{\'a}zquez, Laureano},
  journal={npj Parkinson's Disease},
  volume={10},
  number={1},
  pages={207},
  year={2024},
  publisher={Nature Publishing Group UK London}
}
```

## Acknowledgments

- **Data Sources:** The data used in this study comprises publicly available YouTube videos of celebrities with and without PD. Individuals diagnosed with PD have voluntarily disclosed their diagnosis publicly.

## Contact 📱

For questions or further information, please contact:

**Nishant Nagururu**  
Email: [nishant.nagururu@ufl.edu](mailto:nishant.nagururu@ufl.edu)

---

*Thank you for your interest in our research! We hope **The Face of Parkinson's Disease** dataset contributes significantly to advancing the understanding and analysis of Parkinson's Disease.*