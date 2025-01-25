# Analyzing Progression of Parkinson's Disease via a Longitudinal Analysis of Celebrity Hypomimia Symptoms

## PROJECT IN PROGRESS

## Overview 📈

This study uses the *ParkCeleb* dataset, a novel dataset containing annotated YouTube videos of 40 celebrities before and after their diagnosis and 40 control subjects. The dataset spans ten years before to twenty years after diagnosis, providing a comprehensive view of evolving speech signs associated to PD. This allows for robust longitudinal analysis of the development of Parkinson's Disease (PD).

This specific study adapts *ParkCeleb* into *The Face of Parkinsons* database. Since ParkCeleb was compiled for speech analysis, the annotated segments often do not display the celebrity's face let alone at the proper angle. Thus, to create this dataset, Deepface and Mediapipe were used to isolate the celebrity's face and calculate the face angle and size, to pull the useful segments. This newly annotated videos are optimal for hypomimia analaysis.

### The Face of Parkinson's

To acquire the **The Face of Parkinson's** data set, the **ParkCeleb** data set is needed. **ParkCeleb** is stored in the following [Zenodo repository](https://zenodo.org/uploads/13954768). This repository does not contain the actual audio recordings but provides metadata files with links to YouTube videos, speaker information, and relevant timestamps.

## 1. Installation️ 💻 

To set up the project locally, follow these steps:

1. **Clone the Repository:**

   ```bash
   git clone https://github.com/Nishant-Nagururu/The-Face-of-Parkinsons.git
   cd The-Face-of-Parkinsons
   ```

2. **Install Dependencies** 

   1. Create a virtual environment and install the required packages:

   ```bash
   pip install -r requirements.txt
   ```

   Optionally you could use the environment.yml to create a conda environment like so:

   ```bash
   conda env create -f environment.yml
   conda activate deepface
   ```

   Note that this environment.yml file was exported from our Linux HPC so it will not work on all machines. For most users we recommend the first approach.

   2. Also download the face_landmarker_v2_with_blendshapes.task file from the [Mediapipe Docs](https://ai.google.dev/edge/mediapipe/solutions/vision/face_landmarker) and place it in the directory.

## 2. Get The Face of Parkinson's Videos

   1. After requesting and downloading the Zenodo repository, move it to the top level of this repository. You can download the video files by running the script below:

   ```bash
   python src/download_videos.py path_to_zenodo_directory
   ```

   2. (Optional) All of the rest of the process of segmenting the ParkCeleb videos into clips viable for hypomimia analysis has been done. However, if you would like to recreate it, run the following scripts:

   ```bash
   python createSegments/saveClips.py path_to_zenodo_directory
   python createSegments/segmentClips.py path_to_zenodo_directory path_to_face_landmarker_v2_with_blendshapes.task
   ```
   3. 

## Important Note on Data Availability and Disk Space Requirements  

### Data Availability  
Some data sources referenced in the metadata may no longer be accessible due to removal, restriction, or geo-blocking. The script will download all the videos still available.

## Citing The Face of Parkinson's 📖
If you use **The Face of Parkinson's** in your research, please cite the following publication:

```bibtex
@article{nagururu2024face,
  title={},
  author={},
  journal={},
  volume={},
  number={},
  pages={},
  year={},
  publisher={}
}
```
Also cite the **ParkCeleb** with the following:

```bibtex
@article{favaro2024unveiling,
  title={Unveiling early signs of Parkinson’s disease via a longitudinal analysis of celebrity speech recordings},
  author={Favaro, Anna and Butala, Ankur and Thebaud, Thomas and Villalba, Jes{\'u}s and Dehak, Najim and Moro-Vel{\'a}zquez, Laureano},
  journal={npj Parkinson's Disease},
  volume={10},
  number={1},
  pages={207},
  year={2024},
  publisher={Nature Publishing Group UK London}
}
```

## Acknowledgments 🛜

- **Data Sources:** The speech data used in this study was sourced from publicly available YouTube videos of celebrities with and without PD. Those with PD voluntarily disclosed their diagnosis in public.

## Contact 📱

For questions or further information, please contact [Nishant Nagururu](mailto:nishant.nagururu@ufl.edu).
