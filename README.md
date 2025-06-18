# Multimodal Video Classification Model
This repository contains a modular pipeline for processing and classifying videos based on visual, textual, and audio content. It was designed to identify affiliations of specific organizations in TikTok videos using a combination of custom-trained auxiliary models and external pre-trained models.

## Project Structure
audio_utils/ - 
Utilities for processing and analyzing audio tracks extracted from videos. Includes the affiliated audio IDs.

extract_frames/ -
Scripts for extracting representative video frames. The script that was used for the vectorization process is `extract_frames_and_filter.py`

final_models/ - 
Contains the final classification models that combine multimodal features into a unified prediction. These models use aggregated features from text, symbols, audio, and leader detection modules.

* data/ - Contains .npz files of the video vectors from the different vectorization methods.
* models/ - Contains final models in a .joblib format.
* results/ -
    * predictions/ - Final model results on evaluation set.
    * train_baseline/ - Classifiers results on the video vectors extracted by different methods.
    * train_tuned/ - Classifiers results on the final selected model after hyperparamters fine tuning.
* utils/ - Contains utils for running the classifiers.
* baseline_models.py: Defines the classifier models used for final prediction (Logistic Regression, XGBoost, Balanced Random Forest).
* run_baselines.py: Executes the training and evaluation pipeline using the models defined in baseline_models.py. 

leaders_detection/ - 
Face recognition pipeline to identify known figures in video frames. Relies on pre-trained facial embedding models. The model used in the final vectorization process is InsightFace model.

symbols_detection/ - 
Symbol detection using object detection models. These models are trained to identify organizational logos and flags within frames.
* symbols_detection_models/ - Contains all the models that were trained on the symbols datasets. The model used in the final vectorization process is yolo11s_v3_updated1, which is a YOLOv11 image detection model.
* Contains scripts for augmenting the dataset, and stratify splitting it to train and test groups.

text_detection/ - 
Text recognition from video frames using OCR tools. Contains words banks of affiliated text with the different organizations.

video_vectorizer/ -
Coordinates feature extraction from all auxiliary modules and builds feature vectors for each video. Saves output as .npz for use in final classification.
* notebooks/ - Contains .ipynb notebooks that were used to vectorize the videos on Google Colab framework, each for a different method.
* `create_feature_vectors.ipynb` creates the auxilliary models feature vectors, and `early_fusion_vectors.ipynb` combines the aux features with additional features from the differnt models.

## Getting Started
1. Extract frames:
Navigate to extract_frames/ and run the appropriate script to sample frames from raw video files.

2. Vectorize videos:
Use video_vectorizer/ to combine all features into a unified format.

3. Run classification:
Use the models in final_models/ to classify videos.

