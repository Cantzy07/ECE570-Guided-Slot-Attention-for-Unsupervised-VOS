# Guided Slot Attention for Unsupervised Video

This repository provides an implementation of a **Guided Slot Attention** model for unsupervised video object segmentation. It leverages the [DAVIS](https://davischallenge.org/) dataset and demonstrates how slot attention mechanisms can be guided to discover meaningful object segments in video data.

## Table of Contents
1. [Features](#features)  
2. [Project Structure](#project-structure)  
3. [Installation and Dependencies](#installation-and-dependencies)    
4. [Model Explanation](#model-explanation)  
5. [Acknowledgments](#acknowledgments)  

---

## Features
- **Guided Slot Attention**: Enhanced slot attention mechanism tailored for unsupervised video object segmentation.  
- **Modular Design**: Easily extensible code structure for custom datasets or different video segmentation tasks.  
- **DAVIS Integration**: Out-of-the-box scripts to fetch and preprocess the [DAVIS 2016/2017](https://davischallenge.org/) dataset.  
- **Demo Scripts**: Ready-to-use examples for testing on short video clips.  

---

## Project Structure
- **`Datasets/DAVIS/`**: Contains the DAVIS dataset files after download/preprocessing.  
- **`models/`**: Houses the neural network architectures for the guided slot attention model. Contains the feature aggregation transformer, slot attention, and main gsa model. 
- **`module_tests/`**: Scripts for validating individual parts of the architecture.
- **`download_datasets.py`**: Facilitates downloading the DAVIS dataset or preparing it for use.  
- **`gsa_demo.py`**: Demonstrates quick usage of the trained model on a single image set example.   
- **`load_DAVIS16.py`**: Custom made dataloader to load in a target image, reference images (5), and ground-truth annotation mask for each training iteration. 
- **`train.py`**: Training with custom or advanced options with the DAVIS16 training image set.
- **`test.py`**: Test model accuracy with DAVIS16 validation image set.

---

## Installation and Dependencies

1. **Clone the repository**:
   ```bash
   git clone https://github.com/yourusername/guided-slot-attention.git
   cd guided-slot-attention

2. **Setup a Python Environment**:
    python -m venv venv
    source venv/bin/activate  # On Linux/Mac
    or
    venv\Scripts\activate     # On Windows

3. **Install Dependencies**:
    pip install -r requirements.txt

4. **Download/Prepare DAVIS16**:
    python download_datasets.py

5. **To See an Example Output**:
    python gsa_demo.py

## Model Explanation
- **This module utilizes Guided Slot Attention in order to create a segmentation mask for the most prominent object in a scene or video sequence**
- **It uses the MiT-b2 encoder which is maintained by NVIDIA to encode local and global features as well as two slots to represent the foreground and background by taking input in the form of a target image and reference images**
- **The local and global features pass through a feature aggregation transformer (FAT) and self-attention/cross-attention is performed on the features to create an aggregated feature tensor that preserves the most important features from the local and global encoding**
- **The aggregated features and slots are taken as input to a slot attention module which outputs refined slots by using KNN filtering to select the most applicable features to each individual slot then using the slots as queries in an attention style transformer with the chosen aggregated features**
- **The decoder uses cosine-similarity with a local encoding of the target image and the aggregated features/foreground slot/background slot which are then concatenated into an output mask. This mask goes through two sequential block stages of refinement, one at its original shape and one at its upsampled shape that matches the gt mask dimensions**

## Acknowledgements
- **This re-implementation is inspired from the Guided Slot Attention for Unsupervised Video Object Segmentation paper by Minhyeok Lee, Suhwan Cho, Dogyoon Lee, Chaewon Park, Jungho Lee, and Sangyoun Lee from Yonsei University**
- **{hydragon516,chosuhwan,nemotio,chaewon28,2015142131,syleee}@yonsei.ac.kr**
- **The archived paper can be found here: https://arxiv.org/pdf/2303.08314**
- **The original code can be found here: https://github.com/Hydragon516/GSANet**