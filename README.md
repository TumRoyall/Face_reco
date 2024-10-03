# Face Recognition System

This repository contains a face recognition system that includes scripts for capturing face images, training a face recognition model, recognizing faces in real-time using a camera, and adding new persons to the dataset.

## Table of Contents
- [Overview](#overview)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
    - [Step 1: Capture Face Images](#step-1-capture-face-images)
    - [Step 2: Train the Model](#step-2-train-the-model)
    - [Step 3: Real-time Face Recognition](#step-3-real-time-face-recognition)
    - [Step 4: Add a New Person](#step-4-add-a-new-person)
- [Contributing](#contributing)
- [License](#license)

## Overview

This face recognition system uses `face_recognition` library to detect and recognize faces. It includes:
1. Capturing and saving face images for a new person.
2. Training a face recognition model using the captured dataset.
3. Performing real-time face recognition using a webcam.
4. Adding new persons dynamically to the system.

## Project Structure


## Installation

1. Clone this repository:
    ```bash
    git clone https://github.com/YourUsername/Face_Recognition_System.git
    cd Face_Recognition_System
    ```

2. Install the required libraries:
    ```bash
    pip install -r requirements.txt
    ```
    Ensure you have `opencv-python` and `face_recognition` installed.

3. Create a `dataset/` folder in the root directory to store images of each person.

## Usage

### Step 1: Capture Face Images

Use the `s1_input_face.py` script to capture face images of a person and save them in the `dataset/` directory under a folder named after the person.

```bash
python s1_input_face.py
