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
```


### Step 2: Train the Model

After capturing face images, run the `s2_tranning.py` script to encode the captured face images and prepare them for recognition.

```bash
python s2_tranning.py
```
### Step 3: Recognize Faces
To start recognizing faces in real-time, use the `s3_face_recog.py` script. This script will access your camera and identify faces as they appear in front of the camera.
```bash
python s3_face_recog.py
```
The recognized faces will be displayed on the screen along with their names. If the face is not recognized, it will display "Unknown".

### Step 4: Add New Person
If you want to add a new person to the system, run the s4_add_new_person.py script. This will allow you to capture new images and add them to the dataset.
```bash
python s4_add_new_person.py
```
Follow the same process of capturing images, and the person will be added to the system.

## Contributing
Feel free to submit issues or pull requests to improve this project.

