Mask, Age, Gender & Object Detection System
A comprehensive computer vision application that detects faces, predicts masks, estimates age and gender, and performs object detection in real-time using OpenCV and deep learning models.

Features
Face Detection: Accurate face detection using SSD-based model

Mask Detection: Classifies whether a person is wearing a mask

Age & Gender Prediction: Estimates age group and gender

Object Detection: Detects various objects using MobileNet SSD

Real-time Processing: Works with webcam feeds in real-time

Camera Selection: Supports both built-in and USB cameras

User-friendly Interface: Clean display with information board and warnings

Prerequisites
Python 3.6+

OpenCV

NumPy

Pillow (PIL)

Installation
Clone or download this repository

Install required dependencies:

bash
pip install opencv-python numpy pillow
Download the required model files:

Face detection model: face_detector/ directory with:

deploy.prototxt

res10_300x300_ssd_iter_140000.caffemodel

Age and gender models: age_gender/ directory with:

age_deploy.prototxt

age_net.caffemodel

gender_deploy.prototxt

gender_net.caffemodel

Object detection model: object_detector/ directory with:

MobileNetSSD_deploy.prototxt

MobileNetSSD_deploy.caffemodel

(Optional) Download the Poppins font file for better text rendering

Usage
Run the application:

bash
python detection_app.py
Select your camera source from the initial screen:

Option 1: Laptop Camera (index 0)

Option 2: USB Camera (index 1)

Option 3: Test All Cameras to find the correct index

During operation, you can:

Press 's' to switch cameras

Press 'q' to quit the application

Camera Setup
Using Built-in Laptop Camera
The application will automatically detect and use the built-in camera

Using USB Camera
Connect your USB camera to an available USB port

Wait for your operating system to recognize the device

Select Option 2 (USB Camera) from the initial selection screen

If the camera isn't detected, try Option 3 to test all available cameras

Troubleshooting Camera Issues
Ensure no other applications are using the camera

Try different USB ports if the camera isn't detected

Check your operating system's camera privacy settings

For Linux systems, you may need to install additional packages:

bash
sudo apt install v4l-utils
Project Structure
text
project-directory/
│
├── detection_app.py          # Main application file
├── face_detector/            # Face detection models
│   ├── deploy.prototxt
│   └── res10_300x300_ssd_iter_140000.caffemodel
├── age_gender/               # Age and gender prediction models
│   ├── age_deploy.prototxt
│   ├── age_net.caffemodel
│   ├── gender_deploy.prototxt
│   └── gender_net.caffemodel
├── object_detector/          # Object detection models
│   ├── MobileNetSSD_deploy.prototxt
│   └── MobileNetSSD_deploy.caffemodel
└── README.md                 # This file
Model Information
Face Detection: Caffe-based SSD model with ResNet-10 backbone

Age & Gender Prediction: Custom Caffe models trained on specific datasets

Object Detection: MobileNet-SSD model trained on PASCAL VOC dataset

Customization
You can customize various aspects of the application:

Adjust confidence thresholds in the code

Modify the age buckets or gender classes

Change the display colors and information layout

Add additional object classes to the detection system

Performance Notes
The application is optimized for real-time performance

Processing speed may vary based on your hardware

For better performance with USB cameras, consider reducing the resolution

The application includes error handling for camera disconnections

Limitations
Mask detection uses color-based heuristics and may not be highly accurate

Age and gender predictions are estimates based on pre-trained models

Performance depends on lighting conditions and camera quality

License
This project is for educational and research purposes. Please check the licensing terms of the pre-trained models before commercial use.

Support
For issues related to:

Camera connectivity: Check your operating system's camera settings

Model loading: Ensure all model files are in the correct directories

Performance: Adjust resolution or processing parameters in the code
