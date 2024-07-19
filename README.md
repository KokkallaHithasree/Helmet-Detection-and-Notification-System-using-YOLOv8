# Helmet-Detection-and-Notification-System-using-YOLOv8
## Introduction

This project focuses on enhancing road safety by detecting helmet violations among motorcycle riders using image processing and machine learning techniques. The system uses video input to identify riders without helmets and sends notifications to the registered mobile numbers of the violators.
## Objectives

- Improve helmet detection accuracy during low-light conditions using YOLO.
- Enhance number plate extraction at various angles using data augmentation in PaddleOCR.
- Develop a multilingual notification system using Twilio for effective communication of safety alerts.
## Methodology

The system uses advanced computer vision techniques and OCR to detect helmets and license plates in real-time. The YOLO object detection model processes video frames to identify objects such as individuals wearing helmets, riders, and vehicle number plates. OCR extracts alphanumeric text from identified number plates. Non-compliant riders receive real-time notifications through the Twilio API.
## Key Components

- YOLO (You Only Look Once): Used for real-time object detection.
- OCR (Optical Character Recognition): Extracts text from number plates.
- Twilio API: Sends notifications to vehicle owners in their preferred language.
## System Workflow

1. Capture video input.
2. Use YOLO to detect helmet and rider.
3. Extract number plate using OCR.
4. Send notification to the rider if not wearing a helmet.
## Files

1. **training.py :** Script to train the YOLO model.
2. **main.py :** Main script to run the helmet detection system.
3. **Notification1.py :** Script to handle notifications using Twilio.
Installation and Usage Instructions

## Dataset Used
The dataset collected from kaggle contains 120 images including classes with Helmet,without Helmet,Number Plate,Rider.\
[Dataset](https://www.kaggle.com/datasets/aneesarom/rider-with-helmet-without-helmet-number-plate/data)
## Clone the repository:

```git clone <https://github.com/KokkallaHithasree/Helmet-Detection-and-Notification-System-using-YOLOv8.git>```\
```cd <Helmet-Detection-and-Notification-System-using-YOLOv8>```
## Install the required dependencies:

```pip install -r requirements.txt```

## Train the YOLO model:
Run the training.py script to train the YOLO model with your dataset.

```python training.py```
## Run the helmet detection system:
Use the main.py script to start the helmet detection and number plate extraction process.

```python main.py```

## Set up and send notifications:
Configure the Twilio API credentials in the Notification1.py script and run it to send notifications to non-compliant riders.

```python Notification1.py```

## Results

- The system was tested under various lighting conditions and showed effective detection capabilities.
- Notifications were successfully sent in user preferred language.
## Demo Output Video

## Future Enhancements
- Refining the model with robust dataset for more accurate results.
- Additional refinement to improve extraction of number plate.

