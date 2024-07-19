from ultralytics import YOLO
import cv2
import torch
import math
import cvzone
from paddleocr import PaddleOCR
import os.path

def cleanup_license_plate_numbers(filename):
    with open(filename, "r") as file:
        lines = file.readlines()
    with open(filename, "w") as file:
        seen = set()
        for line in lines:
            plate_number = line.strip()
            if plate_number not in seen and len(plate_number) == 10:
                file.write(plate_number + "\n")
                seen.add(plate_number)

def save_confidences_to_file(confidences, filename):
    with open(filename, "a") as file:
        for conf in confidences:
            file.write(f"{conf}\n")

# Initialize video capture
cap = cv2.VideoCapture("accurate.mp4")  # Update with your video file path

# Initialize YOLO model
model = YOLO("/Users/kokkallahithasree/Downloads/Real-Time-Detection-of-Helmet-Violations-and-Capturing-Bike-Numbers-from-Number-Plates-main/runs/detect/train9/weights/best.pt")  # Update with the correct path to your trained YOLO model

# Device configuration
device = torch.device("cpu")  # Change to "cuda" for GPU inference if available

# Define class names
classNames = ["with helmet", "without helmet", "rider", "number plate"]

# Output files
output_file = "license_plate_numbers.txt"  # Output file to write license plate numbers
output_video_file = "output.mp4"  # Output video file

# Initialize OCR model
ocr = PaddleOCR(use_angle_cls=True, lang='en')  # Load OCR model

# Check if the output file exists, create it if not
if not os.path.isfile(output_file):
    open(output_file, 'a').close()

# Create files to store confidences
confidence_files = {class_name: f"{class_name.replace(' ', '_')}_confidences.txt" for class_name in classNames}
for file in confidence_files.values():
    open(file, 'w').close()  # Clear the content of the confidence files before processing

# File to store extraction confidences
extraction_confidences_file = "extraction_confidences.txt"
open(extraction_confidences_file, 'w').close()  # Clear the content of extraction confidences file before processing

# Initialize dictionaries to track the highest confidence for each class and number plate
highest_confidences = {class_name: 0 for class_name in classNames}
highest_confidence_plate = {"text": "", "confidence": 0}  # Initialize highest confidence number plate

# Grab the width, height, and fps of the frames in the video stream.
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))

# Define video writer
fourcc = cv2.VideoWriter_fourcc(*'XVID')
output = cv2.VideoWriter(output_video_file, fourcc, fps, (frame_width, frame_height))

# Process video frames
while True:
    success, img = cap.read()
    if not success:
        break

    new_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = model(new_img, stream=True, device=device)

    for r in results:
        boxes = r.boxes
        li = dict()
        rider_box = []
        detected_helmet = False  # Flag to check if a helmet is detected
        detected_plate = False
        xy = boxes.xyxy
        confidences = boxes.conf
        classes = boxes.cls
        new_boxes = torch.cat((xy.to(device), confidences.unsqueeze(1).to(device), classes.unsqueeze(1).to(device)), 1)

        try:
            new_boxes = new_boxes[new_boxes[:, -1].sort()[1]]
            indices = torch.where(new_boxes[:, -1] == 2)
            rows = new_boxes[indices]

            for box in rows:
                x1, y1, x2, y2 = box[0], box[1], box[2], box[3]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                rider_box.append((x1, y1, x2, y2))
        except Exception as e:
            print(f"Error processing boxes: {e}")

        for i, box in enumerate(new_boxes):
            x1, y1, x2, y2 = box[0], box[1], box[2], box[3]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            w, h = x2 - x1, y2 - y1
            conf = math.ceil((box[4] * 100)) / 100
            cls = int(box[5])

            if classNames[cls] in ["with helmet", "without helmet", "rider", "number plate"] and conf >= 0.5:
                save_confidences_to_file([conf], confidence_files[classNames[cls]])

                # Track the highest confidence for each class
                if conf > highest_confidences[classNames[cls]]:
                    highest_confidences[classNames[cls]] = conf

                if classNames[cls] == "rider":
                    rider_box.append((x1, y1, x2, y2))
                if classNames[cls] == "with helmet":
                    detected_helmet = True  # Update flag if a helmet is detected

                if rider_box and not detected_helmet:  # Check if rider detected and no helmet detected
                    for j, rider in enumerate(rider_box):
                        if x1 + 10 >= rider_box[j][0] and y1 + 10 >= rider_box[j][1] and x2 <= rider_box[j][2] and \
                                y2 <= rider_box[j][3] and not detected_plate:
                            # Expand bounding boxes
                            expand_ratio = 0.05  # Expand by 5%
                            ex1 = int(x1 - w * expand_ratio)
                            ey1 = int(y1 - h * expand_ratio)
                            ex2 = int(x2 + w * expand_ratio)
                            ey2 = int(y2 + h * expand_ratio)

                            # Ensure the expanded box is within image bounds
                            ex1 = max(ex1, 0)
                            ey1 = max(ey1, 0)
                            ex2 = min(ex2, frame_width)
                            ey2 = min(ey2, frame_height)

                            ew, eh = ex2 - ex1, ey2 - ey1

                            # Thinner bounding boxes and text rectangles
                            cvzone.cornerRect(img, (ex1, ey1, ew, eh), l=7, rt=1, t=1, colorR=(255, 0, 0))
                            cvzone.putTextRect(img, f"{classNames[cls].upper()} {conf:.2f}", (ex1 + 10, ey1 - 10), scale=1.5,
                                               offset=10, thickness=1, colorT=(39, 40, 41), colorR=(248, 222, 34))
                            li.setdefault(f"rider{j}", [])
                            li[f"rider{j}"].append(classNames[cls])
                            if classNames[cls] == "number plate":
                                detected_plate = True  # Set flag to indicate number plate detection
                                crop = img[ey1:ey2, ex1:ex2]
                                # Display the cropped number plate for debugging
                                cv2.imshow("Cropped Number Plate", crop)
                                cv2.waitKey(1)
                                try:
                                    # OCR to extract number plate
                                    ocr_result = ocr.ocr(crop, cls=True)
                                    combined_text = ""
                                    extraction_confidences = []
                                    for line in ocr_result:
                                        for word in line:
                                            text = word[1][0]
                                            combined_text += text
                                            extraction_confidences.append(word[1][1])

                                            if len(combined_text) >= 10:
                                                break

                                        if len(combined_text) >= 10:
                                            break

                                    if len(combined_text) == 10 and combined_text[:2].isalpha() and combined_text[2:4].isdigit() and combined_text[4:6].isalpha() and combined_text[6:].isdigit():
                                        # Calculate average confidence for the detected number plate
                                        avg_confidence = sum(extraction_confidences) / len(extraction_confidences)
                                        print(f"Extracted Number Plate: {combined_text} with confidence {avg_confidence:.2f}")

                                        # Track the highest confidence number plate
                                        if avg_confidence > highest_confidence_plate["confidence"]:
                                            highest_confidence_plate = {"text": combined_text, "confidence": avg_confidence}

                                        try:
                                            with open("license_plate_numbers.txt", "a") as file:
                                                file.write(f"{combined_text}\n")
                                            save_confidences_to_file(extraction_confidences, extraction_confidences_file)
                                        except Exception as e:
                                            print(f"Error writing to file: {e}")
                                        cvzone.putTextRect(img, combined_text, (ex1, ey1 - 50), scale=1.5, offset=10,
                                                           thickness=1, colorT=(39, 40, 41), colorR=(105, 255, 255))
                                    else:
                                        print("No valid number plate detected.")
                                except Exception as e:
                                    print(f"Error processing number plate: {e}")

    output.write(img)
    cv2.imshow('Video', img)
    li = list()
    rider_box = list()

    cleanup_license_plate_numbers("license_plate_numbers.txt")

    if cv2.waitKey(1) & 0xFF == ord('q'):
        output.release()
        break

cap.release()
output.release()
cv2.destroyAllWindows()

# Save the highest confidence number plate to a separate file
if highest_confidence_plate["text"]:
    with open("highest_confidence_plate.txt", "w") as file:
        file.write(f"{highest_confidence_plate['text']}\n")

# Print the highest extraction confidence
print(f"Highest extraction confidence: {highest_confidence_plate['confidence']:.2f}")

# Calculate average of all highest confidences if they are > 0.8
filtered_confidences = [conf for conf in highest_confidences.values() if conf > 0.8]
if filtered_confidences:
    average_highest_confidence = sum(filtered_confidences) / len(filtered_confidences)
    print(f"Overall Accuracy: {average_highest_confidence:.2f}")
else:
    print("No confidences found.")
