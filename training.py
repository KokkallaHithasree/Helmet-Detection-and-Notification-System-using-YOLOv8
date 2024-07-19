from ultralytics import YOLO

# yolo model creation
model = YOLO("yolo-weights/yolov8l.pt")
model.train(data="/Users/kokkallahithasree/Downloads/Real-Time-Detection-of-Helmet-Violations-and-Capturing-Bike-Numbers-from-Number-Plates-main/archive/coco128.yaml", imgsz=320, batch=4, epochs=60, workers=0)
