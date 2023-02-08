from ultralytics import YOLO

model = YOLO("v8.pt")
results = model(0)