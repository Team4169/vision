'''
import torch

model = torch.load('/home/aresuser/Downloads/2025model/runs/detect/train/weights/best.pt')
dummy_input = torch.randn(1, 3, 640, 640)
torch.onnx.export(model, dummy_input, '2025Model.onnx')
'''

import torch
from ultralytics import YOLO

# Load the YOLOv8 model (YOLOv8n in your case)
model = YOLO('/home/aresuser/Downloads/2025model/runs/detect/train/weights/best.pt')

# Set the model to evaluation mode
model.eval()

# Create a dummy input tensor with the correct dimensions
dummy_input = torch.randn(1, 3, 640, 640)

# Export the model to ONNX format
model.export(format='onnx', imgsz=640)  # Automatically exports to ONNX with correct input size

