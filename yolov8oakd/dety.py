from ultralytics import YOLO

# Load a model
# build a new model from scratch
model = YOLO("newReal.pt")  # load a pretrained model (recommended for 

success = model.export(format="onnx")

