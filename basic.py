# Install required packages
# pip install torch torchvision
# pip install cython
# pip install 'git+https://github.com/cocodataset/cocoapi.git#subdirectory=PythonAPI'

# Import necessary libraries
import torch
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.transforms import Compose, ToTensor, Resize
from torchvision.datasets import CocoDetection
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor, FasterRCNN_ResNet50_FPN_Weights
from torch.utils.data import DataLoader

# Define transformation and dataset
transform = Compose([Resize((256, 256)), ToTensor()])
train_dataset = CocoDetection(root='2023dataset/1/train', annFile='2023dataset/1/train/_annotations.coco.json', transform=transform)
# Ensure that item[1] has at least one element before accessing its index
train_dataset = [(item[0], (item[1][0],)) if item[1] else item for item in train_dataset]

print("a")

# Create dataloader
train_dataloader = DataLoader(train_dataset, batch_size=2, shuffle=True, num_workers=2)

# Load pre-trained Faster R-CNN model
model = fasterrcnn_resnet50_fpn(weights=FasterRCNN_ResNet50_FPN_Weights.DEFAULT)

# Modify the model's classifier for the number of classes in the COCO dataset (assuming 91 classes including background)
num_classes = 2  # Change this according to your dataset
in_features = model.roi_heads.box_predictor.cls_score.in_features
model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

# Set device and optimizer
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
model.to(device)
optimizer = torch.optim.SGD(model.parameters(), lr=0.005, momentum=0.9)

# Training loop
num_epochs = 5
for epoch in range(num_epochs):
    for images, targets in train_dataloader:
        images = [image.to(device) for image in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        optimizer.zero_grad()
        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())
        losses.backward()
        optimizer.step()

# Save the trained model
torch.save(model.state_dict(), 'fasterrcnn_coco.pth')
