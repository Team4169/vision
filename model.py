import torch
import torchvision
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator
from torchvision.transforms import functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import functional as F
from torchvision.datasets import CocoDetection
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor, FasterRCNN_ResNet50_FPN_Weights
from PIL import Image

class CustomDataset(Dataset):
    def __init__(self, root, annFile, transforms=None, train=True, image_size=(256, 256)):
        self.coco = CocoDetection(root, annFile)
        self.transforms = transforms
        self.train = train
        self.image_size = image_size

    def __getitem__(self, idx):
        img, target = self.coco.__getitem__(idx)
        img = F.resize(img, self.image_size, Image.BICUBIC)
        img = F.to_tensor(img)
        if self.transforms is not None:
            img = self.transforms(img)
        print(img.shape)
        print(target)
        return img, target

    def __len__(self):
        return len(self.coco)

root_path = "dataset"
train_dataset = CustomDataset(root=root_path + "/train", annFile=root_path + "/train/_annotations.coco.json", train=True)
test_dataset = CustomDataset(root=root_path + "/test", annFile=root_path + "/test/_annotations.coco.json", train=False)
valid_dataset = CustomDataset(root=root_path + "/valid", annFile=root_path + "/valid/_annotations.coco.json", train=False)

train_dataloader = DataLoader(train_dataset, batch_size=2, shuffle=True, num_workers=2)
test_dataloader = DataLoader(test_dataset, batch_size=2, shuffle=False, num_workers=2)
valid_dataloader = DataLoader(valid_dataset, batch_size=2, shuffle=False, num_workers=2)

model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights=FasterRCNN_ResNet50_FPN_Weights.DEFAULT)
num_classes = 3
in_features = model.roi_heads.box_predictor.cls_score.in_features
model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

params = [p for p in model.parameters() if p.requires_grad]
optimizer = torch.optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)
lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)

num_epochs = 10
for epoch in range(num_epochs):
    # Training
    model.train()
    for images, targets in train_dataloader:
        # Move images to the device individually
        images = [img.to(device) for img in images]
        if isinstance(targets[0], list):
            targets = [{k: v[0].to(device) if not isinstance(v, list) else v for k, v in t.items()} for t in targets]
        else:
            targets = [{k: v.to(device) if not isinstance(v, list) else v for k, v in t.items()} for t in targets]

        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())

        optimizer.zero_grad()
        losses.backward()
        optimizer.step()
        
    # Validation
    model.eval()
    all_predictions = []
    all_labels = []

    with torch.no_grad():
        for images, targets in valid_dataloader:
            images = list(img.to(device) for img in images)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            predictions = model(images)
            all_predictions.extend(predictions)
            all_labels.extend(targets)

    # Calculate accuracy
    flat_predictions = [p["labels"].cpu().numpy() for p in all_predictions]
    flat_labels = [t["labels"].cpu().numpy() for t in all_labels]

    accuracy = accuracy_score(flat_labels, flat_predictions)
    print(f"Epoch {epoch + 1}/{num_epochs}, Validation Accuracy: {accuracy}")

    # Update the learning rate
    lr_scheduler.step()

# Testing
model.eval()
all_predictions = []
all_labels = []

with torch.no_grad():
    for images, targets in test_dataloader:
        images = list(img.to(device) for img in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        predictions = model(images)
        all_predictions.extend(predictions)
        all_labels.extend(targets)

# Calculate accuracy
flat_predictions = [p["labels"].cpu().numpy() for p in all_predictions]
flat_labels = [t["labels"].cpu().numpy() for t in all_labels]

accuracy = accuracy_score(flat_labels, flat_predictions)
print(f"Testing Accuracy: {accuracy}")
