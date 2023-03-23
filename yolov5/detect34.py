import os
import sys
from pathlib import Path
from PIL import Image

import torch
import numpy as np

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

from models.common import DetectMultiBackend
from utils.dataloaders import IMG_FORMATS, VID_FORMATS
from utils.general import (LOGGER, Profile, check_file, check_img_size, check_imshow, non_max_suppression, xyxy2xywh)
from utils.plots import Annotator, colors
from utils.torch_utils import select_device
from utils.augmentations import letterbox 

import cv2

torch.no_grad()

weights=ROOT / 'v5.pt' # model path or triton URL
source=ROOT / 'testy.png' # file/dir/URL/glob/screen/0(webcam)
data=ROOT / 'data/coco128.yaml' # dataset.yaml path
imgsz=(416, 640) # inference size (height, width)
conf_thres=0.25 # confidence threshold
iou_thres=0.45 # NMS IOU threshold
max_det=1000 # maximum detections per image
device=0 # cuda device, i.e. 0 or 0,1,2,3 or cpu
view_img=False # show results
nosave=False # do not save images/videos
classes=None # filter by class: --class 0, or --class 0 2 3
agnostic_nms=False # class-agnostic NMS
augment=False # augmented inference
visualize=False # visualize features
update=False # update all models
project=ROOT / 'runs/detect' # save results to project/name
name='exp' # save results to project/name
exist_ok=False # existing project/name ok, do not increment
line_thickness=3 # bounding box thickness (pixels)
half=False # use FP16 half-precision inference
dnn=False # use OpenCV DNN for ONNX inference
vid_stride=1 # video frame-rate stride

# Load model
device = select_device(device)
model = DetectMultiBackend(weights, device=device, dnn=dnn, data=data, fp16=half)
stride, names, pt = model.stride, model.names, model.pt

vc = cv2.VideoCapture(0)

if vc.isOpened(): # try to get the first frame
    rval, frame = vc.read()
else:
    rval = False
    
# Run inference
windows, dt = [], (Profile(), Profile(), Profile())
while rval:
    rval, frame = vc.read()
    key = cv2.waitKey(20)
    if key == 27: # exit on ESC
        break
        
    frame = frame[:320, :640, :]
    im = letterbox(frame, imgsz, stride=stride, auto=pt)[0]
    im = im.transpose((2, 0, 1))[::-1]
    im = np.ascontiguousarray(im)
    
    a = im.reshape(320, 640, 3)
    Image.fromarray(a).save("newtest.png")

    s= ""
    with dt[0]:
        im = torch.from_numpy(im).to(model.device)
        im = im.half() if model.fp16 else im.float()  # uint8 to fp16/32
        im /= 255  # 0 - 255 to 0.0 - 1.0
        if len(im.shape) == 3:
            im = im[None]  # expand for batch dim

    # Inference
    with dt[1]:
        pred = model(im, augment=augment)

    # NMS
    with dt[2]:
        pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)

    # Process predictions
    for i, det in enumerate(pred):  # per image
        s += f'{i}: '
        s += '%gx%g ' % im.shape[2:]  # print string
        gn = torch.tensor([640, 416, 640, 416]) # normalization gain whwh
        annotator = Annotator(np.ascontiguousarray(frame), line_width=line_thickness, example=str(names))
        if len(det):
            # Print results
            for c in det[:, 5].unique():
                n = (det[:, 5] == c).sum()  # detections per class
                s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string
            for *xyxy, conf, cls in reversed(det):
                xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                print(cls, *xywh, conf)

                c = int(cls)  # integer class
                label = f'{names[c]} {conf:.2f}'
                annotator.box_label(xyxy, label, color=colors(c, True))


        # Stream results
        outim = annotator.result()
        cv2.imshow("output", outim)
        cv2.waitKey(1)  # 1 millisecond

    # Print time (inference-only)
    print(f"{s}{'' if len(det) else '(no detections), '}{dt[1].dt * 1E3:.1f}ms")
