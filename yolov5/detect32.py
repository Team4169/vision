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
from utils.general import (LOGGER, Profile, check_file, check_img_size, check_imshow, cv2, non_max_suppression, xyxy2xywh)
from utils.plots import Annotator, colors
from utils.torch_utils import select_device
from utils.augmentations import letterbox


class LoadImages:
    def __init__(self, path, img_size=640, stride=32, auto=True):
        self.img_size = img_size
        self.stride = stride
        self.files = []
        self.nf = 1
        self.auto = auto

    def __iter__(self):
        self.count = 0
        return self

    def __next__(self):
        if self.count == self.nf:
            raise StopIteration
        path = self.files[self.count]
        self.count += 1
        im0 = cv2.imread(path)

        im = letterbox(im0, self.img_size, stride=self.stride, auto=self.auto)[0]
        im = im.transpose((2, 0, 1))[::-1]
        im = np.ascontiguousarray(im)

        return im, im0

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

source = str(source)
is_file = Path(source).suffix[1:] in (IMG_FORMATS + VID_FORMATS)
is_url = source.lower().startswith(('rtsp://', 'rtmp://', 'http://', 'https://'))
webcam = source.isnumeric() or source.endswith('.streams') or (is_url and not is_file)
screenshot = source.lower().startswith('screen')
if is_url and is_file:
    source = check_file(source)  # download

# Load model
device = select_device(device)
model = DetectMultiBackend(weights, device=device, dnn=dnn, data=data, fp16=half)
stride, names, pt = model.stride, model.names, model.pt
imgsz = check_img_size(imgsz, s=stride)  # check image size

im0 = cv2.imread('/home/jetson/objectdetection/yolov5/testy.png')

im = letterbox(im0, imgsz, stride=stride, auto=pt)[0]
im = im.transpose((2, 0, 1))[::-1]
im = np.ascontiguousarray(im)

a = im.reshape(320, 640, 3)
Image.fromarray(a).save("newtest2.png")
quit()

# Run inference
windows, dt = [], (Profile(), Profile(), Profile())
for im, im0 in dataset:
    s= ""
    with dt[0]:
        a = im.reshape(320, 640, 3)
        Image.fromarray(a).save("newtest.png")
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
        annotator = Annotator(im0[0], line_width=line_thickness, example=str(names))
        if len(det):
            # Print results
            for c in det[:, 5].unique():
                n = (det[:, 5] == c).sum()  # detections per class
                s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string
            for *xyxy, conf, cls in reversed(det):
                xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                print(cls, *xywh, conf)

                if view_img:  # Add bbox to image
                    c = int(cls)  # integer class
                    label = f'{names[c]} {conf:.2f}'
                    annotator.box_label(xyxy, label, color=colors(c, True))


        # Stream results
        outim = annotator.result()
        cv2.imshow("output", outim)
        cv2.waitKey(1)  # 1 millisecond

    # Print time (inference-only)
    print(f"{s}{'' if len(det) else '(no detections), '}{dt[1].dt * 1E3:.1f}ms")
