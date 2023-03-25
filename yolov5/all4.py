#!/usr/bin/env python3

import cv2
import depthai as dai
import numpy as np
import os
import platform
import sys
from pathlib import Path

import torch

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

from models.common import DetectMultiBackend
from utils.dataloaders import IMG_FORMATS, VID_FORMATS, LoadImages, LoadScreenshots, LoadStreams
from utils.general import (LOGGER, Profile, check_file, check_img_size, check_imshow, non_max_suppression, scale_boxes, strip_optimizer, xyxy2xywh)
from utils.plots import Annotator, colors, save_one_box
from utils.torch_utils import select_device, smart_inference_mode
from utils.augmentations import letterbox

import cv2, math, ntcore

torch.no_grad()

weights=ROOT / 'v5.pt' # model path or triton URL
source=ROOT / '0' # file/dir/URL/glob/screen/0(webcam)
data=ROOT / 'data/coco128.yaml' # dataset.yaml path
imgsz=(416, 640) # inference size (height, width)
conf_thres=0.4 # confidence threshold
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
save_img = not nosave and not source.endswith('.txt')  # save inference images
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

# Run inference
windows, dt = [], (Profile(), Profile(), Profile())

# Network Tables
inst = ntcore.NetworkTableInstance.getDefault()
table=inst.getTable("SmartDashboard")
xSub=table.getDoubleTopic("x").publish()
ySub=table.getDoubleTopic("y").publish()
distSub = table.getDoubleTopic("dist").publish()
seeSub = table.getDoubleTopic("see").publish()
inst.startClient4("example client")
inst.setServerTeam(4169)
inst.startDSClient()

# DEPTH
newConfig = False

# Create pipeline
pipeline = dai.Pipeline()

# Define sources and outputs
camRgb = pipeline.create(dai.node.ColorCamera)
camRgb.setPreviewSize(640, 416)
camRgb.setFps(4)
monoLeft = pipeline.create(dai.node.MonoCamera)
monoRight = pipeline.create(dai.node.MonoCamera)
stereo = pipeline.create(dai.node.StereoDepth)
spatialLocationCalculator = pipeline.create(dai.node.SpatialLocationCalculator)

xoutDepth = pipeline.create(dai.node.XLinkOut)
xoutSpatialData = pipeline.create(dai.node.XLinkOut)
xinSpatialCalcConfig = pipeline.create(dai.node.XLinkIn)
xoutCamRgb = pipeline.create(dai.node.XLinkOut)

xoutDepth.setStreamName("depth")
xoutSpatialData.setStreamName("spatialData")
xinSpatialCalcConfig.setStreamName("spatialCalcConfig")
xoutCamRgb.setStreamName("rgb")

camRgb.preview.link(xoutCamRgb.input)

# Properties
monoLeft.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
monoLeft.setBoardSocket(dai.CameraBoardSocket.LEFT)
monoRight.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
monoRight.setBoardSocket(dai.CameraBoardSocket.RIGHT)

lrcheck = False
subpixel = False

stereo.setDefaultProfilePreset(dai.node.StereoDepth.PresetMode.HIGH_DENSITY)
stereo.setLeftRightCheck(lrcheck)
stereo.setSubpixel(subpixel)

# Config
topLeft = dai.Point2f(0.4, 0.4)
bottomRight = dai.Point2f(0.6, 0.6)

config = dai.SpatialLocationCalculatorConfigData()
config.depthThresholds.lowerThreshold = 100
config.depthThresholds.upperThreshold = 10000
config.roi = dai.Rect(topLeft, bottomRight)

spatialLocationCalculator.inputConfig.setWaitForMessage(False)
spatialLocationCalculator.initialConfig.addROI(config)

# Linking
monoLeft.out.link(stereo.left)
monoRight.out.link(stereo.right)

spatialLocationCalculator.passthroughDepth.link(xoutDepth.input)
stereo.depth.link(spatialLocationCalculator.inputDepth)

spatialLocationCalculator.out.link(xoutSpatialData.input)
xinSpatialCalcConfig.out.link(spatialLocationCalculator.inputConfig)

# Connect to device and start pipeline
with dai.Device(pipeline) as device:


    # Output queue will be used to get the depth frames from the outputs defined above
    depthQueue = device.getOutputQueue(name="depth", maxSize=4, blocking=False)
    spatialCalcQueue = device.getOutputQueue(name="spatialData", maxSize=4, blocking=False)
    spatialCalcConfigInQueue = device.getInputQueue("spatialCalcConfig")
    rgbQueue = device.getOutputQueue(name="rgb", maxSize=4, blocking=False)

    color = (255, 255, 255)

    while True:
        inDepth = depthQueue.get() # Blocking call, will wait until a new data has arrived

        depthFrame = inDepth.getFrame() # depthFrame values are in millimeters
        rgbFrame = rgbQueue.get().getFrame()
        
        depthFrame = np.flip(depthFrame, (0, 1))
        rgbFrame = np.flip(rgbFrame, (0, 1))
        
        im = rgbFrame
        im = letterbox(im, imgsz, stride=stride, auto=pt)[0]
        im = im.transpose((2, 0, 1))[::-1]
        im = np.ascontiguousarray(im)

        s= ""
        outxyxy = []
        outxywh = []
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
            annotator = Annotator(np.ascontiguousarray(rgbFrame), line_width=line_thickness, example=str(names))
            if len(det):
                # Print results
                for c in det[:, 5].unique():
                    n = (det[:, 5] == c).sum()  # detections per class
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string
                for *xyxy, conf, cls in reversed(det):
                    # print(xyxy)
                    xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                    outxyxy.append(xyxy.copy())
                    outxywh.append(xywh.copy())
                    # print(cls, *xywh, conf)

                    c = int(cls)  # integer class
                    label = f'{names[c]} {conf:.2f}'
                    annotator.box_label(xyxy, label, color=colors(c, True))

            # Stream results
            outim = annotator.result()
            cv2.imshow("output", outim)
            cv2.waitKey(1)  # 1 millisecond
        
        depthFrameColor = cv2.normalize(depthFrame, None, 255, 0, cv2.NORM_INF, cv2.CV_8UC1)
        depthFrameColor = cv2.equalizeHist(depthFrameColor)
        depthFrameColor = cv2.applyColorMap(depthFrameColor, cv2.COLORMAP_HOT)
        
        minDistIndex = 0
        outDistList = []
        for i in range(len(outxyxy)):
            xmin = int(outxyxy[i][0])
            ymin = int(outxyxy[i][1])
            xmax = int(outxyxy[i][2])
            ymax = int(outxyxy[i][3])
            # print(xmin, ymin, xmax, ymax)
            if abs(xmax - xmin) > 0 and abs(ymax - ymin) > 0:
            	outdist = np.average(depthFrame[int(outxyxy[i][0]) : int(outxyxy[i][2]), int(outxyxy[i][1]) : int(outxyxy[i][3])])
            	outDistList.append(outdist)
            else:
                continue
            # print(outdist)
            if not math.isnan(outdist):
                cv2.rectangle(depthFrameColor, (xmin, ymin), (xmax, ymax), color, cv2.FONT_HERSHEY_SCRIPT_SIMPLEX)
                cv2.putText(depthFrameColor, f"Z: {int(outdist)} mm", (xmin + 10, ymin + 20), cv2.FONT_HERSHEY_TRIPLEX, 0.5, 255)
            if 750 < outdist < minDistIndex:
                minDistIndex = i
            
        cv2.imshow("depth", depthFrameColor)
        if len(outDistList) > 0 and not math.isnan(outDistList[minDistIndex]):
            xSub.set(float((outxyxy[minDistIndex][0] + outxyxy[minDistIndex][2]) / 2))
            ySub.set(float((outxyxy[minDistIndex][1] + outxyxy[minDistIndex][3]) / 2))
            distSub.set(float(outDistList[minDistIndex]))
            seeSub.set(float(1))
        else:
            seeSub.set(float(0))
        
        key = cv2.waitKey(1)
