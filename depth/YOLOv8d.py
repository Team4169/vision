#!/usr/bin/env python3

import cv2, time, argparse, json, blobconverter
import depthai as dai
import numpy as np
from pathlib import Path

class TextHelper:
    def __init__(self) -> None:
        self.bg_color = (0, 0, 0)
        self.color = (255, 255, 255)
        self.text_type = cv2.FONT_HERSHEY_SIMPLEX
        self.line_type = cv2.LINE_AA
    def putText(self, frame, text, coords):
        cv2.putText(frame, text, coords, self.text_type, 1.0, self.bg_color, 3, self.line_type)
        cv2.putText(frame, text, coords, self.text_type, 1.0, self.color, 1, self.line_type)
    def rectangle(self, frame, bbox):
        x1,y1,x2,y2 = bbox
        cv2.rectangle(frame, (x1, y1), (x2, y2), self.bg_color, 3)
        cv2.rectangle(frame, (x1, y1), (x2, y2), self.color, 1)

class FPSHandler:
    def __init__(self):
        self.timestamp = time.time() + 1
        self.start = time.time()
        self.frame_cnt = 0
    def next_iter(self):
        self.timestamp = time.time()
        self.frame_cnt += 1
    def fps(self):
        return self.frame_cnt / (self.timestamp - self.start)

# parse arguments
parser = argparse.ArgumentParser()
parser.add_argument("-m", "--model", help="Provide model name or model path for inference",
                    default='/home/jetson/vision/depth/yolov8ntest1/YOLOv8nNORO_openvino_2022.1_6shave.blob', type=str)
parser.add_argument("-c", "--config", help="Provide config path for inference",
                    default='/home/jetson/vision/depth/yolov8ntest1/YOLOv8nNORO.json', type=str)
args = parser.parse_args()

# parse config
configPath = Path(args.config)
if not configPath.exists():
    raise ValueError("Path {} does not exist!".format(configPath))

with configPath.open() as f:
    config = json.load(f)
nnConfig = config.get("nn_config", {})

# parse input shape
if "input_size" in nnConfig:
    W, H = tuple(map(int, nnConfig.get("input_size").split('x')))

# extract metadata
metadata = nnConfig.get("NN_specific_metadata", {})
classes = metadata.get("classes", {})
coordinates = metadata.get("coordinates", {})
anchors = metadata.get("anchors", {})
anchorMasks = metadata.get("anchor_masks", {})
iouThreshold = metadata.get("iou_threshold", {})
confidenceThreshold = metadata.get("confidence_threshold", {})

# parse labels
nnMappings = config.get("mappings", {})
labels = nnMappings.get("labels", {})

# get model path
nnPath = args.model
if not Path(nnPath).exists():
    print("No blob found at {}. Looking into DepthAI model zoo.".format(nnPath))
    nnPath = str(blobconverter.from_zoo(args.model, shaves = 6, zoo_type = "depthai", use_cache=True))
# sync outputs
syncNN = True

labelMap = ["robot", "note"]

# Create pipeline
pipeline = dai.Pipeline()

# Define sources and outputs
camRgb = pipeline.create(dai.node.ColorCamera)
spatialDetectionNetwork = pipeline.create(dai.node.YoloSpatialDetectionNetwork)
monoLeft = pipeline.create(dai.node.MonoCamera)
monoRight = pipeline.create(dai.node.MonoCamera)
stereo = pipeline.create(dai.node.StereoDepth)
nnNetworkOut = pipeline.create(dai.node.XLinkOut)

xoutRgb = pipeline.create(dai.node.XLinkOut)
xoutNN = pipeline.create(dai.node.XLinkOut)
xoutDepth = pipeline.create(dai.node.XLinkOut)

xoutRgb.setStreamName("rgb")
xoutNN.setStreamName("detections")
xoutDepth.setStreamName("depth")
nnNetworkOut.setStreamName("nnNetwork")

# Properties
camRgb.setPreviewSize(640, 640)
camRgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
camRgb.setInterleaved(False)
camRgb.setColorOrder(dai.ColorCameraProperties.ColorOrder.BGR)
camRgb.setIspScale(1, 3)
camRgb.setPreviewKeepAspectRatio(False)

monoLeft.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
monoLeft.setCamera("left")
monoRight.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
monoRight.setCamera("right")

# setting node configs
stereo.setDefaultProfilePreset(dai.node.StereoDepth.PresetMode.HIGH_DENSITY)
# Align depth map to the perspective of RGB camera, on which inference is done
stereo.setDepthAlign(dai.CameraBoardSocket.CAM_A)
stereo.setOutputSize(monoLeft.getResolutionWidth(), monoLeft.getResolutionHeight())
stereo.setSubpixel(True)

spatialDetectionNetwork.setBlobPath(nnPath)
spatialDetectionNetwork.setConfidenceThreshold(0.5)
spatialDetectionNetwork.input.setBlocking(False)
spatialDetectionNetwork.setBoundingBoxScaleFactor(0.5)
spatialDetectionNetwork.setDepthLowerThreshold(100)
spatialDetectionNetwork.setDepthUpperThreshold(5000)

# Yolo specific parameters
spatialDetectionNetwork.setNumClasses(2)
spatialDetectionNetwork.setCoordinateSize(4)
spatialDetectionNetwork.setAnchors([10,14, 23,27, 37,58, 81,82, 135,169, 344,319])
spatialDetectionNetwork.setAnchorMasks({ "side26": [1,2,3], "side13": [3,4,5] })
spatialDetectionNetwork.setIouThreshold(0.5)

# Linking
monoLeft.out.link(stereo.left)
monoRight.out.link(stereo.right)

camRgb.preview.link(spatialDetectionNetwork.input)
if syncNN:
    spatialDetectionNetwork.passthrough.link(xoutRgb.input)
else:
    camRgb.preview.link(xoutRgb.input)

spatialDetectionNetwork.out.link(xoutNN.input)

stereo.depth.link(spatialDetectionNetwork.inputDepth)
spatialDetectionNetwork.passthroughDepth.link(xoutDepth.input)
spatialDetectionNetwork.outNetwork.link(nnNetworkOut.input)

# Connect to device and start pipeline
with dai.Device(pipeline) as device:

    # Output queues will be used to get the rgb frames and nn data from the outputs defined above
    previewQueue = device.getOutputQueue(name="rgb", maxSize=4, blocking=False)
    detectionNNQueue = device.getOutputQueue(name="detections", maxSize=4, blocking=False)
    depthQueue = device.getOutputQueue(name="depth", maxSize=4, blocking=False)
    networkQueue = device.getOutputQueue(name="nnNetwork", maxSize=4, blocking=False)

    startTime = time.monotonic()
    counter = 0
    fps = 0

    while True:
        inPreview = previewQueue.get()
        inDet = detectionNNQueue.get()
        depth = depthQueue.get()
        inNN = networkQueue.get()

        frame = inPreview.getCvFrame()
        depthFrame = depth.getFrame() # depthFrame values are in millimeters

        depth_downscaled = depthFrame[::4]
        if np.all(depth_downscaled == 0):
            min_depth = 0  # Set a default minimum depth value when all elements are zero
        else:
            min_depth = np.percentile(depth_downscaled[depth_downscaled != 0], 1)
        max_depth = np.percentile(depth_downscaled, 99)
        depthFrameColor = np.interp(depthFrame, (min_depth, max_depth), (0, 255)).astype(np.uint8)
        depthFrameColor = cv2.applyColorMap(depthFrameColor, cv2.COLORMAP_HOT)

        counter+=1
        current_time = time.monotonic()
        if (current_time - startTime) > 1 :
            fps = counter / (current_time - startTime)
            counter = 0
            startTime = current_time

        detections = inDet.detections

        # If the frame is available, draw bounding boxes on it and show the frame
        height = frame.shape[0]
        width  = frame.shape[1]
        for detection in detections:
            roiData = detection.boundingBoxMapping
            roi = roiData.roi
            roi = roi.denormalize(depthFrameColor.shape[1], depthFrameColor.shape[0])
            topLeft = roi.topLeft()
            bottomRight = roi.bottomRight()
            xmin = int(topLeft.x)
            ymin = int(topLeft.y)
            xmax = int(bottomRight.x)
            ymax = int(bottomRight.y)
            cv2.rectangle(depthFrameColor, (xmin, ymin), (xmax, ymax), (255, 255, 255), 1)

            # Denormalize bounding box
            x1 = int(detection.xmin * width)
            x2 = int(detection.xmax * width)
            y1 = int(detection.ymin * height)
            y2 = int(detection.ymax * height)
            try:
                label = labelMap[detection.label]
            except:
                label = detection.label
            cv2.putText(frame, str(label), (x1 + 10, y1 + 20), cv2.FONT_HERSHEY_TRIPLEX, 0.5, 255)
            cv2.putText(frame, "{:.2f}".format(detection.confidence*100), (x1 + 10, y1 + 35), cv2.FONT_HERSHEY_TRIPLEX, 0.5, 255)
            cv2.putText(frame, f"X: {int(detection.spatialCoordinates.x)} mm", (x1 + 10, y1 + 50), cv2.FONT_HERSHEY_TRIPLEX, 0.5, 255)
            cv2.putText(frame, f"Y: {int(detection.spatialCoordinates.y)} mm", (x1 + 10, y1 + 65), cv2.FONT_HERSHEY_TRIPLEX, 0.5, 255)
            cv2.putText(frame, f"Z: {int(detection.spatialCoordinates.z)} mm", (x1 + 10, y1 + 80), cv2.FONT_HERSHEY_TRIPLEX, 0.5, 255)

            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 255), cv2.FONT_HERSHEY_SIMPLEX)

        cv2.putText(frame, "NN fps: {:.2f}".format(fps), (2, frame.shape[0] - 4), cv2.FONT_HERSHEY_TRIPLEX, 2, (0, 0, 0))
        # cv2.imshow("depth", depthFrameColor)
        cv2.imshow("rgb", frame)

        if cv2.waitKey(1) == ord('q'):
            break