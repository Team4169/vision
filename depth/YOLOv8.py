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

camRgb = pipeline.create(dai.node.ColorCamera)
camRgb.setPreviewSize(640, 640)
camRgb.setInterleaved(False)
camRgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_4_K)
camRgb.setIspScale(1, 3)
camRgb.setPreviewKeepAspectRatio(False)

xoutFrames = pipeline.create(dai.node.XLinkOut)
xoutFrames.setStreamName("frames")
camRgb.video.link(xoutFrames.input)

# Define a neural network that will make predictions based on the source frames
nn = pipeline.create(dai.node.YoloDetectionNetwork)
nn.setConfidenceThreshold(confidenceThreshold)
nn.setNumClasses(classes)
nn.setCoordinateSize(coordinates)
nn.setAnchors(anchors)
nn.setAnchorMasks(anchorMasks)
nn.setIouThreshold(iouThreshold)
nn.setBlobPath(nnPath)
nn.setNumInferenceThreads(2)
nn.input.setBlocking(False)
camRgb.preview.link(nn.input)

passthroughOut = pipeline.create(dai.node.XLinkOut)
passthroughOut.setStreamName("pass")
nn.passthrough.link(passthroughOut.input)

nnOut = pipeline.create(dai.node.XLinkOut)
nnOut.setStreamName("nn")
nn.out.link(nnOut.input)

# Connect to device and start pipeline
with dai.Device(pipeline) as device:
    qFrames = device.getOutputQueue(name="frames")
    qPass = device.getOutputQueue(name="pass")
    qDet = device.getOutputQueue(name="nn")

    detections = []
    fps = FPSHandler()
    text = TextHelper()

    # nn data (bounding box locations) are in <0..1> range - they need to be normalized with frame width/height
    def frameNorm(frame, bbox):
        normVals = np.full(len(bbox), frame.shape[0])
        normVals[::2] = frame.shape[1]
        return (np.clip(np.array(bbox), 0, 1) * normVals).astype(int)

    def displayFrame(name, frame):
        for detection in detections:
            bbox = frameNorm(frame, (detection.xmin, detection.ymin, detection.xmax, detection.ymax))
            text.putText(frame, labelMap[detection.label], (bbox[0] + 10, bbox[1] + 20))
            # text.putText(frame, labels[detection.label], (bbox[0] + 10, bbox[1] + 20))
            text.putText(frame, f"{int(detection.confidence * 100)}%", (bbox[0] + 10, bbox[1] + 40))
            text.rectangle(frame, bbox)
        # Show the frame
        cv2.imshow(name, frame)

    while True:
        frame = qFrames.get().getCvFrame()

        inDet = qDet.tryGet()
        if inDet is not None:
            detections = inDet.detections
            fps.next_iter()

        inPass = qPass.tryGet()
        if inPass is not None:
            displayFrame('Passthrough', inPass.getCvFrame())

        # If the frame is available, draw bounding boxes on it and show the frame
        text.putText(frame, "NN fps: {:.2f}".format(fps.fps()), (2, frame.shape[0] - 4))
        displayFrame("Frame", frame)

        if cv2.waitKey(1) == ord('q'):
            break