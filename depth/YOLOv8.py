from depthai_sdk import OakCamera, ArgsParser
from depthai_sdk.visualize.configs import BboxStyle, TextPosition
import argparse, time

# parse arguments
parser = argparse.ArgumentParser()
parser.add_argument("-conf", "--config", help="Trained YOLO json config path", default='/home/team4169/vision/depth/2024model/YOLOv8nNORO.json', type=str)
args = ArgsParser.parseArgs(parser)

def printDetections(packet):
    print(len(packet.detections))
    for i in range(len(packet.detections)):
        print(packet.detections[i].confidence)
        print(packet.detections[i].label_str)
        print(packet.detections[i].img_detection.spatialCoordinates.x, packet.detections[i].img_detection.spatialCoordinates.y, packet.detections[i].img_detection.spatialCoordinates.z)

with OakCamera(args=args) as oak:
    color = oak.create_camera('color')
    nn = oak.create_nn(args['config'], color, nn_type='yolo', spatial=True)
    visualizer = oak.visualize(nn.out.passthrough, callback=printDetections)
    visualizer = oak.visualize(nn.out.passthrough, fps=True)
    oak.start(blocking=True)