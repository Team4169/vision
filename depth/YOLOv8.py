from depthai_sdk import OakCamera, ArgsParser
from depthai_sdk.visualize.configs import BboxStyle, TextPosition
import argparse, time
# from networktables import NetworkTables
import logging

logging.basicConfig(level=logging.DEBUG)

# NetworkTables.initialize()
# sd = NetworkTables.getTable("SmartDashboard")

parser = argparse.ArgumentParser()
parser.add_argument("-conf", "--config", help="Trained YOLO json config path", default='/home/aresuser/vision/depth/2025cage/best.json', type=str)
args = ArgsParser.parseArgs(parser)

def printDetections(packet):
    print(len(packet.detections))
    for i in range(len(packet.detections)):
        if packet.detections[i].label_str == "Class_1":
            #print(packet.detections[i].confidence)
            #print(packet.detections[i].label_str)
            #print(packet.detections[i].img_detection.spatialCoordinates.x, packet.detections[i].img_detection.spatialCoordinates.y, packet.detections[i].img_detection.spatialCoordinates.z)
            print(f"objHorizontal: {packet.detections[i].img_detection.spatialCoordinates.y}")
            print(f"objDistance: {packet.detections[i].img_detection.spatialCoordinates.z}")
            # sd.putNumber("objHorizontal", packet.detections[i].img_detection.spatialCoordinates.y + 0.1524)
            # sd.putNumber("objDistance", packet.detections[i].img_detection.spatialCoordinates.z + 0.2159)
        

with OakCamera(args=args) as oak:
    color = oak.create_camera('color')
    nn = oak.create_nn(args['config'], color, nn_type='yolo', spatial=True)
    visualizer = oak.visualize(nn.out.passthrough, callback=printDetections)
    visualizer = oak.visualize(nn.out.passthrough, fps=True)
    oak.start(blocking=True)
