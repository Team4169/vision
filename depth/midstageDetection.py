from depthai_sdk import OakCamera, ArgsParser
from depthai_sdk.visualize.configs import BboxStyle, TextPosition
import argparse, time
from networktables import NetworkTables
import logging

logging.basicConfig(level=logging.DEBUG)

NetworkTables.initialize()
sd = NetworkTables.getTable("SmartDashboard")

parser = argparse.ArgumentParser()
parser.add_argument("-conf", "--config", help="Trained YOLO json config path", default='/home/jetson/vision/depth/2024model/YOLOv8nNORO.json', type=str)
args = ArgsParser.parseArgs(parser)

def printDetections(packet):
    detection = False
    for i in range(len(packet.detections)):
        if packet.detections[i].label_str == "Class_1" and packet.detections[i].confidence > 0.7:
            print(packet.detections[i].bbox)
            print("detecting")
            detection = True        
    if not detection:
        print("not detecting cuh")
    sd.putBoolean("inMid", detection)

with OakCamera(args=args) as oak:
    color = oak.create_camera('color', resolution=100)
    nn = oak.create_nn(args['config'], color, nn_type='yolo')
    visualizer = oak.visualize(nn.out.passthrough, callback=printDetections, fps = True)
    visualizer = oak.visualize(nn.out.passthrough, fps=True)
    oak.start(blocking=True)
    
