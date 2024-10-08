from depthai_sdk import OakCamera, ArgsParser
from depthai_sdk.visualize.configs import BboxStyle, TextPosition
import argparse, time
import ntcore
import logging

logging.basicConfig(level=logging.DEBUG)

inst = ntcore.NetworkTableInstance.getDefault()
table = inst.getTable("SmartDashboard")


inMid = table.getBooleanTopic("inMid").publish()
inst.startClient4("example client")
inst.setServerTeam(4169)
inst.startDSClient()

parser = argparse.ArgumentParser()
parser.add_argument("-conf", "--config", help="Trained YOLO json config path", default='/home/aresuser/vision/depth/2024model/YOLOv8nNORO.json', type=str)
args = ArgsParser.parseArgs(parser)

def printDetections(packet):
    detection = False
    for i in range(len(packet.detections)):
        if packet.detections[i].label_str == "Class_1":
            print(packet.detections[i].bbox)
            print("detecting")
            detection = True        
    if not detection:
        print("not detecting cuh")
    inMid.set(detection)

with OakCamera(args=args) as oak:
    color = oak.create_camera('color', resolution=100)
    nn = oak.create_nn(args['config'], color, nn_type='yolo')
    nn.config_nn(conf_threshold=0.5)
    visualizer = oak.visualize(nn.out.passthrough, callback=printDetections, fps = True)
    #visualizer = oak.visualize(nn.out.passthrough, fps=True)
    oak.start(blocking=True)
    
