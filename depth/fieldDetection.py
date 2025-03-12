from depthai_sdk import OakCamera, ArgsParser
from depthai_sdk.visualize.configs import BboxStyle, TextPosition
import argparse, time
import ntcore
import logging

logging.basicConfig(level=logging.DEBUG)

inst = ntcore.NetworkTableInstance.getDefault()
table = inst.getTable("SmartDashboard")

# networktables publisher: this publishes the distance and horizontal distance to the algae and publishes it over the network. There needs to be a server (roborio) on the network for this to work.
objHor = table.getDoubleTopic("objHorizontal").publish()
objDist = table.getDoubleTopic("objDistance").publish()
detectingNote = table.getBooleanTopic("detectingAlgae").publish()
inst.startClient4("example client")
inst.setServerTeam(4169)
inst.startDSClient()

parser = argparse.ArgumentParser()
parser.add_argument("-conf", "--config", help="Trained YOLO json config path", default='/home/aresuser/vision/depth/2025algae/best.json', type=str)
args = ArgsParser.parseArgs(parser)

def printDetections(packet):
    bestDetection = [1000000,1000000]
    
    
    
    for i in range(len(packet.detections)):
        if packet.detections[i].label_str == "Algae":
            if bestDetection[0] ** 2 + bestDetection[1] ** 2 > (packet.detections[i].img_detection.spatialCoordinates.y) ** 2 + (packet.detections[i].img_detection.spatialCoordinates.z) ** 2:
            	bestDetection = [packet.detections[i].img_detection.spatialCoordinates.y, packet.detections[i].img_detection.spatialCoordinates.z]
    if bestDetection != [1000000,1000000]:
        objHor.set(bestDetection[0])
        objDist.set(bestDetection[1])
        #print(f"objHorizontal: {bestDetection[0]}")
       # print(f"objDistance: {bestDetection[1]}")
    detectingNote.set(len(packet.detections) != 0)

    if bestDetection[0] > -100 and bestDetection[0] < 100:
        print("Center")
with OakCamera(args=args) as oak:
    color = oak.create_camera('color', fps=8)
    nn = oak.create_nn(args['config'], color, nn_type='yolo', spatial=True)
    visualizer = oak.visualize(nn.out.passthrough, callback=printDetections)
    visualizer = oak.visualize(nn.out.passthrough, fps=True)
    oak.start(blocking=True)
    
    
