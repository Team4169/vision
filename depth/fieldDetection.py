from depthai_sdk import OakCamera, ArgsParser
from depthai_sdk.visualize.configs import BboxStyle, TextPosition
import argparse, time
import ntcore
import logging
from math import sqrt

logging.basicConfig(level=logging.DEBUG)

inst = ntcore.NetworkTableInstance.getDefault()
table = inst.getTable("SmartDashboard")

# networktables publisher: this publishes the distance and horizontal distance to the algae and publishes it over the network. There needs to be a server (roborio) on the network for this to work.
objHor = table.getDoubleTopic("objHorizontal").publish()
objDist = table.getDoubleTopic("objDistance").publish()
detectingAlgae = table.getBooleanTopic("detectingAlgae").publish()
algaeInIntake = table.getBooleanTopic("algaeInIntake").publish()
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
    detectingAlgae.set(len(packet.detections) != 0)
    # print(bestDetection)
    
    
    def dispCompass(objHor, objDist):
	    xInRange = False
	    yInRange = False
	    pickedUp = False
	    locked = False
	    
	    offset = 0
	    if 0.0 < abs(objHor) < 25.4:
		    yInRange = True
	    else:
		    offset = round(sqrt(abs(objHor)))
		    if  objHor < 0.0:
			    offset = -offset
			    #The offset cannot be greater than 10 because that's what each section of LEDs is limited to
		    if offset > 10:
			    offset = 10
		    if offset < -10: 
			    offset = -10
	    if 1040 < objDist < 1310:
		    xInRange = True
		    
	    if xInRange == True and yInRange == True:
		    locked = True
	    # time.sleep(1)

		    
	    lowerlimitL = 10 + offset
	    # detectingAlgae = True # FOR TESTING ONLY
	    if detectingAlgae == True: #This makes sure that the lights are only on if the robot is detecting algae
		    for i in range(lowerlimitL, lowerlimitL + 2): 
			    print(i)
		    for i in range(lowerlimitL + 23, lowerlimitL + 25):
			    print(i)
		    for i in range(lowerlimitL + 46, lowerlimitL + 48):
			    print(i)
	    print(bestDetection[0])
	    print(f"Distance = {bestDetection[1]}")
	    print(f"Offset = {offset}")
	    print(f"Locked = {locked}")
        
        
        
    dispCompass(bestDetection[0], bestDetection[1])
        
    
    
    
with OakCamera(args=args) as oak:
    color = oak.create_camera('color', fps=8)
    nn = oak.create_nn(args['config'], color, nn_type='yolo', spatial=True)
    visualizer = oak.visualize(nn.out.passthrough, callback=printDetections)
    visualizer = oak.visualize(nn.out.passthrough, fps=True)
    oak.start(blocking=True)
    
    
