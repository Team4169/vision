import apriltag, cv2, subprocess
from math import sin, cos, atan2, pi
import numpy as np
#import ntcore
import pickle
from getmac import get_mac_address
from time import time

enable_network_tables = False

detector = apriltag.Detector(
apriltag.DetectorOptions(families='tag36h11',
                         border=1,
                         nthreads=4,
                         quad_decimate=0.0,
                         quad_blur=0.0,
                         refine_edges=True,
                         refine_decode=False,
                         refine_pose=False,
                         debug=True, # change maybe
                         quad_contours=False))

def findtags(cap, name):

    image = cap.read()[1]

    # <get params> v
    camera_matrix = cam_props[name]['cam_matrix']
    distortion_coefficients = cam_props[name]['dist']
    origin_offset = cam_props[name]['offset']
    # </get params> ^

    # <undistort> v
    h, w = image.shape[:2]
    new_camera_matrix, roi = cv2.getOptimalNewCameraMatrix(camera_matrix, distortion_coefficients, (w,h), 1, (w,h))
    mapx, mapy = cv2.initUndistortRectifyMap(camera_matrix, distortion_coefficients, None, new_camera_matrix, (w,h), 5)
    dst = cv2.remap(image, mapx, mapy, cv2.INTER_LINEAR) # faster than undistort.
    x, y, w, h = roi
    dst = dst[y:y+h, x:x+w]
    image = dst
    # </undistort> ^

    results = detector.detect(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY))

    posList = []
    rotList = []
    # loop over the AprilTag detection results, do math to find robots position and rotation and append to posList and rotList.
    for r in results:
        tagId = r.tag_id + 1
        if tagId not in range(1,17): # Competition only uses tags 1 to 16, if another one is found ignore it.
            continue

        # 0.085725 is for meters, use 3.375 of you want inches. This number is the size of the tags, change it if the tag size changes.
        object_points = np.array([[-0.085725,-0.085725,0],[0.085725,-0.085725,0],[0.085725,0.085725,0],[-0.085725,0.085725,0],[0,0,0]], dtype=np.float32)
        image_points = np.array([r.corners[0],r.corners[1],r.corners[2],r.corners[3],r.center], dtype=np.float32)

        _, rvec, tvec = cv2.solvePnP(object_points, image_points, camera_matrix, distortion_coefficients)
        for i, offset_num in enumerate(origin_offset):
            tvec[i] += offset_num

        # <Rotate Code> v
        # Based on the tag that we see, and which camera sees it, do math to find out where the robot must be to see that tag in that relative position and orientation.
        c=cos(field_tags[tagId][2]);s=sin(field_tags[tagId][2])
        tvec = [tvec[0]*c - tvec[2]*s, tvec[1], tvec[0]*s + tvec[2]*c]
        
        if name == 'Front':
            rvec[2] += -pi/2
        elif name == 'Back':
            rvec[2] += pi/2
        elif name == 'Right':
            rvec[2] += -pi
        #elif name == 'Left':
        #    rvec[2] += 0

        position = [field_tags[tagId][0] - tvec[0], field_tags[tagId][1] - tvec[2]]
        angle = float((field_tags[tagId][2] - rvec[2])[0])
        # </Rotate Code> ^
        
        posList.append(position)
        rotList.append(angle)

    return posList, rotList

def getJetson():
    MACdict = {"48:b0:2d:c1:63:9c" : 1, "48:b0:2d:ec:31:82" : 2, "48:b0:2d:c1:63:9b" : 1}
    return MACdict.get(get_mac_address(), None) # Returns None if invalid Jetson ID

def parse_v4l2_devices(output):
    mappings = {}
    lines = str(output).split('\\n\\n')
    for line in lines:
        if 'usb-70090000.xusb-' in line and '/dev/video' in line:
            mappings[line.split('usb-70090000.xusb-')[1][0:3]] = int(line.split('/dev/video')[1])
    return mappings

def get_v4l2_device_mapping():
    try:
        output = subprocess.check_output(['v4l2-ctl', '--list-devices'])
        output = str(output)[2:-1]
        return parse_v4l2_devices(output)
    except subprocess.CalledProcessError as e:
        print("Error occurred")
        return parse_v4l2_devices(e.output)
 
jetsonID = getJetson()
cam_0_name = ''
cam_1_name = ''
# Initialize cams with correct Jetson
if jetsonID == 1:
    cam_0_name = "Front"
    cam_1_name = "Right"
elif jetsonID == 2:
    cam_0_name = "Back"
    cam_1_name = "Left"
else:
    raise Exception('Invalid Jetson ID')
cam_mapping = get_v4l2_device_mapping()
cam_0 = cv2.VideoCapture(int(cam_mapping["2.1"]))
cam_1 = cv2.VideoCapture(int(cam_mapping["2.2"]))

# <Init Constants> v (cam_props and field_tags)
cam_props = {}
for cam_name in {cam_0_name, cam_1_name}:
    with open (f"/home/robotics4169/vision/calibration/camConfig/camConfig{cam_name}.pkl", 'rb') as f:
        f_data = pickle.load(f)
        cam_props[cam_name] = {'cam_matrix': f_data[0], 'dist': f_data[1], 'offset': f_data[2]}

with open (f"/home/robotics4169/vision/apriltags/maps/fieldTagsConfig.pkl", 'rb') as f:
    field_tags = pickle.load(f)
# <Init Constants> ^

# <Init NetworkTables> v
if enable_network_tables:
    inst = ntcore.NetworkTableInstance.getDefault()
    table = inst.getTable("SmartDashboard")

    wPub = table.getDoubleTopic("w1").publish()
    xPub = table.getDoubleTopic("x1").publish()
    yPub = table.getDoubleTopic("y1").publish()
    rPub = table.getDoubleTopic("r1").publish()
# <Init NetworkTables> ^

del getJetson, parse_v4l2_devices, get_v4l2_device_mapping # Delete these functions from memory because they are no longer needed

start_time = time()
frame_count = 0
while True:
    frame_count += 1
    print('FPS:',frame_count/(time()-start_time))
    
    posList0, rotList0 = findtags(cam_0, cam_0_name)
    posList1, rotList1 = findtags(cam_1, cam_1_name)
    fullPosList = posList0 + posList1
    fullRotList = rotList0 + rotList1

    if len(fullPosList) > 0:
        avg_pos = [sum(coord[0] for coord in fullPosList) / len(fullPosList), sum(coord[1] for coord in fullPosList) / len(fullPosList)]
        avg_rot = atan2(sum(sin(angle) for angle in fullRotList) / len(fullRotList), sum(cos(angle) for angle in fullRotList) / len(fullRotList)) % (2 * pi)

        # Set Network table values (Weight, Xposition, Yposition, and Rotation)
        if enable_network_tables:
            wPub.set(len(fullPosList))
            xPub.set(avg_pos[0])
            yPub.set(avg_pos[1])
            rPub.set(avg_rot)
        else:
            print(f"w: {len(fullPosList)}\nx: {avg_pos[0]}\ny: {avg_pos[1]}\nr: {avg_rot}\n")

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
