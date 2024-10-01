# This code runs with back and left cams

import apriltag, cv2, subprocess
from math import sin, cos, atan2, pi
import numpy as np
import ntcore
import logging
import pickle
from portHandler import get_v4l2_device_mapping

pickled_cam_props = {}
with open ("/vision/apriltags/calibration/test/calibrationFiles/cameraMatrix.pkl", 'rb') as f:
    pickled_cam_props = pickle.load(f)
cam_props = {
'back':{'cam_matrix': np.array([[659.5522522254913, 0.0, 342.14593411596394], [0.0, 660.0855257028237, 233.07985632799412], [0.0, 0.0, 1.0]],dtype=np.float32), 'dist': np.array([[0.18218171362352173, -1.3943575501329653, -0.0034890991822150033, -0.003111058479543986, 2.4948141925852796]], dtype = np.float32), 'offset':np.array([0.3048,0.4572,0.22225],dtype=np.float32)},
'front':{'cam_matrix': np.array([[650.6665701168481, 0.0, 308.11247568203765], [0.0, 649.267759423238, 230.2397074540069], [0.0, 0.0, 1.0]],dtype=np.float32), 'dist': np.array([[0.142925049930884, -1.1502926269495592, -0.0019150557540761415, -0.00328202292619461, 1.8141065950524837]], dtype = np.float32), 'offset':np.array([0.1778,0.4572,0.19685],dtype=np.float32)},
'left':{'cam_matrix': np.array([[668.2138474014353, 0.0, 332.83301545896086], [0.0, 666.4860881212383, 214.33779667521517], [0.0, 0.0, 1.0]],dtype=np.float32), 'dist': np.array([[0.22224705297101408, -1.7549821808892665, -0.005523738126667523, 0.0051301529546101616, 3.4133532108023994]], dtype = np.float32), 'offset':np.array([-0.14605,0.4572,0.3302],dtype=np.float32)},
'right':{'cam_matrix': np.array([[660.6703723058181, 0.0, 321.2980455248988], [0.0, 658.6516133373474, 218.49261248405028], [0.0, 0.0, 1.0]],dtype=np.float32), 'dist': np.array([[0.18802634354539693, -1.5669527368643557, -0.0006972309753818612, -0.0018548904430247361, 3.04483663171066]], dtype = np.float32), 'offset':np.array([-0.13335,0.4572,0.2413],dtype=np.float32)}}

print(pickled_cam_props)
print(cam_props)
quit()
# Enviornment Initialization
# I used this, but the fmap could work as well.
FIELD_TAGS = [[0, 0, 0], [6.808597, -3.859403, (120+90)*pi/180], [7.914259, -3.221609, (120+90)*pi/180], [8.308467, 0.877443, (180+90)*pi/180], [8.308467, 1.442593, (180+90)*pi/180], [6.429883, 4.098925, (270+90)*pi/180 - 2*pi], [-6.429375, 4.098925, (270+90)*pi/180], [-8.308975, 1.442593, (0+90)*pi/180], [-8.308975, 0.877443, (0+90)*pi/180], [-7.914767, -3.221609, (60+90)*pi/180], [-6.809359, -3.859403, (60+90)*pi/180], [3.633851, -0.392049, (300+90)*pi/180], [3.633851, 0.393065, (60+90)*pi/180], [2.949321, -0.000127, (180+90)*pi/180], [-2.950083, -0.000127, (0+90)*pi/180], [-3.629533, 0.393065, (120+90)*pi/180], [-3.629533, -0.392049, (240+90)*pi/180]]


FIELD_TAGS_X = [FIELD_TAG[0] for FIELD_TAG in FIELD_TAGS]
FIELD_TAGS_Y = [FIELD_TAG[1] for FIELD_TAG in FIELD_TAGS]
FIELD_TAGS_ID = [i for i in range(len(FIELD_TAGS))]; FIELD_TAGS_ID[0] = 'x'

options = apriltag.DetectorOptions(families='tag36h11',
                                   border=1,
                                   nthreads=4,
                                   quad_decimate=0.0,
                                   quad_blur=0.0,
                                   refine_edges=True,
                                   refine_decode=False,
                                   refine_pose=False,
                                   debug=True,
                                   quad_contours=False)

detector = apriltag.Detector(options) 

def findtags(cap, name):

    image = cap.read()[1]

    #image = cv2.resize(image, (640,480))

    # <get params> v
    camera_matrix = cam_props[name]['cam_matrix']
    distortion_coefficients = cam_props[name]['dist']
    origin_offset = cam_props[name]['offset']
    # </get params> ^

    # <undistort> v
    h,  w = image.shape[:2]
    new_camera_matrix, roi = cv2.getOptimalNewCameraMatrix(camera_matrix, distortion_coefficients, (w,h), 1, (w,h))
    mapx, mapy = cv2.initUndistortRectifyMap(camera_matrix, distortion_coefficients, None, new_camera_matrix, (w,h), 5)
    dst = cv2.remap(image, mapx, mapy, cv2.INTER_LINEAR) #faster than undistort.
    x, y, w, h = roi
    dst = dst[y:y+h, x:x+w]
    image = dst
    # </undistort> ^

    results = detector.detect(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY))

    posList = []
    rotList = []
    # loop over the AprilTag detection results
    for r in results:
        tagId = r.tag_id + 1
        if tagId not in range(1,17):
            continue

        #0.085725 is for meters, use 3.375 of you want inches.
        object_points = np.array([[-0.085725,-0.085725,0],[0.085725,-0.085725,0],[0.085725,0.085725,0],[-0.085725,0.085725,0],[0,0,0]], dtype=np.float32)
        image_points = np.array([r.corners[0],r.corners[1],r.corners[2],r.corners[3],r.center], dtype=np.float32)

        _, rvec, tvec = cv2.solvePnP(object_points, image_points, camera_matrix, distortion_coefficients)
        for i, offset_num in enumerate(origin_offset):
            tvec[i] += offset_num

        # <Rotate Code> v
        c=cos(FIELD_TAGS[tagId][2]);s=sin(FIELD_TAGS[tagId][2])
        tvec = [tvec[0]*c - tvec[2]*s, tvec[1], tvec[0]*s + tvec[2]*c]

        rvec[2] += -pi/2
        if name == 'back':
            rvec[2] += pi
        elif name == 'left':
            rvec[2] += pi/2
        elif name == 'right':
            rvec[2] += -pi/2

        position = [FIELD_TAGS[tagId][0] - tvec[0], FIELD_TAGS[tagId][1] - tvec[2]]
        angle = FIELD_TAGS[tagId][2] - rvec[2]
        # </Rotate Code> ^

        posList.append(position)
        rotList.append(angle)

    return posList, rotList

# Init cams
cam_mapping = get_v4l2_device_mapping()
back_cap = cv2.VideoCapture(int(cam_mapping["2.1"]))
left_cap = cv2.VideoCapture(int(cam_mapping["2.2"]))

# <Init NetworkTables> v

logging.basicConfig(level=logging.DEBUG)

inst = ntcore.NetworkTableInstance.getDefault()

table = inst.getTable("SmartDashboard")

wPub = table.getDoubleTopic("w1").publish()
xPub = table.getDoubleTopic("y1").publish()
yPub = table.getDoubleTopic("x1").publish()
rPub = table.getDoubleTopic("r1").publish()

# <Init NetworkTables> ^

while True:
    try:
        fullPosList, fullRotList = [], []
        posList0, rotList0 = findtags(back_cap, "back")
        posList1, rotList1 = findtags(left_cap, "left")
        fullPosList.extend(posList0); fullPosList.extend(posList1)
        fullRotList.extend(rotList0); fullRotList.extend(rotList1)

        if len(fullPosList) > 0:

            avg_pos = [sum(coord[0] for coord in fullPosList) / len(fullPosList), sum(coord[1] for coord in fullPosList) / len(fullPosList)]
            avg_rot = atan2(sum(sin(angle) for angle in fullRotList) / len(fullRotList), sum(cos(angle) for angle in fullRotList) / len(fullRotList)) % (2 * pi)

            # Set Network table values (Weight, Xposition, Yposition, and Rotation)
            wPub.set(len(fullPosList))
            xPub.set(avg_pos[0])
            yPub.set(avg_pos[1])
            rPub.set(avg_rot)


    except(KeyboardInterrupt):
        back_cap.release()
        left_cap.release()

        print("/nCams Off. Program ended.")
        cv2.destroyAllWindows()
        break