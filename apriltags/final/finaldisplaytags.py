import apriltag, cv2, subprocess
from math import sin, cos, atan2, pi
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from networktables import NetworkTables
import logging

cam_props = {
'back':{'cam_matrix': np.array([[659.5522522254913, 0.0, 342.14593411596394], [0.0, 660.0855257028237, 233.07985632799412], [0.0, 0.0, 1.0]],dtype=np.float32), 'dist': np.array([[0.18218171362352173, -1.3943575501329653, -0.0034890991822150033, -0.003111058479543986, 2.4948141925852796]], dtype = np.float32), 'offset':np.array([0.3048,0.4572,0.22225],dtype=np.float32)},
'front':{'cam_matrix': np.array([[650.6665701168481, 0.0, 308.11247568203765], [0.0, 649.267759423238, 230.2397074540069], [0.0, 0.0, 1.0]],dtype=np.float32), 'dist': np.array([[0.142925049930884, -1.1502926269495592, -0.0019150557540761415, -0.00328202292619461, 1.8141065950524837]], dtype = np.float32), 'offset':np.array([0.1778,0.4572,0.19685],dtype=np.float32)},
'left':{'cam_matrix': np.array([[668.2138474014353, 0.0, 332.83301545896086], [0.0, 666.4860881212383, 214.33779667521517], [0.0, 0.0, 1.0]],dtype=np.float32), 'dist': np.array([[0.22224705297101408, -1.7549821808892665, -0.005523738126667523, 0.0051301529546101616, 3.4133532108023994]], dtype = np.float32), 'offset':np.array([-0.14605,0.4572,0.3302],dtype=np.float32)},
'right':{'cam_matrix': np.array([[660.6703723058181, 0.0, 321.2980455248988], [0.0, 658.6516133373474, 218.49261248405028], [0.0, 0.0, 1.0]],dtype=np.float32), 'dist': np.array([[0.18802634354539693, -1.5669527368643557, -0.0006972309753818612, -0.0018548904430247361, 3.04483663171066]], dtype = np.float32), 'offset':np.array([-0.13335,0.4572,0.2413],dtype=np.float32)}}

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
    dst = cv2.remap(image, mapx, mapy, cv2.INTER_LINEAR)
    x, y, w, h = roi
    dst = dst[y:y+h, x:x+w]
    image = dst
    # </undistort> ^

    results = detector.detect(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY))

    posList = []
    rotList = []
    # loop over the AprilTag detection results
    for r in results:

        if r.tag_id not in range(1,17):
            continue

        # extract the bounding box (x, y)-coordinates for the AprilTag
        # and convert each of the (x, y)-coordinate pairs to integers
        (ptA, ptB, ptC, ptD) = r.corners
        iptB = (int(ptB[0]), int(ptB[1]))
        iptC = (int(ptC[0]), int(ptC[1]))
        iptD = (int(ptD[0]), int(ptD[1]))
        iptA = (int(ptA[0]), int(ptA[1]))

        # draw the bounding box of the AprilTag detection
        cv2.line(image, iptA, iptB, (255, 0, 0), 5)
        cv2.line(image, iptB, iptC, (255, 0, 0), 5)
        cv2.line(image, iptC, iptD, (255, 0, 0), 5)
        cv2.line(image, iptD, iptA, (255, 0, 0), 5)
        cv2.line(image, iptA, iptC, (255, 0, 0), 5)
        cv2.line(image, iptB, iptD, (255, 0, 0), 5)

        # draw the center (x, y)-coordinates of the AprilTag
        icenter = (int(r.center[0]), int(r.center[1]))
        cv2.circle(image, icenter, 10, (0, 0, 255), -1)

        #0.085725 is for meters, use 3.375 of you want inches.
        object_points = np.array([[-0.085725,-0.085725,0],[0.085725,-0.085725,0],[0.085725,0.085725,0],[-0.085725,0.085725,0],[0,0,0]], dtype=np.float32)

        image_points = np.array([ptA,ptB,ptC,ptD,r.center], dtype=np.float32)

        _, rvec, tvec = cv2.solvePnP(object_points, image_points, camera_matrix, distortion_coefficients)

        for i, offset_num in enumerate(origin_offset):
            tvec[i] += offset_num
        # <Rotate Code> v
        c=cos(FIELD_TAGS[r.tag_id][2]);s=sin(FIELD_TAGS[r.tag_id][2])
        tvec = [tvec[0]*c - tvec[2]*s, tvec[1], tvec[0]*s + tvec[2]*c]

        rvec[2] += -pi/2
        if name == 'back':
            rvec[2] += pi
        elif name == 'left':
            rvec[2] += pi/2
        elif name == 'right':
            rvec[2] += -pi/2
            
        position = [FIELD_TAGS[r.tag_id][0] - tvec[0], FIELD_TAGS[r.tag_id][1] - tvec[2]]
        angle = FIELD_TAGS[r.tag_id][2] - rvec[2]
        # </Rotate Code> ^

        posList.append(position)
        rotList.append(angle)

        cv2.putText(image, str(r.tag_id), (icenter[0], icenter[1] - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # show the output image after AprilTag detection
    cv2.imshow(name, image)
    return posList, rotList

def parse_v4l2_devices(output):
    mappings = {}
    lines = output.split('\n')
    current_device = None
    for line in lines:
        if line.strip().endswith(':'):
            current_device = line.strip()[:-1]
        elif '/dev/video' in line:
            video_index = line.strip().split('/')[-1]
            if current_device:
                mappings[current_device[-4:-1]] = int(video_index[-1])
    return mappings

def get_v4l2_device_mapping():
    try:
        output = subprocess.check_output(['v4l2-ctl', '--list-devices'], text=True)
        return parse_v4l2_devices(output)
    except subprocess.CalledProcessError as e:
        print("Error occurred")
        return parse_v4l2_devices(e.output)

# Init cams
cam_mapping = get_v4l2_device_mapping()
front_cap = cv2.VideoCapture(int(cam_mapping["2.1"]))
right_cap = cv2.VideoCapture(int(cam_mapping["2.2"]))

# <Init NetworkTables> v

logging.basicConfig(level=logging.DEBUG)

NetworkTables.initialize()
sd = NetworkTables.getTable("SmartDashboard")

#inst = ntcore.NetworkTableInstance.getDefault()

#table = inst.getTable("datatable")

##Old NetworkTables Code
##wPub = table.getDoubleTopic("w1").publish()
##xPub = table.getDoubleTopic("y1").publish()
##yPub = table.getDoubleTopic("x1").publish()
##rPub = table.getDoubleTopic("r1").publish()

# <Init NetworkTables> ^

# Init plot
plt.ion()
fig, ax = plt.subplots(figsize=(8, 8))

while True:
    try:
        fullPosList, fullRotList = [], []
        posList0, rotList0 = findtags(front_cap, "front")
        posList1, rotList1 = findtags(right_cap, "right")
        fullPosList.extend(posList0); fullPosList.extend(posList1)
        fullRotList.extend(rotList0); fullRotList.extend(rotList1)

        if len(fullPosList) > 0:

            avg_pos = [sum(coord[0] for coord in fullPosList) / len(fullPosList), sum(coord[1] for coord in fullPosList) / len(fullPosList)]
            avg_rot = atan2(sum(sin(angle) for angle in fullRotList) / len(fullRotList), sum(cos(angle) for angle in fullRotList) / len(fullRotList)) % (2 * pi)

            ##Old NetworkTables Code
            ##wPub.set(len(avg_pos), ntcore._now())
            ##xPub.set(avg_pos[0], ntcore._now())
            ##yPub.set(avg_pos[1], ntcore._now())
            ##rPub.set(avg_rot, ntcore._now())

            sd.putNumber("w1", len(avg_pos))
            sd.putNumber("x1", avg_pos[0])
            sd.putNumber("y1", avg_pos[1])
            sd.putNumber("r1", avg_rot)

        # <Draw Code with matplotlib> v
        ax.clear()

        # Plot AprilTag locations
        ax.scatter(FIELD_TAGS_X, FIELD_TAGS_Y, color='b')

        for i in range(len(FIELD_TAGS_X)):
            ax.annotate(FIELD_TAGS_ID[i], (FIELD_TAGS_X[i] + 0.6, FIELD_TAGS_Y[i] - 0.15), textcoords="offset points", xytext=(0, 0), ha='center')

        # Draw Game Field Boundary
        fieldrect = patches.Rectangle((0, -2), 7.04215, 4, linewidth=1, edgecolor='b', facecolor='none')
        ax.add_patch(fieldrect)

        # Adjusting plot limits
        ax.set_xlim(-9, 9)
        ax.set_ylim(-4.5, 4.5)
        ax.set_aspect('equal', adjustable='box')
        ax.set_title('2024 Game Field Positioning Simulation')
        ax.grid(False)

        # Draw robot
        if len(fullPosList) > 0:

            avg_pos = [sum(coord[0] for coord in fullPosList) / len(fullPosList), sum(coord[1] for coord in fullPosList) / len(fullPosList)]
            avg_rot = sum(fullRotList) / len(fullRotList)

            # draw each calculated position of robot, and average of all those.
            for pos in fullPosList:
                plt.plot(pos[0], pos[1], markersize=4, color='green')
            plt.plot(avg_pos[0],avg_pos[1], 'rx')
            # draw line segment showing direction robot is facing.
            end_point = (avg_pos[0] + cos(avg_rot)/2, avg_pos[1] + sin(avg_rot)/2)
            plt.plot([avg_pos[0], end_point[0]], [avg_pos[1], end_point[1]], 'r-')

            # Update plot
            fig.canvas.draw()
            fig.canvas.flush_events()
        # </Draw Code with matplotlib> ^

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    except(KeyboardInterrupt):
        front_cap.release()
        right_cap.release()

        print("/nCams Off. Program ended.")
        cv2.destroyAllWindows()
        break
