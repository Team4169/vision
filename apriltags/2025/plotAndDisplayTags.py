import apriltag, cv2, subprocess
from math import sin, cos, atan2, pi
import numpy as np
import pickle
from time import time
from dotenv import load_dotenv
from os import getenv
import sys
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# <Intialize enviornment variables and options> v
load_dotenv()

ENABLE_NETWORK_TABLES = int(getenv("ENABLE_NETWORK_TABLES"))
NEW_OS = int(getenv("NEW_OS"))
JETSON_ID = getenv("JETSON_ID")

SHOW_FPS = True
PRINT_FOUND_TAGS = True
# </Intialize enviornment variables and options> ^

if ENABLE_NETWORK_TABLES:
    import ntcore

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
        if tagId not in range(1,23): # Competition only uses tags 1 to 23, if another one is found ignore it.
            continue
        seen_tag = field_tags[tagId]

        # extract the bounding box (x, y)-coordinates for the AprilTag, and convert each of the (x, y)-coordinate pairs to integers
        (ptA, ptB, ptC, ptD) = r.corners
        iptA = (int(ptA[0]), int(ptA[1]))
        iptB = (int(ptB[0]), int(ptB[1]))
        iptC = (int(ptC[0]), int(ptC[1]))
        iptD = (int(ptD[0]), int(ptD[1]))
        icenter = (int(r.center[0]), int(r.center[1]))

        # draw the bounding box of the AprilTag detection, and a circle at the center of the tag.
        cv2.line(image, iptA, iptB, (255, 0, 0), 5)
        cv2.line(image, iptB, iptC, (255, 0, 0), 5)
        cv2.line(image, iptC, iptD, (255, 0, 0), 5)
        cv2.line(image, iptD, iptA, (255, 0, 0), 5)
        cv2.circle(image, icenter, 10, (0, 0, 255), -1)

        tag_size = .08255 # Distance in meters from the middle of the tag to the end of the black part of the tag. Also equal to half the side length of the black part of the april tag. Change it if the tag size changes. 
        object_points = np.array([[-tag_size, -tag_size, 0], [tag_size, -tag_size, 0], [tag_size, tag_size, 0], [-tag_size, tag_size, 0], [0, 0, 0]], dtype=np.float32)

        image_points = np.array([r.corners[0], r.corners[1], r.corners[2], r.corners[3], r.center], dtype=np.float32)

        _, rvec, tvec = cv2.solvePnP(object_points, image_points, camera_matrix, distortion_coefficients)
        for i, offset_num in enumerate(origin_offset):
            tvec[i] += offset_num

        # <Rotate Code> v
        # Based on the tag that we see, and which camera sees it, do math to find out where the robot must be to see that tag in that relative position and orientation.
        c=cos(seen_tag['Z-Rotation'] + rvec[1] + pi/2);s=sin(seen_tag['Z-Rotation'] + rvec[1] + pi/2)
        tvec = [tvec[0]*c - tvec[2]*s, tvec[1], tvec[0]*s + tvec[2]*c]
        
        #if name == 'Front':
        #    rvec[1] += 0
        #el
        if name == 'Back':
            rvec[1] += pi
        elif name == 'Right':
            rvec[1] += pi/2
        elif name == 'Left':
            rvec[1] += -pi/2

        position = [seen_tag['X'] - tvec[0], seen_tag['Y'] - tvec[2]]
        angle = float((seen_tag['Z-Rotation'] + rvec[1] - pi)[0])
        # </Rotate Code> ^
        
        posList.append(position)
        rotList.append(angle)

        cv2.putText(image, 'id: ' + str(tagId), (icenter[0], icenter[1] - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # show the output image after AprilTag detection
    cv2.putText(image, name, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
    cv2.imshow(name, image)
    return posList, rotList

def parse_v4l2_devices(output):
    mappings = {}
    if NEW_OS:
        lines = str(output).split('\\n\\n')
        for line in lines:
            if 'usb-70090000.xusb-' in line and '/dev/video' in line:
                mappings[line.split('usb-70090000.xusb-')[1][0:3]] = int(line.split('/dev/video')[1])
    else:
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

def get_v4l2_devices():
    try:
        if NEW_OS:
            output = str(subprocess.check_output(['v4l2-ctl', '--list-devices']))[2:-1]
        else:
            output = subprocess.check_output(['v4l2-ctl', '--list-devices'], text=True)
        return parse_v4l2_devices(output)
    except subprocess.CalledProcessError as e:
        print("Error occurred")
        return parse_v4l2_devices(e.output)

cam_0_name = ''
cam_1_name = ''
# Initialize cams with correct Jetson
if JETSON_ID == '1':
    cam_0_name = 'Front'
    cam_1_name = 'Right'
elif JETSON_ID == '2':
    cam_0_name = 'Back'
    cam_1_name = 'Left'
else:
    raise Exception('Invalid Jetson ID')

cam_mapping = get_v4l2_devices()

cam_0 = cv2.VideoCapture(int(cam_mapping["2.1"]))
cam_1 = cv2.VideoCapture(int(cam_mapping["2.2"]))

# <Init Constants> v (cam_props and field_tags)
cam_props = {}

for cam_name in {cam_0_name, cam_1_name}:
    with open (f"{'/home/robotics4169/vision' if NEW_OS else '/home/aresuser/vision'}/calibration/camConfig/camConfig{cam_name}.pkl", 'rb') as f:
        f_data = pickle.load(f)
        cam_props[cam_name] = {'cam_matrix': f_data[0], 'dist': f_data[1], 'offset': f_data[2]}

with open (f"{'/home/robotics4169/vision' if NEW_OS else '/home/aresuser/vision'}/apriltags/maps/fieldTagsConfig_2025.pkl", 'rb') as f:
    field_tags = pickle.load(f)
    # 'field_tags' is a list of dictionaries. It contains the data for the apriltags on the 2025 field (update for future seasons).
    # It looks looks like: [{'ID': 1, 'X': 16.7, 'Y': 0.655, 'Z': 1.49, 'Z-Rotation': 2.20, 'Y-Rotation': 0.0} ...]
# <Init Constants> ^

# <Init NetworkTables> v
if ENABLE_NETWORK_TABLES:
    inst = ntcore.NetworkTableInstance.getDefault()
    table = inst.getTable("SmartDashboard")

    wPub = table.getDoubleTopic(f"w{JETSON_ID}").publish()
    xPub = table.getDoubleTopic(f"x{JETSON_ID}").publish()
    yPub = table.getDoubleTopic(f"y{JETSON_ID}").publish()
    rPub = table.getDoubleTopic(f"r{JETSON_ID}").publish()

    inst.startClient4(f"Cameras from Jetson {JETSON_ID}")
    inst.setServerTeam(4169)
    inst.startDSClient()
# <Init NetworkTables> ^

# <Init Plot> v
plt.ion()
fig, ax = plt.subplots(figsize=(8, 8))
# </Init Plot ^

if SHOW_FPS:
    start_time = time()
    frame_count = 0

while True: # Periodic code
    if SHOW_FPS:
        print('FPS:',frame_count/(time()-start_time))
        frame_count += 1
        if time()-start_time > 5:
            start_time = time()
            frame_count = 1

    posList0, rotList0 = findtags(cam_0, cam_0_name)
    posList1, rotList1 = findtags(cam_1, cam_1_name)
    fullPosList = posList0 + posList1
    fullRotList = rotList0 + rotList1

    if len(fullPosList) > 0:
        avg_pos = [sum(coord[0] for coord in fullPosList) / len(fullPosList), sum(coord[1] for coord in fullPosList) / len(fullPosList)]
        avg_rot = atan2(sum(sin(angle) for angle in fullRotList) / len(fullRotList), sum(cos(angle) for angle in fullRotList) / len(fullRotList)) % (2 * pi)

        # Set Network table values (Weight, Xposition, Yposition, and Rotation)
        if ENABLE_NETWORK_TABLES:
            wPub.set(len(fullPosList))
            xPub.set(avg_pos[0])
            yPub.set(avg_pos[1])
            rPub.set(avg_rot)
        if PRINT_FOUND_TAGS:
            print(f"w: {len(fullPosList)}\nx: {avg_pos[0]}\ny: {avg_pos[1]}\nr: {avg_rot}\n")

    # <Draw Code with matplotlib> v
    ax.clear()

    # Plot AprilTag locations
    field_tags_x, field_tags_y, field_tags_r, field_tags_id = [], [], [], []
    for tag in field_tags:
        field_tags_x.append(tag['X'])
        field_tags_y.append(tag['Y'])
        field_tags_r.append(tag['Z-Rotation'])
        field_tags_id.append(tag['ID'])
    field_tags_id[0] = 'origin'
    ax.scatter(field_tags_x, field_tags_y, color='b')

    for i in range(len(field_tags_x)):
        ax.annotate(field_tags_id[i], (field_tags_x[i] + 0.45, field_tags_y[i] - 0.45), textcoords="offset points", xytext=(0, 0), ha='center')

    # Draw Game Field Boundary
    fieldrect = patches.Rectangle((0, 0), 17.548352, 8.0518, linewidth=1, edgecolor='b', facecolor='none')
    ax.add_patch(fieldrect)

    # Adjusting plot limits
    ax.set_xlim(-1, 18.548352)
    ax.set_ylim(-1, 9.0518)
    ax.set_aspect('equal', adjustable='box')
    ax.set_title('2025 Game Field Positioning Simulation')
    ax.grid(False)

    # Draw robot
    if len(fullPosList) > 0:
        # draw each calculated position of robot, and average of all those.
        for pos in fullPosList:
            plt.plot(pos[0], pos[1], 'go', markersize=3)
        plt.plot(avg_pos[0],avg_pos[1], 'bo', markersize=10)
        # draw line segment showing direction robot is facing.
        end_point = (avg_pos[0] + cos(avg_rot)/2, avg_pos[1] + sin(avg_rot)/2)
        plt.plot([avg_pos[0], end_point[0]], [avg_pos[1], end_point[1]], 'r-')

        # Update plot
    fig.canvas.draw()
    fig.canvas.flush_events()
    # </Draw Code with matplotlib> ^

    cv2.waitKey(100)