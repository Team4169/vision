# This code runs with back and left cams

import apriltag, cv2, subprocess
from math import sin, cos, atan2, pi
import numpy as np
import ntcore
import pickle
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# Initialize Constants (cam_props and field_tags)
cam_props = {}
for cam_name in {"Front", "Back", "Left", "Right"}:
    with open (f"/home/aresuser/vision/calibration/camConfig/camConfig{cam_name}.pkl", 'rb') as f:
        f_data = pickle.load(f)
        cam_props[cam_name] = {'cam_matrix': f_data[0], 'dist': f_data[1], 'offset': f_data[2]}

with open (f"/home/aresuser/vision/apriltags/maps/fieldTagsConfig.pkl", 'rb') as f:
    field_tags = pickle.load(f)

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

        # 0.085725 is for meters, use 3.375 of you want inches. This number is the size of the tags, change it if the tag size changes.
        object_points = np.array([[-0.085725,-0.085725,0],[0.085725,-0.085725,0],[0.085725,0.085725,0],[-0.085725,0.085725,0],[0,0,0]], dtype=np.float32)

        image_points = np.array([ptA,ptB,ptC,ptD,r.center], dtype=np.float32)

        _, rvec, tvec = cv2.solvePnP(object_points, image_points, camera_matrix, distortion_coefficients)

        for i, offset_num in enumerate(origin_offset):
            tvec[i] += offset_num
        # <Rotate Code> v
        # Based on the tag that we see, do math to find out where the robot must be to see that tag.
        c=cos(field_tags[tagId][2]);s=sin(field_tags[tagId][2])
        tvec = [tvec[0]*c - tvec[2]*s, tvec[1], tvec[0]*s + tvec[2]*c]

        rvec[2] += -pi/2
        if name == 'back':
            rvec[2] += pi
        elif name == 'left':
            rvec[2] += pi/2
        elif name == 'right':
            rvec[2] += -pi/2

        position = [field_tags[tagId][0] - tvec[0], field_tags[tagId][1] - tvec[2]]
        angle = field_tags[tagId][2] - rvec[2]
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
back_cap = cv2.VideoCapture(int(cam_mapping["2.1"]))
left_cap = cv2.VideoCapture(int(cam_mapping["2.2"]))

# <Init NetworkTables> v

inst = ntcore.NetworkTableInstance.getDefault()

table = inst.getTable("SmartDashboard")

wPub = table.getDoubleTopic("w1").publish()
xPub = table.getDoubleTopic("x1").publish()
yPub = table.getDoubleTopic("y1").publish()
rPub = table.getDoubleTopic("r1").publish()

# <Init NetworkTables> ^

# Init plot
plt.ion()
fig, ax = plt.subplots(figsize=(8, 8))

while True:
    try:
        fullPosList, fullRotList = [], []
        posList0, rotList0 = findtags(back_cap, "Back")
        posList1, rotList1 = findtags(left_cap, "Left")
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

        # <Draw Code with matplotlib> v
        ax.clear()

        # Plot AprilTag locations
        field_tags_x, field_tags_y, field_tags_r = zip(*field_tags)
        field_tags_id = [i for i in range(len(field_tags))]; field_tags_id[0] = 'x'
        ax.scatter(field_tags_x, field_tags_y, color='b')

        for i in range(len(FIELD_TAGS_X)):
            ax.annotate(field_tags_id[i], (field_tags_x[i] + 0.45, field_tags_y[i] - 0.45), textcoords="offset points", xytext=(0, 0), ha='center')

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
            #THIS CODE IS NOT NEEDED, since the same variables are defined above. delete it and verify.
            avg_pos = [sum(coord[0] for coord in fullPosList) / len(fullPosList), sum(coord[1] for coord in fullPosList) / len(fullPosList)]
            avg_rot = atan2(sum(sin(angle) for angle in fullRotList) / len(fullRotList), sum(cos(angle) for angle in fullRotList) / len(fullRotList)) % (2 * pi)

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

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    except(KeyboardInterrupt):
        back_cap.release()
        left_cap.release()

        print("/nCams Off. Program ended.")
        cv2.destroyAllWindows()
        break
