# This code runs with back and left cams

import apriltag, cv2, subprocess
from math import sin, cos, atan2, pi
import numpy as np
import ntcore
import pickle

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

        # 0.085725 is for meters, use 3.375 of you want inches. This number is the size of the tags, change it if the tag size changes.
        object_points = np.array([[-0.085725,-0.085725,0],[0.085725,-0.085725,0],[0.085725,0.085725,0],[-0.085725,0.085725,0],[0,0,0]], dtype=np.float32)
        image_points = np.array([r.corners[0],r.corners[1],r.corners[2],r.corners[3],r.center], dtype=np.float32)

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


    except(KeyboardInterrupt):
        back_cap.release()
        left_cap.release()

        print("/nCams Off. Program ended.")
        cv2.destroyAllWindows()
        break
