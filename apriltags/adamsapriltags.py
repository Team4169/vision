import apriltag, cv2, ntcore
from math import sin, cos, atan2, pi
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

cam_props = {
'back':{'cam_matrix': np.array([[649.12576246,0,349.34554103],[0,650.06837252,219.01641695],[0,0,1]],dtype=np.float32), 'dist': np.array([0.09962565,-0.92286434,-0.00491307,0.00470977,1.3384658], dtype = np.float32), 'offset':np.array([0,0,0],dtype=np.float32)},
'front':{'cam_matrix': np.array([[677.79747434,0,327.64289497],[0,677.8796147,227.7857478],[0,0,1]],dtype=np.float32), 'dist': np.array([ 8.96320625e-02,-8.66510276e-01,-7.79278879e-04,6.59458019e-03,1.81798852e+00], dtype = np.float32), 'offset':np.array([0,0,0],dtype=np.float32)},
'left':{'cam_matrix': np.array([[667.59437965,0,325.05798259],[0,667.62500105,222.46972227],[0,0,1]],dtype=np.float32), 'dist': np.array([[2.05714549e-01,-1.63695216e+00,1.35826526e-03,-9.93778299e-04,3.32154871e+00]], dtype = np.float32), 'offset':np.array([0,0,0],dtype=np.float32)},
'right':{'cam_matrix': np.array([[675.54311877,0,315.7372509],[0,675.09333584,230.65457206],[0,0,1]],dtype=np.float32), 'dist': np.array([1.56483688e-01,-1.12875978e+00,4.13870402e-03,-1.00809719e-03,1.65813324e+00], dtype = np.float32), 'offset':np.array([0,0,0],dtype=np.float32)}}

# <Assign correct params to correct cams> v
camidnames = ['x','x']

# Enviornment Initialization
# I used this, but the fmap could work as well.
FIELD_TAGS = [[0, 0, 0], [6.808597, -3.859403, (120+90)*pi/180], [7.914259, -3.221609, (120+90)*pi/180], [8.308467, 0.877443, (180+90)*pi/180], [8.308467, 1.442593, (180+90)*pi/180], [6.429883, 4.098925, (270+90)*pi/180 - 2*pi], [-6.429375, 4.098925, (270+90)*pi/180], [-8.308975, 1.442593, (0+90)*pi/180], [-8.308975, 0.877443, (0+90)*pi/180], [-7.914767, -3.221609, (60+90)*pi/180], [-6.809359, -3.859403, (60+90)*pi/180], [3.633851, -0.392049, (300+90)*pi/180], [3.633851, 0.393065, (60+90)*pi/180], [2.949321, -0.000127, (180+90)*pi/180], [-2.950083, -0.000127, (0+90)*pi/180], [-3.629533, 0.393065, (120+90)*pi/180], [-3.629533, -0.392049, (240+90)*pi/180]]


FIELD_TAGS_X = [FIELD_TAG[0] for FIELD_TAG in FIELD_TAGS]
FIELD_TAGS_Y = [FIELD_TAG[1] for FIELD_TAG in FIELD_TAGS]
FIELD_TAGS_ID = [i for i in range(len(FIELD_TAGS))]; FIELD_TAGS_ID[0] = 'x'

# </Assign correct params to correct cams> v
def IDCams(caps):
    while True:
        for i in range(len(caps)):
            cap = cv2.VideoCapture(i)
            image = cap.read()[1]
            cv2.imshow('cap ' + str(i), image)
            cv2.waitKey(200)
            camname=input('which cam is this? (l)eft,(r)ight,(f)ront,(b)ack: ')
            if camname == 'l':
                camname = 'left'
            elif camname == 'r':
                camname = 'right'
            elif camname == 'f':
                camname = 'front'
            elif camname == 'b':
                camname = 'back'
            else:
                print('Error. Use l, r, f, or b.')
                break
            cap.release()
            camidnames[i] = camname
            cv2.destroyAllWindows()
        areWeDone = input('Are we done? (y or n): ').lower()
        if areWeDone == 'y': break
        else: pass
# </Assign correct params to correct cams> ^

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
    camera_matrix = cam_props[camidnames[int(name)]]['cam_matrix']
    distortion_coefficients = cam_props[camidnames[int(name)]]['dist']
    origin_offset = cam_props[camidnames[int(name)]]['offset']
    # </get params> ^
    
    
    # <undistort> v
    h,  w = image.shape[:2]
    new_camera_matrix, roi = cv2.getOptimalNewCameraMatrix(camera_matrix, distortion_coefficients, (w,h), 1, (w,h))

    mapx, mapy = cv2.initUndistortRectifyMap(camera_matrix, distortion_coefficients, None, new_camera_matrix, (w,h), 5)
    dst = cv2.remap(image, mapx, mapy, cv2.INTER_LINEAR)
    # crop the image
    
    x, y, w, h = roi
    dst = dst[y:y+h, x:x+w]
    image = dst
    # </undistort> ^
    
    results = detector.detect(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY))

    # loop over the AprilTag detection results
    posList = []
    rotList = []
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
        cv2.line(image, iptA, iptB, (0, 255, 0), 2)
        cv2.line(image, iptB, iptC, (0, 255, 0), 2)
        cv2.line(image, iptD, iptA, (0, 255, 0), 2)
        
        # draw the center (x, y)-coordinates of the AprilTag
        icenter = (int(r.center[0]), int(r.center[1]))
        cv2.circle(image, icenter, 5, (0, 0, 255), -1)

        U = 0.0857 #In meters, use 3.375 of you want inches.

        object_points = np.array([[-U,-U,0],[U,-U,0],[U,U,0],[-U,U,0],[0,0,0]], dtype=np.float32)
        
        image_points = np.array([ptA,ptB,ptC,ptD,r.center], dtype=np.float32)
        
        
        _, rvec, tvec = cv2.solvePnP(object_points, image_points, camera_matrix, distortion_coefficients)


        for i, offset_num in enumerate(origin_offset):
            tvec[i] -= offset_num
        
        c=cos(FIELD_TAGS[r.tag_id][2]);s=sin(FIELD_TAGS[r.tag_id][2])
        tvec = [tvec[0]*c - tvec[2]*s, tvec[1], tvec[0]*s + tvec[2]*c]

        rvec[2] += -pi/2
        if camidnames[int(name)] == 'back':
            rvec[2] += pi
        elif camidnames[int(name)] == 'left':
            rvec[2] += pi/2
        elif camidnames[int(name)] == 'right':
            rvec[2] += -pi/2
        position = [FIELD_TAGS[r.tag_id][0] - tvec[0], FIELD_TAGS[r.tag_id][1] - tvec[2]]

        angle = FIELD_TAGS[r.tag_id][2] - rvec[2]
        

        c=cos(angle);s=sin(angle)
        tvec = [tvec[0]*c - tvec[2]*s, tvec[1], tvec[0]*s + tvec[2]*c]
        # <Magic Rotate Code> ^
        
        posList.append(position)
        rotList.append(angle)


        cv2.putText(image, str(r.tag_id), (icenter[0], icenter[1] - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    # show the output image after AprilTag detection
    cv2.imshow(name, image)
    return posList, rotList


IDCams([0,1])

# Init cams
cap0 = cv2.VideoCapture(0)
cap1 = cv2.VideoCapture(1)
#cap2 = cv2.VideoCapture(2)
#cap3 = cv2.VideoCapture(3)
all_caps = [cap0, cap1]
#scalefac = 1# Max range = 13ft * scalefac
#for capn in all_caps:
#    capn.set(3, 480 * scalefac)
#    capn.set(4, 640 * scalefac)
#    capn.set(5, 12) #fps


# <Init NetworkTables> v
inst = ntcore.NetworkTableInstance.getDefault()

table = inst.getTable("datatable")

wPub = table.getDoubleTopic("w1").publish()
xPub = table.getDoubleTopic("y1").publish()
yPub = table.getDoubleTopic("x1").publish()
rPub = table.getDoubleTopic("r1").publish()
# <Init NetworkTables> ^

# Init plot
plt.ion()
fig, ax = plt.subplots(figsize=(8, 8))

while True:
    try:
    
        fullPosList, fullRotList = [], []
        posList0, rotList0 = findtags(cap0,'0')
        posList1, rotList1 = findtags(cap1,'1')
        fullPosList.extend(posList0); fullPosList.extend(posList1)
        fullRotList.extend(rotList0); fullRotList.extend(rotList1)
        
        if len(fullPosList) > 0:
            
            avg_pos = [sum(coord[0] for coord in fullPosList) / len(fullPosList), sum(coord[1] for coord in fullPosList) / len(fullPosList)]
            avg_rot = math.atan2(sum(math.sin(angle) for angle in fullRotList) / len(fullRotList), sum(math.cos(angle) for angle in fullRotList) / len(fullRotList)) % (2 * math.pi)
            
            wPub.set(len(avg_pos), ntcore._now())
            xPub.set(avg_pos[0], ntcore._now())
            yPub.set(avg_pos[1], ntcore._now())
            rPub.set(avg_rot, ntcore._now())
        
        # <Draw Code with matplotlib> v
        if True: #DRAW
            # Clear previous robots from the plot
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
                
                # draw line segment
                start_point = (avg_pos[0], avg_pos[1])
                end_point = (avg_pos[0] + cos(avg_rot)/2, avg_pos[1] + sin(avg_rot)/2)
                plt.plot(start_point[0],start_point[1], 'rx')
                plt.plot([start_point[0], end_point[0]], [start_point[1], end_point[1]], 'r-')

                # Update plot
                fig.canvas.draw()
                fig.canvas.flush_events()
        # </Draw Code with matplotlib> ^
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        
    except(KeyboardInterrupt):
        cap0.release()
        cap1.release()
        # cap2.release()
        # cap3.release()

        print("/nCams Off. Program ended.")
        cv2.destroyAllWindows()
        break
