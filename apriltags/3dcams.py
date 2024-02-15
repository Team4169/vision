import apriltag, cv2, json
import numpy as np
import json

cam_props = {
'back':{'cam_matrix': np.array([[649.12576246,0,349.34554103],[0,650.06837252,219.01641695],[0,0,1]],dtype=np.float32), 'dist': np.array([0.09962565,-0.92286434,-0.00491307,0.00470977,1.3384658], dtype = np.float32), 'offset':np.array([5.5,17,-0.5],dtype=np.float32)},
'front':{'cam_matrix': np.array([[677.79747434,0,327.64289497],[0,677.8796147,227.7857478],[0,0,1]],dtype=np.float32), 'dist': np.array([ 8.96320625e-02,-8.66510276e-01,-7.79278879e-04,6.59458019e-03,1.81798852e+00], dtype = np.float32), 'offset':np.array([-7,21,.5],dtype=np.float32)},
'left':{'cam_matrix': np.array([[667.59437965,0,325.05798259],[0,667.62500105,222.46972227],[0,0,1]],dtype=np.float32), 'dist': np.array([[2.05714549e-01,-1.63695216e+00,1.35826526e-03,-9.93778299e-04,3.32154871e+00]], dtype = np.float32), 'offset':np.array([8,19,2],dtype=np.float32)},
'right':{'cam_matrix': np.array([[675.54311877,0,315.7372509],[0,675.09333584,230.65457206],[0,0,1]],dtype=np.float32), 'dist': np.array([1.56483688e-01,-1.12875978e+00,4.13870402e-03,-1.00809719e-03,1.65813324e+00], dtype = np.float32), 'offset':np.array([-7,20.5,-1.5],dtype=np.float32)}}

# <Assign correct params to correct cams> v
camidnames = ['x','x']

# Get Apritag locations
file_path = "/home/jetson/vision/apriltags/maps/computerLab.fmap"
with open(file_path, 'r') as file:
    data = json.load(file)
coordinates = {}
for apriltag_ in data["fiducials"]:
    coordinates[apriltag_["id"]] = [apriltag_["transform"][3], apriltag_["transform"][7]]

def IDCams():
    for i in range(len(all_caps)):
        camidnames[i] = 'front'
        image = all_caps[i].read()[1]
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
            raise ValueError('Error. Use l, r, f, or b.')
        camidnames[i] = camname
        cv2.destroyAllWindows()
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
    
    #image = cv2.resize(image, (640,480)) ONLY USE IF using capn.set()
    
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
    for r in results:

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

        U = .085725 # meters, use 3.375 for inches

        object_points = np.array([[-U,-U,0],[U,-U,0],[U,U,0],[-U,U,0],[0,0,0]], dtype=np.float32)
        
        image_points = np.array([ptA,ptB,ptC,ptD,r.center], dtype=np.float32)
        
        
        _, rvec, tvec = cv2.solvePnP(object_points, image_points, camera_matrix, distortion_coefficients)
        
        # <rotate> v
        if camidnames[int(name)] == 'back':
            tvec = np.array([-tvec[0],tvec[1],-tvec[2]], dtype=np.float32)
        elif camidnames[int(name)] == 'left':
            tvec = np.array([-tvec[2],tvec[1],tvec[0]], dtype=np.float32)
        elif camidnames[int(name)] == 'right':
            tvec = np.array([tvec[2],tvec[1],-tvec[0]], dtype=np.float32)
        # </rotate> ^
        
        for i, offset_num in enumerate(origin_offset):
            tvec[i] -= offset_num

        
        if (r.tag_id in coordinates):
            posList.append(
                ([coordinates[r.tag_id][0] - tvec[0], coordinates[r.tag_id][1] - tvec[2]])
            )

        '''
        if tvec is not None:
            print("tag found on cam", name, "\nrvec:",rvec,"\ntvec:",tvec,"\npythag dist:",(float(tvec[0])**2+float(tvec[1])**2+float(tvec[2])**2)**.5)
        else:
            print("bad tvec")
        '''
        cv2.putText(image, str(r.tag_id), (icenter[0], icenter[1] - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    # show the output image after AprilTag detection
    cv2.imshow(name, image)
    return posList

cap0 = cv2.VideoCapture(0)
cap1 = cv2.VideoCapture(1)
all_caps = [cap0, cap1]
#scalefac = 0.5
#for capn in all_caps:
#    capn.set(3, 480 * scalefac)
#    capn.set(4, 640 * scalefac)
#    capn.set(5, 2) #fps


IDCams()

while True:
    try:
        fullPosList = []
        fullPosList.extend(findtags(cap0,'0'))
        fullPosList.extend(findtags(cap1,'1'))
        totalPos=[0,0]
        weight=0
        for i in fullPosList:
            if len(i) == 0: continue
            weight+=1
            totalPos[0]+=i[0][0]
            totalPos[1]+=i[1][0]
        if weight > 0:        
            totalPos[0] /= weight
            totalPos[1] /= weight
        else:
            totalPos[0] = 0
            totalPos[1] = 0
        dataToSend = {'weight':weight, 'posx':totalPos[0], 'posy':totalPos[1]}
        print(json.dumps(dataToSend))
        
        # On roborio, weight average of fullPosList(jetson1) and fullPosList(jetson2) based on len(fullPosList)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        
    except(KeyboardInterrupt):
        cap0.release()
        cap1.release()
        
        print("/nCams Off. Program ended.")
        cv2.destroyAllWindows()
        break
