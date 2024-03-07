import apriltag, cv2, json
import numpy as np
import json

cam_props = {
'back':{'cam_matrix': np.array([[659.5522522254913, 0.0, 342.14593411596394], [0.0, 660.0855257028237, 233.07985632799412], [0.0, 0.0, 1.0]],dtype=np.float32), 'dist': np.array([[0.18218171362352173, -1.3943575501329653, -0.0034890991822150033, -0.003111058479543986, 2.4948141925852796]], dtype = np.float32), 'offset':np.array([5.5,17,-0.5],dtype=np.float32)},
'front':{'cam_matrix': np.array([[650.6665701168481, 0.0, 308.11247568203765], [0.0, 649.267759423238, 230.2397074540069], [0.0, 0.0, 1.0]],dtype=np.float32), 'dist': np.array([[0.142925049930884, -1.1502926269495592, -0.0019150557540761415, -0.00328202292619461, 1.8141065950524837]], dtype = np.float32), 'offset':np.array([-7,21,.5],dtype=np.float32)},
'left':{'cam_matrix': np.array([[668.2138474014353, 0.0, 332.83301545896086], [0.0, 666.4860881212383, 214.33779667521517], [0.0, 0.0, 1.0]],dtype=np.float32), 'dist': np.array([[0.22224705297101408, -1.7549821808892665, -0.005523738126667523, 0.0051301529546101616, 3.4133532108023994]], dtype = np.float32), 'offset':np.array([8,19,2],dtype=np.float32)},
'right':{'cam_matrix': np.array([[660.6703723058181, 0.0, 321.2980455248988], [0.0, 658.6516133373474, 218.49261248405028], [0.0, 0.0, 1.0]],dtype=np.float32), 'dist': np.array([[0.18802634354539693, -1.5669527368643557, -0.0006972309753818612, -0.0018548904430247361, 3.04483663171066]], dtype = np.float32)}}

# <Assign correct params to correct cams> v
camidnames = ['x']

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
    #origin_offset = cam_props[camidnames[int(name)]]['offset']
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

        U = 3.375 # meters, use 3.375 for inches

        object_points = np.array([[-U,-U,0],[U,-U,0],[U,U,0],[-U,U,0],[0,0,0]], dtype=np.float32)
        
        image_points = np.array([ptA,ptB,ptC,ptD,r.center], dtype=np.float32)
        
        
        _, rvec, tvec = cv2.solvePnP(object_points, image_points, camera_matrix, distortion_coefficients)
        
        # <rotate> v
        #if camidnames[int(name)] == 'back':
        #    tvec = np.array([-tvec[0],tvec[1],-tvec[2]], dtype=np.float32)
        #elif camidnames[int(name)] == 'left':
        #    tvec = np.array([-tvec[2],tvec[1],tvec[0]], dtype=np.float32)
        #elif camidnames[int(name)] == 'right':
        #    tvec = np.array([tvec[2],tvec[1],-tvec[0]], dtype=np.float32)
        # </rotate> ^
        # Dont calculate Field Position, just print tvec&rvec
        
        print("rvec: " + str(rvec), 'tvec: ' + str(tvec))
        
        
        
       # if r.tag_id in coordinates:
        #    posList.append(
         #       ([coordinates[r.tag_id][0] - tvec[0], coordinates[r.tag_id][1] - tvec[2]])
        #    )

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
#cap1 = cv2.VideoCapture(1)
all_caps = [cap0]
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
        #fullPosList.extend(findtags(cap1,'1'))
        totalPos=[0,0]
        
        # On roborio, weight average of fullPosList(jetson1) and fullPosList(jetson2) based on len(fullPosList)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        
    except(KeyboardInterrupt):
        cap0.release()
        cap1.release()
        
        print("/nCams Off. Program ended.")
        cv2.destroyAllWindows()
        break
