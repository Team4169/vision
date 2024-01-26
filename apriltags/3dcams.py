import apriltag, cv2
import numpy as np
#this cammatrix and distortion coff. is for cam[1] (2nd), (on right)
camera_matrix = np.array([[669.37858609,0,325.4807048],[0,667.33437347,211.64211242],[0,0,1]], dtype=np.float32)
        
distortion_coefficients = np.array([ 1.73661285e-01,-1.29981726e+00,-1.77963313e-03,-1.91796294e-03,2.22693881e+00], dtype=np.float32)

options = apriltag.DetectorOptions(families='tag16h5',
                                   border=2,
                                   nthreads=4,
                                   quad_decimate=1.0,
                                   quad_blur=0.0,
                                   refine_edges=True,
                                   refine_decode=False,
                                   refine_pose=False,
                                   debug=True,
                                   quad_contours=True)
def findtags(cap, name):
    
    detector = apriltag.Detector(options) 
    image = cap.read()[1]
    
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
    for r in results:
        if r.tag_id != 0:
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
        cv2.line(image, iptC, iptD, (0, 255, 0), 2)
        cv2.line(image, iptD, iptA, (0, 255, 0), 2)
        
        # draw the center (x, y)-coordinates of the AprilTag
        icenter = (int(r.center[0]), int(r.center[1]))
        cv2.circle(image, icenter, 5, (0, 0, 255), -1)
        u = 3 * scalefac
        object_points = np.array([[-u,-u,0],[u,-u,0],[u,u,0],[-u,u,0],[0,0,0]], dtype=np.float32)
        
        image_points = np.array([ptA,ptB,ptC,ptD,r.center], dtype=np.float32)
        
        print("-------------\n"*5)
        
        _, rvec, tvec = cv2.solvePnP(object_points, image_points, camera_matrix, distortion_coefficients)
        
        if tvec is not None:
            print("tag found on cam", name, "\nrvec:",rvec,"\ntvec:",tvec,"\npythag dist:",(float(tvec[0])**2+float(tvec[1])**2+float(tvec[2])**2)**.5)
        else:
            print("bad tvec")
        cv2.putText(image, str(r.tag_id), (icenter[0], icenter[1] - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        #print("Just found: {}".format(str(r.tag_id)))
        
    # show the output image after AprilTag detection
    cv2.imshow(name, image)

#cap0 = cv2.VideoCapture(0)
cap1 = cv2.VideoCapture(1)
#cap2 = cv2.VideoCapture(2)
#cap3 = cv2.VideoCapture(3)
scalefac = 1 # Max range = 10ft * scalefac
for cap in [cap1]:
    cap.set(3, 480 * scalefac)
    cap.set(4, 640 * scalefac)
    cap.set(5, 12) #fps

lastframe=0
while True:
    try:
    
#        findtags(cap0,'0')
        findtags(cap1,'1')
#        findtags(cap2,'2')
#        findtags(cap3,'3')

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        
    except(KeyboardInterrupt):
#        cap0.release()
        cap1.release()
#        cap2.release()
#        cap3.release()

        print("/nCams Off. Program ended.")
        cv2.destroyAllWindows()
        break
