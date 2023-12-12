import apriltag
import cv2

def findtags(cap, name):

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
    detector = apriltag.Detector(options)
    
    image = cap.read()[1]
    results = detector.detect(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY))

    # loop over the AprilTag detection results
    for r in results:
	    # extract the bounding box (x, y)-coordinates for the AprilTag
	    # and convert each of the (x, y)-coordinate pairs to integers
	    (ptA, ptB, ptC, ptD) = r.corners
	    ptB = (int(ptB[0]), int(ptB[1]))
	    ptC = (int(ptC[0]), int(ptC[1]))
	    ptD = (int(ptD[0]), int(ptD[1]))
	    ptA = (int(ptA[0]), int(ptA[1]))
	    
	    # draw the bounding box of the AprilTag detection
	    cv2.line(image, ptA, ptB, (0, 255, 0), 2)
	    cv2.line(image, ptB, ptC, (0, 255, 0), 2)
	    cv2.line(image, ptC, ptD, (0, 255, 0), 2)
	    cv2.line(image, ptD, ptA, (0, 255, 0), 2)
	    
	    # draw the center (x, y)-coordinates of the AprilTag
	    (cX, cY) = (int(r.center[0]), int(r.center[1]))
	    cv2.circle(image, (cX, cY), 5, (0, 0, 255), -1)
	    
	    # draw the tag family on the image
	    cv2.putText(image, str(r.tag_id), (ptA[0], ptA[1] - 15),
		    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
	    #print("Just found: {}".format(str(r.tag_id)))
	    
    # show the output image after AprilTag detection
    cv2.imshow(name, image)

cap0 = cv2.VideoCapture(0)
cap1 = cv2.VideoCapture(1)
cap2 = cv2.VideoCapture(2)
cap3 = cv2.VideoCapture(3)

for cap in [cap0, cap1, cap2, cap3]:
    cap.set(3, 360)
    cap.set(4, 480)
    cap.set(5, 12)
    
while True:
    try:
    
        findtags(cap0,'0')
        findtags(cap1,'1')
        findtags(cap2,'2')
        findtags(cap3,'3')

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        
    except(KeyboardInterrupt):
        cap0.release()
        cap1.release()
        cap2.release()
        cap3.release()

        print("/nCams Off. Program ended.")
        cv2.destroyAllWindows()
        break
