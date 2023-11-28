from dt_apriltags import Detector
import numpy as np
import os, cv2 

at_detector = Detector(families='tag16h5',
                       nthreads=1,
                       quad_decimate=4.0,
                       quad_sigma=0,
                       refine_edges=1,
                       decode_sharpening=0,
                       debug=0)

camera_params = [5.72,4.29,1280,720]
tag_size = 0.1524

key = cv2. waitKey(1)

cap1 = cv2.VideoCapture(0)
cap2 = cv2.VideoCapture(1)

DEBUGA = ''
DEBUGB = True

def runCamera(cap, index):
    global DEBUGA
    global DEBUGB
    ret, frame = cap.read()
    if ret:
        tags = at_detector.detect(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), estimate_tag_pose=True, camera_params=camera_params, tag_size=tag_size)
    
        for tag in tags:
            for idx in range(len(tag.corners)):
                cv2.line(frame, tuple(tag.corners[idx-1, :].astype(int)), tuple(tag.corners[idx, :].astype(int)), (0, 255, 0))

            cv2.putText(frame, str(tag.tag_id),
                        org=(tag.corners[0, 0].astype(int)+10,tag.corners[0, 1].astype(int)+10),
                        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                        fontScale=0.8,
                        color=(0, 0, 255))
            if (DEBUGB):
                DEBUGB = False
                DEBUGA = tag.corners
            print(DEBUGA)
            
        cv2.imshow("Camera " + str(index), frame)

while True:
    try:
        runCamera(cap1, 1)
        runCamera(cap2, 2)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        
    except(KeyboardInterrupt):
        print("Turning off camera.")
        cap1.release()
        cap2.release()
        print("Camera off.")
        print("Program ended.")
        cv2.destroyAllWindows()
        break
