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

key = cv2. waitKey(1)

cap0 = cv2.VideoCapture(0)
cap1 = cv2.VideoCapture(1)
cap2 = cv2.VideoCapture(2)
cap3 = cv2.VideoCapture(3)

for cap in [cap0, cap1, cap2, cap3]:
    cap.set(3, 360)
    cap.set(4, 480)
    cap.set(5, 12)

def runCamera(cap, index):
    ret, frame = cap.read()
    if ret:
        tags = at_detector.detect(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))

        for tag in tags:
            for idx in range(len(tag.corners)):
                cv2.line(frame, tuple(tag.corners[idx-1, :].astype(int)), tuple(tag.corners[idx, :].astype(int)), (0, 255, 0))

            cv2.putText(frame, str(tag.tag_id),
                        org=(tag.corners[0, 0].astype(int)+10,tag.corners[0, 1].astype(int)+10),
                        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                        fontScale=0.8,
                        color=(0, 0, 255))
            print("DETECTION!!!   " + str(index))

        cv2.imshow("Camera " + str(index), frame)

while True:
    try:
    
        runCamera(cap0, 0)
        runCamera(cap1, 1)
        runCamera(cap2, 2)
        runCamera(cap3, 3)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        
    except(KeyboardInterrupt):
        cap0.release()
        cap1.release()
        cap2.release()
        cap3.release()

        print("Cams Off. Program ended.")
        cv2.destroyAllWindows()
        break
