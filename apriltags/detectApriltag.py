from dt_apriltags import Detector
import numpy as np
import os, cv2

at_detector = Detector(families='tag16h5',
                       nthreads=1,
                       quad_decimate=1.0,
                       quad_sigma=0.0,
                       refine_edges=1,
                       decode_sharpening=0.25,
                       debug=0)

key = cv2. waitKey(1)
webcam = cv2.VideoCapture(0)
while True:
    try:
        check, frame = webcam.read()
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) 
        tags = at_detector.detect(frame)
        frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR) 

        for tag in tags:
            for idx in range(len(tag.corners)):
                cv2.line(frame, tuple(tag.corners[idx-1, :].astype(int)), tuple(tag.corners[idx, :].astype(int)), (0, 255, 0))

            cv2.putText(frame, str(tag.tag_id),
                        org=(tag.corners[0, 0].astype(int)+10,tag.corners[0, 1].astype(int)+10),
                        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                        fontScale=0.8,
                        color=(0, 0, 255))

        cv2.imshow("Capturing", frame)
        key = cv2.waitKey(1)
        
    except(KeyboardInterrupt):
        print("Turning off camera.")
        webcam.release()
        print("Camera off.")
        print("Program ended.")
        cv2.destroyAllWindows()
        break