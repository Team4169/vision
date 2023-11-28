from dt_apriltags import Detector
import numpy as np
import os, cv2

def update_history(history, value):
    history.append(value)
    if len(history) > 25: # <-- History Length
        history.pop(0)

def is_real_detection(history, current_value):
    if len(history) < 20:
        return False  # Not enough history yet, consider it a false positive to be safe

        # Check if the current value is similar to the average value in the history
        average_position = [sum(x)/len(x) for x in zip(*[y for y in a])]
        return abs([sum(x)/len(x) for x in zip(*[y for y in a])]current_position - average_position) < self.position_threshold

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
cap2 = cv2.VideoCapture(2)

aprilTagTracker = ObjectTracker()

def rot_matrix_to_euler(R):
    yaw = np.arctan2(R[1][0],R[0][0])
    pitch = np.arctan2(-R[2,0],(R[2][1]**2+R[2][2]**2)**.5)
    roll = np.arctan2(R[2][1],R[2][2])
    return(yaw, pitch, roll)

def runCamera(cap, index):
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
            detectd_value = {"pos": [POSITION], "rot":rot_matrix_to_euler(tag.pose_R), "id":tag.tag_id}
            update_history(detected_position)
            if tracker.is_consistent_detection(detected_position):
                # Update the tracker with the current position if it's consistent
                tracker.update_history(detected_position)
                # Process the valid detections
            print()
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
