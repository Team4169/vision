import cv2
import numpy as np
import apriltag

#New code, organised but slower

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
camera_params = [5.72,4.29,1280,720]
tag_size = 0.1524

def matrix_to_euler(rotation_matrix):
    """
    Converts a 3x3 rotation matrix to Euler angles (roll, pitch, yaw).
 
    Parameters:
    - rotation_matrix: 3x3 numpy array representing the rotation matrix.
 
    Returns:
    - euler_angles: Tuple (roll, pitch, yaw) representing the Euler angles in radians.
    """
    sy = np.sqrt(rotation_matrix[0, 0] ** 2 + rotation_matrix[1, 0] ** 2)
 
    # Avoid division by zero
    if sy > 1e-6:
        roll = np.arctan2(rotation_matrix[2, 1], rotation_matrix[2, 2])
        pitch = np.arctan2(-rotation_matrix[2, 0], sy)
        yaw = np.arctan2(rotation_matrix[1, 0], rotation_matrix[0, 0])
    else:
        # Singular case: sy is approximately zero, pitch is near ±90 degrees
        roll = np.arctan2(-rotation_matrix[1, 2], rotation_matrix[1, 1])
        pitch = np.pi / 2.0 if rotation_matrix[2, 0] > 0 else -np.pi / 2.0
        yaw = 0.0
 
    return roll, pitch, yaw

def caploop(cap, name):
    
    tags = findtags(cap, name)
    drawfoundtags(cap, name, tags)
    
def findtags(cap, name):

    detector = apriltag.Detector(options)

    image = cap.read()[1]
    results = detector.detect(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY))

    # loop over the AprilTag detection results
    detected_tags = []
    for r in results:
        # extract the bounding box (x, y)-coordinates for the AprilTag
        # and convert each of the (x, y)-coordinate pairs to integers
        (ptA, ptB, ptC, ptD) = r.corners
        ptB = (int(ptB[0]), int(ptB[1]))
        ptC = (int(ptC[0]), int(ptC[1]))
        ptD = (int(ptD[0]), int(ptD[1]))
        ptA = (int(ptA[0]), int(ptA[1]))
        
        pre_angle = np.degrees(matrix_to_euler(r.homography))
        angle = ["%.1f" % pre_angle[0],"%.1f" % pre_angle[1],"%.1f" % pre_angle[2]]
        detected_tags.append({"corners":[ptA, ptB, ptC, ptD],
                              "center":(int(r.center[0]), int(r.center[1])),
                              "id":str(r.tag_id),
                              "3d":angle})
    
    return detected_tags

def drawfoundtags(cap, name, tags):
    image = cap.read()[1]
    for tag in tags:
        cv2.line(image, tag["corners"][0], tag["corners"][1], (0, 255, 0), 2)
        cv2.line(image, tag["corners"][1], tag["corners"][2], (0, 255, 0), 2)
        cv2.line(image, tag["corners"][2], tag["corners"][3], (0, 255, 0), 2)
        cv2.line(image, tag["corners"][3], tag["corners"][0], (0, 255, 0), 2)

        # draw the center (x, y)-coordinates of the AprilTag
        cv2.circle(image, tag["center"], 5, (0, 0, 255), -1)

        # draw the tag family on the image
        cv2.putText(image, tag["id"], (tag["center"][0], tag["center"][1] - 45),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        print("tag found on cam " + name + ". tag rotation: ", tag["3d"])

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
lastframe=0
while True:
    try:

        caploop(cap0,'0')
        caploop(cap1,'1')
        caploop(cap2,'2')
        caploop(cap3,'3')
        

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
