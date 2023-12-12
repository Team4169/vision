import apriltag
import numpy as np
import os, cv2

def detect_tags(image):
    print("Begin looking for tags")
    options = apriltag.DetectorOptions(families="tag36h11")
    detector = apriltag.Detector(options)
    results = detector.detect(image)
    print("[INFO] {} total AprilTags detected".format(results))
    
    #Draw found april tags
    found_tags = []
    for r in results:
        # extract the bounding box (x, y)-coordinates for the AprilTag
        # and convert each of the (x, y)-coordinate pairs to integers
        (ptA, ptB, ptC, ptD) = r.corners
        ptB = (int(ptB[0]), int(ptB[1]))
        ptC = (int(ptC[0]), int(ptC[1]))
        ptD = (int(ptD[0]), int(ptD[1]))
        ptA = (int(ptA[0]), int(ptA[1]))
        
        found_tags.append({'family' : r.tag_family.decode("utf-8"),
                           'id'     : r.tag_id,
                           'corners': [ptA,ptB,ptC,ptD],
                           'center' : (int(r.center[0]), int(r.center[1])) })
    print("about to return:")
    print(found_tags)
    return found_tags

key = cv2. waitKey(1)

cap0 = cv2.VideoCapture(0)

def runCamera(cap, index):
    ret, frame = cap.read()
    if ret:
        tags = detect_tags(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))

        for tag in tags:
            # draw the bounding box of the AprilTag detection
            cv2.line(image, tag['corners'][0], tag['corners'][1], (0, 255, 0), 2)
            cv2.line(image, tag['corners'][1], tag['corners'][2], (0, 255, 0), 2)
            cv2.line(image, tag['corners'][2], tag['corners'][3], (0, 255, 0), 2)
            cv2.line(image, tag['corners'][3], tag['corners'][0], (0, 255, 0), 2)

            cv2.putText(image, tag['id'], (tag['corners'][0][0], tag['corners'][0][1] - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            print("[INFO] tag family: {}".format(tagFamily))

        cv2.imshow("Camera " + str(index), frame)

while True:
    try:
    
        runCamera(cap0, 0)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        
    except(KeyboardInterrupt):
        cap0.release()

        print("Cams Off. Program ended.")
        cv2.destroyAllWindows()
        break
