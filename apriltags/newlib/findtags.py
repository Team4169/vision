import apriltag
import cv2
def find_tags image:
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
	    
	    # draw the bounding box of the AprilTag detection
	    cv2.line(image, ptA, ptB, (0, 255, 0), 2)
	    cv2.line(image, ptB, ptC, (0, 255, 0), 2)
	    cv2.line(image, ptC, ptD, (0, 255, 0), 2)
	    cv2.line(image, ptD, ptA, (0, 255, 0), 2)
	    
	    # draw the center (x, y)-coordinates of the AprilTag
	    (cX, cY) = (int(r.center[0]), int(r.center[1]))
	    cv2.circle(image, (cX, cY), 5, (0, 0, 255), -1)
	    
	    # draw the tag family on the image
	    tagFamily = r.tag_family.decode("utf-8")
	    cv2.putText(image, tagFamily, (ptA[0], ptA[1] - 15),
		    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
	    found_tags.append({'family': r.tag_family.decode("utf-8"),
	                       'id'    : r.tag_id,
	                       'center': (int(r.center[0]), int(r.center[1]))})
    print(found_tags)
