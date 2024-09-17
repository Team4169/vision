import cv2
from time import time

# getImages.py will save an image everytime you hit (s).
# Tips to get good images: 1) Make sure the calibration grid is very flat. It should be taped to
#    something sturdy and flat like a clipboard or similar.
# 2) Try to save pictures of the calibration grid at many different angles and tilts and positions
#    on the camera.

starttime = time()
cap = cv2.VideoCapture(0)

num = 0

while cap.isOpened():

    succes, img = cap.read()

    k = cv2.waitKey(5)

    if k == 27:
        break
    elif k == ord('s'): # wait for 's' key to save and exit
        cv2.imwrite('./images/img' + str(starttime) + str(num) + '.png', img)
        print("image saved!")
        num += 1

    cv2.imshow('Img',img)

# Release and destroy all windows before termination
cap.release()

cv2.destroyAllWindows()
