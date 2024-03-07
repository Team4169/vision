import cv2
from time import time
 
<<<<<<< HEAD

starttime = time() 

=======
>>>>>>> 83f96681f409c7b8f4942725506b4bdd546fc148
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
