from utils.general import cv2
from utils.augmentations import letterbox
import numpy as np

vc = cv2.VideoCapture(0)

if vc.isOpened(): # try to get the first frame
    rval, frame = vc.read()
else:
    rval = False
    
while rval:    
    rval, im = vc.read()
    key = cv2.waitKey(20)
    if key == 27: # exit on ESC
        break
    
    im = letterbox(im, 640, stride=1, auto=True)[0]  # padded resize
    im = im.reshape(640, 400, 3)  # HWC to CHW, BGR to RGB
    im = np.ascontiguousarray(im)  # contiguous
    
    cv2.imshow("out", im)
