import numpy as np
import cv2 as cv
import glob, os, pickle
from time import time

running = True

while running:
  starttime = time()
  cap = cv.VideoCapture(0)

  num = 0

  #photographing 
  while cap.isOpened():

      succes, img = cap.read()

      k = cv.waitKey(5)

      if k == 27: # escape key.
          break
      elif k == ord('s'): # wait for 's' key to save and exit
          cv.imwrite('./images/img' + str(starttime) + str(num) + '.png', img)
          print("image saved!")
          num += 1

      cv.imshow('Img',img)

  # Release and destroy all windows before termination
  cap.release()

  cv.destroyAllWindows()

  #init calibration

  #paper size = w: 279.4mm, h: 215.9mm
  chessboardSize = (7,9) #8 rows by 10 columns
  size_of_chessboard_squares_mm = 25 
  
  frameSize = (640,480) 

  criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)
  objp = np.zeros((chessboardSize[0] * chessboardSize[1], 3), np.float32)
  objp[:,:2] = np.mgrid[0:chessboardSize[0],0:chessboardSize[1]].T.reshape(-1,2)
  objp = objp * size_of_chessboard_squares_mm
  
  objpoints = [] 
  imgpoints = [] 


  #cheking pictures human
  images = glob.glob('images/*.png')

  for image in images:

      img = cv.imread(image)
      gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
      ret, corners = cv.findChessboardCorners(gray, chessboardSize, None)

      
      # If found, add object points, image points (after refining them)
      if ret == True:
          objpoints.append(objp)
          corners2 = cv.cornerSubPix(gray, corners, (11,11), (-1,-1), criteria)
          imgpoints.append(corners)
          if('good' in str(image)):
              pass
          else:
              # Draw and display the corners
              cv.drawChessboardCorners(img, chessboardSize, corners2, ret)
              cv.imshow('Is this image good? (d) to delete, (s) to save', img)
              while True:
                  k = cv.waitKey(20)
                  if k == ord('d'):
                      os.remove(image)
                      break
                  if k == ord('s'):
                      os.rename(image, str(image).replace('img','good'))
                      break
      else:
          os.remove(image)

  cv.destroyAllWindows()
  print("done loop")