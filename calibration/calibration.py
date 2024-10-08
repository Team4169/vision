import numpy as np
import cv2 as cv
import glob, os, pickle
from time import time

running = True

while running:
  start_time = time()
  cap = cv.VideoCapture(0)

  num = 0

  while cap.isOpened():

      succes, img = cap.read()

      k = cv.waitKey(5)

      if k == 27: # escape key.
          break
      elif k == ord('s'): # wait for 's' key to save and exit
          cv.imwrite('./images/img' + str(start_time) + str(num) + '.png', img)
          print("image saved!")
          num += 1

      cv.imshow('Img',img)
  cap.release()

  cv.destroyAllWindows()

  #paper size = w: 279.4mm, h: 215.9mm
  chessboard_size = (7,9) #8 rows by 10 columns
  size_of_chessboard_squares_mm = 25 
  
  frame_size = (640,480) 

  criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)
  objp = np.zeros((chessboard_size[0] * chessboard_size[1], 3), np.float32)
  objp[:,:2] = np.mgrid[0:chessboard_size[0],0:chessboard_size[1]].T.reshape(-1,2)
  objp = objp * size_of_chessboard_squares_mm
  
  obj_points = [] 
  img_points = [] 

  images = glob.glob('images/*.png')
  if len(images) < 10:
      print("less than 10 pictures take more")
      continue

  for image in images:

      img = cv.imread(image)
      gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
      ret, corners = cv.findChessboardCorners(gray, chessboard_size, None)

      
      # If found, add object points, image points (after refining them)
      if ret == True:
          obj_points.append(objp)
          corners2 = cv.cornerSubPix(gray, corners, (11,11), (-1,-1), criteria)
          img_points.append(corners)
          if('good' in str(image)):
              pass
          else:
              # Draw and display the corners
              cv.drawChessboardCorners(img, chessboard_size, corners2, ret)
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
  ret, cameraMatrix, dist, rvecs, tvecs = cv.calibrateCamera(obj_points, img_points, frame_size, None, None)
  mean_error = 0

  for i in range(len(obj_points)):
      imgpoints2, _ = cv.projectPoints(obj_points[i], rvecs[i], tvecs[i], cameraMatrix, dist)
      error = cv.norm(img_points[i], imgpoints2, cv.NORM_L2)/len(imgpoints2)
      print(error)
      mean_error += error

  print("total error: {}".format(mean_error/len(obj_points)))

  offset = input("Please enter offset seperated by spaces (like this \"1.1 2.2 3.3\"): ") # offset should be the absolute position of the camera from the origin (center) of the robot (in meters i think)
  offset = offset.split(' ')
  while True:
    cam_name = input("Which camera is this? (should be \"Front\", \"Back\", \"Left\", or \"Right\"): ")
    if cam_name in ["Front", "Back", "Left", "Right"]:
        break
    else:
        print("Bad try again")
        
  with open(f"camConfig/camConfig{cam_name}.pkl", 'wb') as f: # Creates pkl file and puts our data in it
    pickle.dump([f"np.array({cameraMatrix}, dtype = np.float32)", f"np.array({dist}, dtype = np.float32)",f"np.array({offset}, dtype = np.float32)"], f)

  with open(f"camConfig/camConfig{cam_name}.pkl", 'rb') as f: # reopen and read the pkl file to make sure it looks good
    camConfig = pickle.load(f)
    print("camConfig:", camConfig)

  break
