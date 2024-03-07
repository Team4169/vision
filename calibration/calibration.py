import numpy as np
import cv2 as cv
<<<<<<< HEAD
import glob, os, pickle
=======
import glob, os
import pickle
>>>>>>> 83f96681f409c7b8f4942725506b4bdd546fc148

##### FIND CHESSBOARD CORNERS - OBJECT POINTS AND IMAGE POINTS #####

chessboardSize = (7,9)
frameSize = (640,480)


# termination criteria
criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)


# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros((chessboardSize[0] * chessboardSize[1], 3), np.float32)
objp[:,:2] = np.mgrid[0:chessboardSize[0],0:chessboardSize[1]].T.reshape(-1,2)

size_of_chessboard_squares_mm = 25
objp = objp * size_of_chessboard_squares_mm


# Arrays to store object points and image points from all the images.
objpoints = [] # 3d point in real world space
imgpoints = [] # 2d points in image plane.


images = glob.glob('images/*.png')

for image in images:

    img = cv.imread(image)
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    
    # Find the chess board corners
    ret, corners = cv.findChessboardCorners(gray, chessboardSize, None)

    
    # If found, add object points, image points (after refining them)
    if ret == True:
        objpoints.append(objp)
        corners2 = cv.cornerSubPix(gray, corners, (11,11), (-1,-1), criteria)
        imgpoints.append(corners)
<<<<<<< HEAD
        if('good' in str(image)):
            pass
        else:
            # Draw and display the corners
            cv.drawChessboardCorners(img, chessboardSize, corners2, ret)
            cv.imshow('img', img)
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
=======
        # Draw and display the corners
        cv.drawChessboardCorners(img, chessboardSize, corners2, ret)
        cv.imshow('img', img)
        cv.waitKey(1000)
        if input('delete?') == 'd':
            os.remove(image)
>>>>>>> 83f96681f409c7b8f4942725506b4bdd546fc148

cv.destroyAllWindows()


##### CALIBRATION #####

ret, cameraMatrix, dist, rvecs, tvecs = cv.calibrateCamera(objpoints, imgpoints, frameSize, None, None)

# Save the camera calibration result for later use
pickle.dump(cameraMatrix, open("vars/cameraMatrix.pkl", "wb"))
pickle.dump(dist, open("vars/dist.pkl", "wb"))


# Reprojection Error
mean_error = 0

for i in range(len(objpoints)):
    imgpoints2, _ = cv.projectPoints(objpoints[i], rvecs[i], tvecs[i], cameraMatrix, dist)
    error = cv.norm(imgpoints[i], imgpoints2, cv.NORM_L2)/len(imgpoints2)
    mean_error += error

print( "total error: {}".format(mean_error/len(objpoints)) )

with open('vars/cameraMatrix.pkl', 'rb') as f:
 data = pickle.load(f).tolist()

print('CameraMatrixData:\n',data)

print()

with open('vars/dist.pkl', 'rb') as f:
 data = pickle.load(f).tolist()

print('vars/distData:\n',str(data))
