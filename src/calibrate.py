#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script to run OpenCV calibration for a camera, based on the OpenCV tutorials

@author: nhewitt
"""

import numpy as np
import cv2 as cv
import glob
import pickle
import sys


def calibrate(image_glob, dump_pickle = True):
    # termination criteria
    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    
    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
    objp = np.zeros((9*6,3), np.float32)
    objp[:,:2] = np.mgrid[0:9,0:6].T.reshape(-1,2)
    
    # Arrays to store object points and image points from all the images.
    objpoints = [] # 3d point in real world space
    imgpoints = [] # 2d points in image plane.
    images = glob.glob(image_glob)
    
    for fname in images:
        img = cv.imread(fname)
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    
        # Find the chess board corners
        ret, corners = cv.findChessboardCorners(gray, (9,6), None)
    
        # If found, add object points, image points (after refining them)
        if ret == True:
            objpoints.append(objp)
            corners2 = cv.cornerSubPix(gray,corners, (11,11), (-1,-1), criteria)
            imgpoints.append(corners)
    
            # Draw and display the corners
            cv.drawChessboardCorners(img, (9,6), corners2, ret)
            # cv.imshow('img', img)
            # cv.waitKey(500)
    
    cv.destroyAllWindows()
    
    # return, camera matrix, distortion coefs, rotation vecs, translation vecs
    ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
    
    # Check error
    mean_error = 0
    
    for i in range(len(objpoints)):
        imgpoints2, _ = cv.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
        error = cv.norm(imgpoints[i], imgpoints2, cv.NORM_L2)/len(imgpoints2)
        mean_error += error
        
    print( "total error: {}".format(mean_error/len(objpoints)) )
    
    calibration = {'camera-matrix': mtx,
                   'dist-coefs': dist,
                   'rotate-vecs': rvecs,
                   'translate-vecs': tvecs,
                   }
    
    if dump_pickle:
        pickle.dump(calibration, open('calibration.pickle', 'wb'))
        print('wrote calibration to "calibration.pickle"')
    
    return calibration
    

def main():
    argc = len(sys.argv)
    if argc == 1:
        image_glob = '../imgs/*.jpg'
    elif argc == 2:
        image_glob = str(sys.argv[1])
    else:
        print('USAGE:')
        print('$ python3 calibrator.py [images_glob_string]')
    
    calibrate(image_glob)
    

if __name__=='__main__':
    main()