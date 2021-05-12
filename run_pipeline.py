#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May  5 12:53:41 2021

@author: nhewitt
"""

import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from Depthstimator import Depthstimator
from Segmentor import SamplingSegmentor
from HumanSegmentor import EfficientHRNetSegmentor, ResnetSegmentor
from TargetSegmentor import TargetSegmentor
from Projector import Projector
from FinalStage import DistancingFinalStage, TargetingFinalStage

import pickle
from PIL import Image
from time import time

import numpy as np
import scipy

import cv2


# Median absolute deviation outlier detection, one-dimensional
def outlier_detection(points, thresh=2.0):
    median = np.median(points, axis=0)
    diff = points - median
    
    med_abs_dev = np.median(diff)
    modified_z_score = 0.6745 * diff / med_abs_dev
    return modified_z_score > thresh
    


def main():
    mode = 'Target'
    
    img = Image.open('/home/nhewitt/Pictures/mpdp-imgs/target/2.jpg')
    width, height = img.size
    
    calibration = pickle.load(open('src/calibration.pickle', 'rb'))
    camera_mtx = calibration['camera-matrix']
    
    depth_estimator = Depthstimator((height, width), cuda=False)
    
    if mode == 'Distance':
        segmentor = ResnetSegmentor((height, width), cuda=False)
    elif mode == 'Target':
        segmentor = TargetSegmentor((height, width))
    # segmentor = EfficientHRNetSegmentor((height, width))
    # segmentor = SamplingSegmentor((height, width), 0.25)
    
    point_projector = Projector(camera_mtx)
    
    if mode == 'Distance':
        final_stage = DistancingFinalStage()
    elif mode == 'Target':
        final_stage = TargetingFinalStage()
    
    t = time()
    
    depth = depth_estimator.predict(img)
    pixels_of_interest = segmentor.segment(img)
    
    totalXYZ = None
    totalrgb = None
    centerpoints = []
    
    # mask = np.zeros((height, width, 3))
    # i = 0
    
    for pixels in pixels_of_interest:
        # if i == 0:
        #     mask[pixels[:,1], pixels[:,0]] = [0,0,255]
        #     i = 1
        # else:
        #     mask[pixels[:,1], pixels[:,0]] = [0,255,0]
        
        XYZ = point_projector.projectMany(depth, pixels)
        rgb = point_projector.getColors(img, pixels)
        
        # Outlier detection: sadly, this usually removes the pixels we want, not the erroneous ones
        # z_outliers = outlier_detection(XYZ[:,2])
        # remaining_idx = np.where(z_outliers==1)
        
        # Naively select closest pixels to avoid the stretched ones
        z_cutoff = np.percentile(XYZ[:,2], 80.0)
        remaining_idx = np.where(XYZ[:,2] < z_cutoff)
        XYZ = XYZ[remaining_idx]
        rgb = rgb[remaining_idx]
        
        if totalXYZ is None:
            totalXYZ = XYZ
            totalrgb = rgb
        else:
            totalXYZ = np.vstack((totalXYZ, XYZ))
            totalrgb = np.vstack((totalrgb, rgb))
            
        # Find the object's average point
        thiscenter = np.mean(XYZ, axis=0)
        
        # Correction for janky human detections
        if mode == 'Distance':
            thiscenter[1] = 0
        
        centerpoints.append(thiscenter)
        
    results = final_stage.process(centerpoints)
    print(results['out'])
        
    t = time() - t
    print(f'Total pipeline time: {t}')
    
    point_projector.visualize(totalrgb, totalXYZ, circle_points=centerpoints, axes=True)
    
    # img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    # output = cv2.addWeighted(img.astype("uint8"), 0.6, mask.astype("uint8"), 0.4, 0)
    # cv2.imwrite('/home/nhewitt/segoutput.jpg', output)
    
    
    
if __name__=='__main__':
    main()