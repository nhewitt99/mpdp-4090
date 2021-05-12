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
from HumanSegmentor import EfficientHRNetSegmentor, resnetSegmentor
from Projector import Projector

import pickle
from PIL import Image
from time import time

import numpy as np
import scipy


# Median absolute deviation outlier detection, one-dimensional
def outlier_detection(points, thresh=2.0):
    median = np.median(points, axis=0)
    diff = points - median
    
    med_abs_dev = np.median(diff)
    modified_z_score = 0.6745 * diff / med_abs_dev
    return modified_z_score > thresh
    


def main():
    img = Image.open('/home/nhewitt/Pictures/mpdp-imgs/1-2-5.jpg')
    width, height = img.size
    
    calibration = pickle.load(open('src/calibration.pickle', 'rb'))
    camera_mtx = calibration['camera-matrix']
    
    depth_estimator = Depthstimator((height, width), cuda=True)
    
    segmentor = resnetSegmentor((height, width), cuda=True)
    # segmentor = EfficientHRNetSegmentor((height, width))
    # segmentor = SamplingSegmentor((height, width), 0.25)
    
    point_projector = Projector(camera_mtx)
    
    t = time()
    
    depth = depth_estimator.predict(img)
    pixels_of_interest = segmentor.segment(img)
    
    totalXYZ = None
    totalrgb = None
    centerpoints = []
    
    for pixels in pixels_of_interest:
        XYZ = point_projector.projectMany(depth, pixels)
        rgb = point_projector.getColors(img, pixels)
        
        # Outlier detection: sadly, this usually removes the pixels we want, not the erroneous ones
        # z_outliers = outlier_detection(XYZ[:,2])
        # remaining_idx = np.where(z_outliers==1)
        
        # Naively select closest pixels to avoid the stretched ones
        z_cutoff = np.percentile(XYZ[:,2], 25.0)
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
        thiscenter[1] = 0
        centerpoints.append(thiscenter)
    
    # Calculate pairwise distances
    dists = scipy.spatial.distance.pdist(np.vstack(centerpoints))
    dists = scipy.spatial.distance.squareform(dists)
    
    # Calculate distances from camera
    for idx, point in enumerate(centerpoints):
        dists[idx,idx] = np.linalg.norm(point)
        
    t = time() - t
    print(f'Total pipeline time: {t}')
    
    print(dists)
    point_projector.visualize(totalrgb, totalXYZ, circle_points=centerpoints, axes=True)
    
    
if __name__=='__main__':
    main()