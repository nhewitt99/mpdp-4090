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


def main():
    img = Image.open('src/test-me.jpg')
    width, height = img.size
    
    calibration = pickle.load(open('src/calibration.pickle', 'rb'))
    camera_mtx = calibration['camera-matrix']
    
    depth_estimator = Depthstimator((height, width))
    
    segmentor = resnetSegmentor((height, width))
    # segmentor = EfficientHRNetSegmentor((height, width))
    # segmentor = SamplingSegmentor((height, width), 0.125)
    
    point_projector = Projector(camera_mtx)
    
    depth = depth_estimator.predict(img)
    pixels_of_interest = segmentor.segment(img)
    
    for pixels in pixels_of_interest:
        XYZ = point_projector.projectMany(depth, pixels)
        rgb = point_projector.getColors(img, pixels)
        
        point_projector.visualize(rgb, XYZ)
    
    
if __name__=='__main__':
    main()