#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 12 18:12:02 2021

@author: nhewitt, Keith Chang, Bradley Matheson
"""

import cv2
import numpy as np
from scipy import ndimage

from Segmentor import Segmentor


class TargetSegmentor(Segmentor):
    def __init__(self, image_shape, prune_threshold=1000):
        self.holder = 0
        self.prune_threshold = prune_threshold
        super().__init__(image_shape)
        
    
    def segment(self, image):
        frame = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        
        #copy = frame
        frame = cv2.medianBlur(frame,5)
        cimg = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        #Change function based on camera resolution
        circles = cv2.HoughCircles(cimg,method=cv2.HOUGH_GRADIENT, dp=2, 
                            minDist=1000, param1=650, param2=65,minRadius=10, maxRadius=200)
        
        new_mask = np.zeros((cimg.shape), dtype=np.uint8)
        xy = []
        
        if(circles is None):
            new_mask = np.asarray(new_mask)
            return new_mask, xy
        
        circles = np.uint16(np.around(circles))
        for i in circles[0,:]:
            # draw the outer circle
            cv2.circle(cimg,(i[0],i[1]),i[2],(0,255,0),2)
            # draw the center of the circle
            cv2.circle(cimg,(i[0],i[1]),2,(0,0,255),3)

            #get a set of cordinates
            xy.append([i[0], i[1]])
            # draw the outer circle
            cv2.circle(new_mask,(i[0],i[1]),i[2],(255,255,255),-1)
        # Here is where you can obtain the coordinate you are looking for
        cv2.floodFill(new_mask, None, (0, 0), 0)    
        new_mask = np.asarray(new_mask)
        
        # Split into unique occurrences
        labeled_arr, num_labels = ndimage.label(new_mask)
        dets = []
        for label in range(num_labels):
            label += 1
            
            # shape (n,2)
            wheretuple = np.where(labeled_arr == label)
            wherearr = np.transpose(np.vstack((wheretuple[1],wheretuple[0])))
            
            # Prune occurrences with too few pixels
            if wherearr.shape[0] < self.prune_threshold:
                continue
            
            dets.append(wherearr)
        
        return dets