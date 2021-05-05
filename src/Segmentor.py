#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Abstract base class for segmentation phase of pipeline and a simple sampling-based
segmentor.

@author: nhewitt
"""

from abc import ABC, abstractmethod
import numpy as np


class Segmentor(ABC):
    def __init__(self, image_shape):
        self.height = image_shape[0]
        self.width = image_shape[1]
        super().__init__()
        
    
    '''
    Should always return List of 2d np array of shape (points_in_object, 2),
    where each list entry corresponds to a uniquely segmented object
    '''
    @abstractmethod
    def segment(self, image):
        pass
    
    
class SamplingSegmentor(Segmentor):
    def __init__(self, image_shape, scaling):
        self.scaling = scaling
        super().__init__(image_shape)
     
        
    def segment(self, image):
        w = round(self.width * self.scaling)
        h = round(self.height * self.scaling)
        
        ret = np.zeros((w*h, 2))
        
        # Must be a more pythonic way to do this...
        for i in range(w):
            for j in range(h):
                x = round(i / self.scaling)
                y = round(j / self.scaling)
                
                index = i*h + j
                ret[index,:] = np.array([x, y])
                
        return [ret.astype('int32')]
    