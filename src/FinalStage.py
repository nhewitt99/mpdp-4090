#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Base and inherited classes for final stage of pipeline

@author: nhewitt
"""

from abc import ABC, abstractmethod
import numpy as np
import scipy
from math import atan2, pi


class FinalStage(ABC):
    def __init__(self):
        super().__init__()
        
    
    '''
    Return results as dictionary (?)
    '''
    @abstractmethod
    def process(self, centerpoints):
        pass
    
    
class DistancingFinalStage(FinalStage):
    def __init__(self):
        super().__init__()
        
    
    def process(self, centerpoints):
        # Calculate pairwise distances
        if len(centerpoints) > 0:
            dists = scipy.spatial.distance.pdist(np.vstack(centerpoints))
            dists = scipy.spatial.distance.squareform(dists)
        else:
            dists = [[]]
        
        # Calculate distances from camera
        for idx, point in enumerate(centerpoints):
            dists[idx,idx] = np.linalg.norm(point)
            
        return {'out': dists}
    
    
class TargetingFinalStage(FinalStage):
    def __init__(self):
        super().__init__()
        
    
    def process(self, centerpoints):
        adjustments = []
        
        for target in centerpoints:
            # Correct for the fact that the camera frame is upside-down
            x = target[0]
            y = -target[1]
            z = target[2]
            
            yaw = atan2(x,z) * 180 / pi
            pitch = atan2(y,z) * 180 / pi
            
            # Assume x,y are currently at 90deg, or 50% servo engagement.
            xperc = yaw + 90
            yperc = pitch + 90
            
            # Convert to percent
            xperc = 100 * xperc / 180
            yperc = 100 * yperc / 180
            
            adjustments.append([round(xperc, 2), round(yperc, 2)])
        
        return {'out': adjustments}