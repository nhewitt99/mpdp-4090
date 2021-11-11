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
