#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Segmentation phase wrapper for human segmentation

@author: nhewitt, Keith Chang
"""

from human_seg.EfficientSegNet.config import seg_cfg
from human_seg.EfficientSegNet.EfficientSegmentation import EfficientSegNet
from human_seg.segImplementation import SegImp

from Segmentor import Segmentor

class HumanSegmentor(Segmentor):
    def __init__(self, image_shape):
        seg_cfg.defrost()
        seg_cfg.merge_from_file('src/human_seg/EfficientSegNet/seg_config.yaml')
        seg_cfg.freeze()
        
        self.seg_net = EfficientSegNet(seg_cfg)
        self.si = SegImp(self.seg_net, imageShape=(image_shape[0], image_shape[1], 3))
        
        super().__init__(image_shape)
        
    
    def segment(self, image):
        return self.si.processFramePIL(image)