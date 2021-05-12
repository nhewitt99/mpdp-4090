#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Segmentation phase wrapper for human segmentation

@author: nhewitt, Keith Chang
"""

from human_seg.EfficientSegNet.config import seg_cfg
from human_seg.EfficientSegNet.EfficientSegmentation import EfficientSegNet
from human_seg.segImplementation import SegImp

import torchvision.models as models
import torchvision.transforms as T
import torch
from PIL import Image
import cv2
import numpy as np
from scipy import ndimage

from time import time

from Segmentor import Segmentor

class EfficientHRNetSegmentor(Segmentor):
    def __init__(self, image_shape):
        seg_cfg.defrost()
        seg_cfg.merge_from_file('src/human_seg/EfficientSegNet/seg_config.yaml')
        seg_cfg.freeze()
        
        self.seg_net = EfficientSegNet(seg_cfg)
        self.si = SegImp(self.seg_net, imageShape=(image_shape[0], image_shape[1], 3))
        
        super().__init__(image_shape)
        
    
    def segment(self, image):
        return self.si.processFramePIL(image)
    
    
class ResnetSegmentor(Segmentor):
    def __init__(self, image_shape, resize_to=256, prune_threshold=1000, cuda=True):
        self.cuda = cuda
        self.fcn = models.segmentation.fcn_resnet101(pretrained=True)
        
        if self.cuda:
            self.fcn = self.fcn.cuda()
        else:
            self.fcn = self.fcn.cpu()
        self.fcn.eval()
        
        # ImageNet normalization
        self.trf = T.Compose([T.Resize(resize_to),
                 #T.CenterCrop(224),
                 T.ToTensor(),
                 T.Normalize(mean = [0.485, 0.456, 0.406],
                             std = [0.229, 0.224, 0.225])])
        
        self.prune_threshold = prune_threshold
        
        super().__init__(image_shape)
        
        
    def segment(self, image):
        # Run net and get human pixels for small image
        inp = self.trf(image).unsqueeze(0)
        
        if self.cuda:
            inp = inp.cuda()
        else:
            inp = inp.cpu()
        
        t = time()
        out = self.fcn(inp)['out']
        t = time() - t
        print(f'Segment model time: {t}')
        
        classes = torch.argmax(out.squeeze(), dim=0).detach()
        if self.cuda:
            classes = classes.cpu()
        classes = classes.numpy()
        
        humanpix = np.where(classes==15)

        # Resize up to 2d mask where humans are 1 and all else 0
        mask = np.zeros((classes.shape[0], classes.shape[1],3), dtype=np.uint8)
        mask[humanpix] = [1,1,1]
        mask = cv2.resize(mask, (self.width, self.height) , interpolation=cv2.INTER_NEAREST)
        
        mask = mask[:,:,0]
        
        # Split into unique occurrences
        labeled_arr, num_labels = ndimage.label(mask)
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