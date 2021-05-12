#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
The Depthstimator class estimtes depth in an image by wrapping a trained neural
network. For now, this only supports FCRN.

This class' code draws heavily from FCRN's predict.py

@author: nhewitt
"""

import torch
import torch.nn.parallel
import torch.optim
import torchvision.transforms as T

from PIL import Image
import numpy as np
from skimage.transform import resize

from matplotlib import pyplot as plt
from time import time

import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__),
                             os.pardir, 'fast-depth'))


class Depthstimator:
    # image_shape: (height, width)
    def __init__(self, image_shape, ckpt_path = None, cuda = True):
        self.image_shape = image_shape
        self.cuda = cuda
        
        if ckpt_path is None:
            self.ckpt_path = os.path.join(os.path.dirname(__file__),
                                                'cfg', 'mobilenet-nnconv5dw-skipadd-pruned.pth.tar')
        else:
            self.ckpt_path = ckpt_path
        
        # Default input size
        height = 224
        width = 224
        self.input_shape = (height, width)
        
        # Load the network
        checkpoint = torch.load(self.ckpt_path)
        self.model = checkpoint['model']
        self.model.eval()
        
        if self.cuda:
            self.model = self.model.cuda()
        else:
            self.model = self.model.cpu()
        
        # Create a transform for the input image
        self.trf = T.Compose([T.Resize((224,224)),
                        T.ToTensor()])
        
        
    def predict(self, img):
        # Transform image to tensor    
        inp = self.trf(img).unsqueeze(0)
        if self.cuda:
            inp = inp.cuda()
        else:
            inp = inp.cpu()
        
        # Run net and convert results to numpy
        t = time()
        out = self.model(inp).detach()
        t = time() - t
        print(f'Depth model time: {t}')
        
        if self.cuda:
            out = out.cpu()
        out = out.numpy()
        out = out[0,0,:,:].squeeze()
        
        # Upscale
        pred = resize(out, self.image_shape)
        
        return pred
    
    
    def test(self):
        img = Image.open('test.jpg')
        pred = self.predict(img)
        
        fig = plt.figure()
        #ii = plt.imshow(pred[0,:,:,0], interpolation='nearest')
        ii = plt.imshow(pred, interpolation='nearest')
        fig.colorbar(ii)
        plt.show()
    
        
def main():
    d = Depthstimator((3024, 4032))
    d.test()
    
    
if __name__=='__main__':
    main()