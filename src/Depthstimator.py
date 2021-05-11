#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
The Depthstimator class estimtes depth in an image by wrapping a trained neural
network. For now, this only supports FCRN.

This class' code draws heavily from FCRN's predict.py

@author: nhewitt
"""

import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

from PIL import Image
import numpy as np
from skimage.transform import resize

from matplotlib import pyplot as plt

import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__),
                             os.pardir, 'FCRN-DepthPrediction', 'tensorflow'))
import models


class Depthstimator:
    # image_shape: (height, width)
    def __init__(self, image_shape, model_data_path = None):
        self.image_shape = image_shape
        
        if model_data_path is None:
            self.model_data_path = os.path.join(os.path.dirname(__file__),
                                                os.pardir, 'FCRN-DepthPrediction',
                                                'tensorflow', 'models', 'NYU_FCRN.ckpt')
        else:
            self.model_data_path = model_data_path
        
        # Default input size
        height = 228
        width = 304
        self.input_shape = (height, width)
        channels = 3
        batch_size = 1
        
        # Create a placeholder for the input image
        self.input_node = tf.placeholder(tf.float32, shape=(None, height, width, channels))
    
        # Construct the network
        self.net = models.ResNet50UpProj({'data': self.input_node}, batch_size, 1, False)
        tf.get_variable_scope().reuse_variables()
        
        self.sess = tf.Session()
        
        # Load the converted parameters
        print('Loading the model')

        # Use to load from ckpt file
        saver = tf.train.Saver()     
        saver.restore(self.sess, self.model_data_path)
        
        
    def predict(self, img):
        # Preprocess to net input size
        img = np.array(img).astype('float32')
        img = resize(img, self.input_shape)
        img = np.expand_dims(np.asarray(img), axis = 0)
        
        # Run net
        pred = self.sess.run(self.net.get_output(), feed_dict={self.input_node: img})
        pred = pred[0,:,:,0].squeeze()
        
        # Upscale
        pred = resize(pred, self.image_shape)
        
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