#!/usr/bin/python3

'''
author: Keith Chang
'''

from human_seg.EfficientSegNet.config import seg_cfg
from human_seg.EfficientSegNet.EfficientSegmentation import EfficientSegNet
from human_seg.segImplementation import SegImp
        
def main():
    
    seg_cfg.defrost()
    seg_cfg.merge_from_file('human_seg/EfficientSegNet/seg_config.yaml')
    seg_cfg.freeze()
    
    seg_frontend_net = EfficientSegNet(seg_cfg)
        
    if seg_frontend_net != None:
        image_path = 'images/008439.jpg'
        image_write_path = 'images/out'
        
        
        si = SegImp(seg_frontend_net, image_write_path)
        si.processFrame(image_path)
        
    else:
        print("Not Okay")
        
if __name__ == '__main__':
    main()