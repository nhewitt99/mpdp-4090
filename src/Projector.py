#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
The Projector class uses camera extrinsics to project depth points out of the
image plane and into the world frame. Segmented groups of pixels can be
specified to only run the projection on certain regions of interest

@author: nhewitt
"""

import numpy as np
import open3d as o3d # Optional, can comment out this and visualize() if open3d is not installed.

colors = [[1,0,0],
          [0,1,0],
          [0,0,1],
          [1,1,0],
          [1,0,1],
          [0,1,1],
          [1,1,1]]

class Projector:
    def __init__(self, camera_mtx):
        self.camera_mtx = camera_mtx
        
    
    '''
    Get colors for many pixels from image
    
    img: PIL.Image
    pixelsOfInterest: np.array or List of ints, [[x1,y1], ..., [xn,yn]]
    return: colors per pixel, np.array of float32, [[r1,b1,g1], ..., [rn,bn,gn]]
    '''
    def getColors(self, img, pixelsOfInterest):
        img_arr = np.array(img).astype('float32')
        
        rgb = img_arr[pixelsOfInterest[:,1], pixelsOfInterest[:,0], :]
        
        return rgb / 255
    
    
    '''
    Get depth value for many pixels from depth image
    
    depth: depth image as np.array, List, or similar
    pixelsOfInterest: np.array or List of ints, [[x1,y1], ..., [xn,yn]]
    return: vector of depth per pixel, np.array of float32, [d1, ..., dn]
    '''
    def getDepth(self, depth, pixelsOfInterest):
        d_arr = np.array(depth).astype('float32')
        
        d_vect = d_arr[pixelsOfInterest[:,1], pixelsOfInterest[:,0]]
        
        return d_vect
        
    
    '''
    Project one point from image frame to world frame
    
    xyd: np.array, List, tuple, or similar of x (int), y (int), and depth (float)
    f_*, c_*: camera extrinsics
    return: world-frame coordinates, np.array of float32, [X,Y,Z]
    '''
    def projectOnce(self, xyd, f_x, f_y, c_x, c_y):
        # Image coordinates
        x = xyd[0]
        y = xyd[1]
        d = xyd[2]
        
        # Camera frame coordinates
        Z = d
        X = (x - c_x) * Z / f_x
        Y = (y - c_y) * Z / f_y
        
        return np.array([X,Y,Z])
    
    
    '''
    Project many points from image frame to world frame
    
    depth: depth image as np.array, List, or similar
    pixelsOfInterest: np.array or List of ints, [[x1,y1], ..., [xn,yn]]
    return: array of all pixels' world-frame coordinates,
            np.array of float32, [[X1,Y1,Z1], ..., [Xn,Yn,Zn]]
    '''
    def projectMany(self, depth, pixelsOfInterest):
        # Get column of depth and add as new column to pixels
        d = self.getDepth(depth, pixelsOfInterest)
        d = np.expand_dims(d, axis=1) # Convert to 2D column
        xyd = np.hstack((pixelsOfInterest, d))
        
        # Pull extrinsics from calibration
        f_x = self.camera_mtx[0,0]
        f_y = self.camera_mtx[1,1]
        c_x = self.camera_mtx[0,2]
        c_y = self.camera_mtx[1,2]
        
        # Project each point
        Z = xyd[:,2]
        X = (xyd[:,0] - c_x) * Z / f_x
        Y = (xyd[:,1] - c_y) * Z / f_y
        
        X = np.expand_dims(X,0).transpose()
        Y = np.expand_dims(Y,0).transpose()
        Z = np.expand_dims(Z,0).transpose()
        
        XYZ = np.hstack((X,Y,Z))
        
        return XYZ


    '''
    Optionally add an axis indicator for visualization
    
    rgb: np.array of colors per pixel
    XYZ: np.array of world-frame points
    length: how far in meters(?) to place indicator points from origin
    '''
    def addAxes(self, rgb, XYZ, length = 0.5):
        colors = np.array([[0,0,0],
                           [1,0,0],
                           [0,1,0],
                           [0,0,1]]).astype('float32')
        
        points = np.array([[0,0,0],
                           [length, 0, 0],
                           [0, length, 0],
                           [0, 0, length]])
        
        # Stack as new rows and return
        return np.vstack((rgb, colors)), np.vstack((XYZ, points))
    
    
    '''
    Optionally visualize with open3d
    
    rgb: np.array of colors per pixel
    XYZ: np.array of world-frame points
    axes: boolean to add an axis indicator
    '''
    def visualize(self, rgb, XYZ, circle_points = None, axes = False):
        objects = []
        
        if axes:
            frame = o3d.geometry.create_mesh_coordinate_frame(0.5)
            objects.append(frame)
            
        if circle_points is not None:
            i = 0
            for point in circle_points:
                mesh_sphere = o3d.geometry.create_mesh_sphere(radius=0.1)
                mesh_sphere.compute_vertex_normals()
                mesh_sphere.paint_uniform_color(colors[i])
                i += 1
                
                point = tuple(point)
                mesh_sphere.translate(point)
                objects.append(mesh_sphere)
                
        if XYZ is not None:
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(XYZ)
            pcd.colors = o3d.utility.Vector3dVector(rgb)
            objects.append(pcd)
        
        
        o3d.visualization.draw_geometries(objects)