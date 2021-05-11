import os
import cv2
import numpy as np
import time

import random

use_EfficientSegNet = True

##########################
#
# Some functions alternate between row/column
# and x/y configurations due to how some function
# parameters are x/y and the fact that the images
# are read as row/column from the segmentation
#
##########################


class SegImp():
    def __init__(self, seg_frontend_net, imageShape = (720,1280,3), image_write_path = None):
        self.seg_frontend_net = seg_frontend_net
        self.image_out_path = image_write_path
        
        #holds person segmentation pixels
        #self.segPix = []
        
        #holds significant separate objects
        self.separateObjects = []
        
        self.holdColors = []
        self.holdImage = []
        
        self.imageShape = imageShape
    
    def parseImageName(self, image):
        image_name = os.path.basename(image)
        return image_name
    
    def runSeg(self, image):
        #image_name = self.parseImageName(image)
        #image_path = self.image_out_path + "/" + image_name
        #print(image_path)
        
        #frame = cv2.imread(image)
        #print(frame)
        
        st = time.time()
        segmented_image = self.seg_frontend_net.runSegmentation(image)

        preds = np.asarray(np.argmax(segmented_image, axis=1), dtype=np.uint8)
        
        mask = np.zeros(self.imageShape, dtype=np.int8)
        
        # Generate mask from predictions
        temp = preds[0,:,:]
        mask[np.where(temp == 11)] = [0,255,0]    #truck
        mask[np.where(temp == 17)] = [255,0,0]    #person
        
        #for some reason the segmentation sees people as trucks
        personPix = np.where(temp == 11)
        
        self.segPix = []
        holder = []
        for x in range(0, len(personPix[0])):
            y = [personPix[0][x], personPix[1][x]]
            z = tuple(y)
            holder.append(z)
        self.segPix = tuple(holder)

        #print(self.segPix)
        #print(len(out))
        
        plain_background = np.zeros(self.imageShape, np.uint8)
        mask = cv2.resize(mask, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)
        output = cv2.addWeighted(image.astype("uint8"), 0.7, mask.astype("uint8"), 0.3, 0)
        # output = cv2.addWeighted(plain_background.astype("uint8"), 1, mask.astype("uint8"), 1, 0)
        en = time.time()
        print(str(en - st) + " seconds")
        # cv2.imshow("window", output)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        
        
        finaloutput = cv2.cvtColor(output, cv2.COLOR_RGB2BGR)
        cv2.imwrite('segout.jpg', finaloutput)
        
        #return out
    
    def candidate_neighbors(self, node):
        return ((node[0]-1, node[1]-1), (node[0]-1, node[1]), (node[0]-1, node[1]+1), (node[0], node[1]-1), 
                (node[0], node[1]+1), (node[0]+1, node[1]-1), (node[0]+1, node[1]), (node[0]+1, node[1]+1))

    def neighboring_groups(self, nodes):
        remain = set(nodes)
        while len(remain) > 0:
            visit = [remain.pop()]
            group = []
            while len(visit) > 0:
                node = visit.pop()
                group.append(node)
                for nb in self.candidate_neighbors(node):
                    if nb in remain:
                        remain.remove(nb)
                        visit.append(nb)
            yield tuple(group)

    def getColors(self, image):
        for i in range(0, len(self.separateObjects)):
            for j in range(0, len(self.separateObjects[i])):
                self.holdColors.append(image[self.separateObjects[i][j][0]][self.separateObjects[i][j][1]])
    
    def displayRandColor(self):
        plain_background = np.zeros(self.imageShape, np.uint8)
        for i in range(0, len(self.separateObjects)):
            color = [random.randrange(0, 255, 20), random.randrange(0, 255, 20), random.randrange(0, 255, 20)]
            for j in range(0, len(self.separateObjects[i])):
                row = self.separateObjects[i][j][0]
                col = self.separateObjects[i][j][1]
                plain_background[row][col] = color
        
        radius = 1
        color = [0, 255, 255]
        thickness = 2
        for person in self.fourPoints:
            left_point = person[0]
            top_point = person[1]
            right_point = person[2]
            bot_point = person[3]
            plain_background = cv2.circle(plain_background, left_point, radius, color, thickness)
            plain_background = cv2.circle(plain_background, top_point, radius, color, thickness)
            plain_background = cv2.circle(plain_background, right_point, radius, color, thickness)
            plain_background = cv2.circle(plain_background, bot_point, radius, color, thickness)
        
        self.holdImage = cv2.cvtColor(plain_background, cv2.COLOR_RGB2BGR)
        #cv2.imshow("window", self.holdImage)
        #cv2.waitKey(0)
        #cv2.destroyAllWindows()
    
    def displayOldColor(self, image):
        self.getColors(image)
        colorcounter = 0
        plain_background = np.zeros(self.imageShape, np.uint8)
        for i in range(0, len(self.separateObjects)):
            for j in range(0, len(self.separateObjects[i])):
                row = self.separateObjects[i][j][0]
                col = self.separateObjects[i][j][1]
                plain_background[row][col] = self.holdColors[colorcounter]
                colorcounter += 1

        #finaloutput = cv2.cvtColor(plain_background, cv2.COLOR_RGB2BGR)
        cv2.imshow("window", plain_background)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    
    def pixChecker(self, sep_objs):
        print("Total number of separate objects (np): " + str(len(self.separateObjects)))
        
        flen = 0
        for i in range(0, len(self.separateObjects)):
            flen += len(self.separateObjects[i])
            
        print("Total number of pixels post separation (np): " + str(flen))
        
        print("Total number of separate objects: " + str(len(sep_objs)))
        
        flen = 0
        for i in range(0, len(sep_objs)):
            flen += len(sep_objs[i])
        print("Total number of pixels post separation: " + str(flen))
        print("total number of pixels pre separation: " + str(len(self.segPix)))
    
    def splitObjects(self, image_string):
        #print(tuple(self.neighboring_groups(self.segPix)))
        self.separateObjects = []

        sep_objs = tuple(self.neighboring_groups(self.segPix))
        
        ind = 0
        sep_objs = list(sep_objs)
        while ind < len(sep_objs) - 1:
            if len(sep_objs[ind]) < 2000:
                sep_objs.remove(sep_objs[ind])
                ind = 0
            else:
                ind += 1
        
        for i in range(0, len(sep_objs)):
            self.separateObjects.append(np.array(sep_objs[i]))
        
        #self.pixChecker(sep_objs)
        #self.displayRandColor()
        
    def separateDimensions(self):
        
        self.sepDimensions = []
        temprowstore = []
        tempcolstore = []
        for person in self.separateObjects:
            temprow2store = []
            tempcol2store = []
            for pixel in person:
                temprow2store.append(pixel[0])
                tempcol2store.append(pixel[1])
            temprowstore.append(temprow2store)
            tempcolstore.append(tempcol2store)
        self.sepDimensions.append(temprowstore)
        self.sepDimensions.append(tempcolstore)
        #print(self.sepDimensions)
        #print(len(self.sepDimensions))
        #print(len(self.sepDimensions[0]))
        #print(len(self.sepDimensions[1]))
        
    def getOuterPoints(self):
        
        self.sepWidthHeight = []
        self.fourPoints = []
        temprow = np.array(self.sepDimensions[0])
        tempcol = np.array(self.sepDimensions[1])
        
        x1 = []
        x2 = []
        y1 = []
        y2 = []
        for person in range(0, len(tempcol)):
            f_right = np.argmax(tempcol[person], axis=0)
            f_top = np.argmax(temprow[person], axis=0)
            f_left = np.argmin(tempcol[person], axis=0)
            f_bot = np.argmin(temprow[person], axis=0)
            
            x1.append(f_left)
            x2.append(f_right)
            y1.append(f_bot)
            y2.append(f_top)
            
        # print(x1)
        # print(x2)
        # print(y1)
        # print(y2)
        
        for index in range(0, len(x1)):
            left = self.separateObjects[index][x1[index]]
            top = self.separateObjects[index][y2[index]]
            right = self.separateObjects[index][x2[index]]
            bottom = self.separateObjects[index][y1[index]]
            
            left_point = (left[1], left[0])
            top_point = (top[1], top[0])
            right_point = (right[1], right[0])
            bot_point = (bottom[1], bottom[0])
            
            self.fourPoints.append([left_point, top_point, right_point, bot_point])
        
        #print(self.fourPoints)
        # print(len(self.fourPoints))
        # for person in self.fourPoints:
            # print(person)
        
        #self.displayRandColor()
    
    def getWidthHeight(self):
        self.pixWidHei = []
        
        for person in self.fourPoints:
            pWidth = person[2][1] - person[0][1]
            pHeight = person[1][0] - person[3][0]
            self.pixWidHei.append([pWidth, pHeight])
            
        #print(self.pixWidHei)
        
    def getBboxPoints(self):
        self.bboxPoints = []
        
        for person in self.fourPoints:
            top_left = (person[0][0], person[1][1])
            bot_right = (person[2][0], person[3][1])
            self.bboxPoints.append([top_left, bot_right])
            
        #print(self.bboxPoints)
    
    def drawBboxes(self):
        
        color = [0, 0, 255]
        thickness = 3
        for person in self.bboxPoints:
            top_left = person[0]
            bot_right = person[1]
            self.holdImage = cv2.rectangle(self.holdImage, top_left, bot_right, color, thickness)
            
        #finaloutput = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        cv2.imshow("window", self.holdImage)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        
    def processFrame(self, image_string):
        image = cv2.imread(image_string)
        
        self.runSeg(image)
        self.splitObjects(image)
        
        return self.separateObjects
        
        
    def processFramePIL(self, image):
        image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        
        self.runSeg(image)
        self.splitObjects(image)
        
        return self.separateObjects
            
            
    def visualizeFrame(self):
            self.separateDimensions()
            self.getOuterPoints()
            self.getWidthHeight()
            self.getBboxPoints()
            
            self.displayRandColor()
            self.drawBboxes()
            #self.displayOldColor(image)
        
        
        
        
        
        
        
        
        
        