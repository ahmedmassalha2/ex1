
import cv2 as cv
import random
import numpy as np
import math
import time
from matplotlib import pyplot as plt
from sift_KeyPoints_And_Detectors import sift_KeyPoints_And_Detectors
class siftSolver:
    def __init__(self,img1Path,img2Path):
        #load both image and create ip and descriptors for both
        startingTime = time.time()
        print("Loading images for sift matcher\n")
        self.im1 = sift_KeyPoints_And_Detectors(img1Path)
        self.im2 = sift_KeyPoints_And_Detectors(img2Path)
        print("Images loaded successfully, Load time:",str(time.time() - startingTime))
        
    
    def ratioTest(self):
        ip1, des1 = self.im1.kp, self.im1.des
        ip2, des2 = self.im2.kp, self.im2.des
        matches = []
        for i in range(len(des1)):
            f1 = des1[i]
            m1, m2 = float('inf'), float('inf')
            idx1,idx2 = 0, 0
            for j in range(len(des2)):
                x = math.sqrt(np.sum((np.array(f1) - np.array(des2[j]))**2))
                if x <= m1:
                    m1, m2 = x, m1
                    idx1,idx2 = j , idx1
                elif x < m2:
                     idx2 = j
            f21, f22 = des2[idx1], des2[idx2]
            if math.sqrt(np.sum((np.array(f1) - np.array(f21))**2)) / math.sqrt(np.sum((np.array(f1) - np.array(f22))**2)) < 0.8:
                matches.append([ip1[i], ip2[idx1]])
        return matches
    
    def getNN(self, f1, des):
        idx = 0
        m1 = float('inf')
        for j in range(len(des)):
            x = math.sqrt(np.sum((np.array(f1) - np.array(des[j]))**2))
            if x < m1:
                m1 = x
                idx = j
        return idx
    def initNN(self):
        ip1, des1 = self.im1.kp, self.im1.des
        ip2, des2 = self.im2.kp, self.im2.des
        ip1NN, ip2NN = dict(), dict()
        for i in range(len(des1)):
            f1 = des1[i]
            ip1NN[i] = self.getNN(f1, des2)
        for i in range(len(des2)):
            f1 = des2[i]
            ip2NN[i] = self.getNN(f1, des1)
        
        matches = []
        for key in ip1NN.keys():
            if key == ip2NN[ip1NN[key]]:
                matches.append([key,ip1NN[key]])
        matchedPoints = []
        for match in matches:
            p1, p2 = ip1[match[0]], ip2[match[1]]
            matchedPoints.append([p1, p2])
        return matchedPoints
    def bidirectionalTest(self):
        return self.initNN()
    def  matching(self, ratioOrBid = 0):
        
        startingTime = time.time() 
        matches = []
        if ratioOrBid == 0:
            print("Started matching with Ratio-Test")
            matches = self.ratioTest()
        else:
            print("Started matching with Bidirectional-Test")
            matches = self.bidirectionalTest()
            
        h1, w1 = self.im2.im.shape[:2]
        h2, w2 = self.im1.im.shape[:2]
        nWidth = w1 + w2
        nHeight = max(h1, h2)
        hdif = (h1 - h2) / 2
        newimg = np.zeros((nHeight, nWidth, 3), np.uint8)
        newimg[int(hdif):int(hdif)+h2, :w2] = self.im1.im
        newimg[:h1, w2:w1+w2] = self.im2.im
        matches = random.sample(matches, 10)
        for item in matches:
            pt1, pt2 = (int(item[0].pt[0]),int(item[0].pt[1] + hdif)), (int(item[1].pt[0] +w2),int(item[1].pt[1]))
            cv.line(newimg, pt1, pt2, (255, 0, 0))
            cv.circle(newimg, pt1, 3, (147,20,255), -1)
            cv.circle(newimg, pt2, 3, (147,20,255), -1)
        cv.imwrite('matches.jpg', newimg)
        cv.imshow('image',newimg)
        cv.waitKey(0)
        cv.destroyAllWindows()
        print("See results in matches.jpg\nRun time: ",str(time.time() - startingTime))
        return matches
# startingTime = time.time()           
# # a = siftSolver("pair1_imageA.JPG", "pair1_imageB.JPG")
# # a.im1.drawKeyPoints()
# a = sift_KeyPoints_And_Detectors("UoH.JPG")
# a.drawKeyPoints()
# print(time.time()-startingTime)
