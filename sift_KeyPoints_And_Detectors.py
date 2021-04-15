
import cv2 as cv
import math
import random
import numpy as np

class sift_KeyPoints_And_Detectors:
    def __init__(self,img1Path):
        
  
        # Loading the images
        img1 = cv.imread(img1Path)
        self.im = img1
        # img2 = cv2.imread(img2Path)
        gray= cv.cvtColor(img1,cv.COLOR_BGR2GRAY)
        self.grayIMG = gray
        sift = cv.xfeatures2d.SIFT_create()
        self.kp, self.des = sift.detectAndCompute(gray,None)
    def getSecondPoint(self, p, scale, angle):

        x =  int(p[0] + (scale * math.cos(angle* math.pi / 180.0)))
        y =  int(p[1] + (scale * math.sin(angle* math.pi / 180.0)))
        return(x,y)
    def drawKeyPoints(self,randomAmount = None):
        points = []
        if randomAmount == None:
            points = self.kp
        else:
            points = random.sample(self.kp, randomAmount)
        img1 = self.im
        for point in points:
            p1, p2 = (int(point.pt[0]), int(point.pt[1])), self.getSecondPoint(point.pt, point.size, point.angle)
            cv.arrowedLine(img1, p1, p2, (255,0,0), 1)
        cv.imwrite('intrestPoints.jpg',img1)
        print("See results in intrestPoints.jpg")
# a = sift_KeyPoints_And_Detectors("Q2/UoH.JPG")
# a.drawKeyPoints()