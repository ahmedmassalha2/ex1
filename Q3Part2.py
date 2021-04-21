
import cv2 as cv
import random
import numpy as np
import math
import time
from sift import siftSolver
from PIL import Image
from matplotlib import pyplot as plt
class ransacWarper:
    def __init__(self,img1,img2):
        self.siftMatcher = siftSolver("Q3/pair1_imageA.JPG", "Q3/pair1_imageB.JPG")
        print("Creatin matchings:")
        self.matches = self.computeMatches()
        print("matched points created successfuly")
        
        
    def getRandomKmatchedPoints(self):
        projMatches=random.sample(self.matches,4)
        affinMatches=random.sample(self.matches,3)
        return projMatches, affinMatches
        
    def computeMatches(self,option = 1):
        return self.siftMatcher.ratioTest()
        
    def getBestTransform(self, Iters = 1000):
        print("Getting best transform using RANSAC loop")
        startTime = time.time()
        affineVote, projVote = 0, 0
        dis = []
        bestAffin, bestProj, H = None, None, None
        for iteration in range(Iters):
            
            projSet, affinSet = self.getRandomKmatchedPoints()
            
            srcProj = np.float32([x[0].pt for x in projSet])
            targetProj = np.float32([x[1].pt for x in projSet])
            
            srcAffine = np.float32([x[0].pt for x in affinSet])
            targetAffine = np.float32([x[1].pt for x in affinSet])
            
            
            warp_mat_affine = cv.estimateAffine2D(srcAffine, targetAffine)[0]
            warp_mat_proj = cv.getPerspectiveTransform(srcProj, targetProj)

            affinInliers, projInliers = 0, 0
            for match in self.matches:
                pSrc, pTarget = match[0].pt, match[1].pt
                newPointProj = warp_mat_proj.dot(np.transpose(np.matrix([pSrc[0],pSrc[1],1])))
                newPointProj = [newPointProj[0]/newPointProj[2] , newPointProj[1]/newPointProj[2]]
                newPointAffin = warp_mat_affine.dot(np.transpose(np.matrix([pSrc[0],pSrc[1],1])))
                
                
                affinDistance = np.linalg.norm(np.array(newPointAffin)-np.array(pTarget))
                projDistance = np.linalg.norm(np.array(newPointProj)-np.array(pTarget))
                
                dis.append(projDistance)
                if affinDistance <= 11:
                    affinInliers += 1
                if projDistance <= 6:
                    projInliers += 1
            if affinInliers > affineVote:
                affineVote = affinInliers
                bestAffin = warp_mat_affine
            if projInliers > projVote:
                projVote = projInliers
                bestProj = warp_mat_proj
                
                
                
        #Save results as jpg images       
        warp_dst_affin = cv.warpAffine(self.siftMatcher.im1.im, bestAffin, (self.siftMatcher.im2.im.shape[1], self.siftMatcher.im2.im.shape[0]))
        cv.imwrite('affineResult.jpg',warp_dst_affin)
        warp_dst_proj = cv.warpPerspective(self.siftMatcher.im1.im, bestProj, (self.siftMatcher.im2.im.shape[1], self.siftMatcher.im2.im.shape[0]))
        cv.imwrite('projResult.jpg', warp_dst_proj)
        
        
        #plot results
        plt.title('Using affin')
        plt.imshow(warp_dst_affin)
        plt.show()
        plt.title('Using proj')
        plt.imshow(warp_dst_proj)
        plt.show()
        print("Results saved in affineResult.jpg and projResult.jpg")
        print("Total time: ",time.time() - startTime)
        return bestAffin, bestProj
                
# a = ransacWarper()
# a.getBestTransform()