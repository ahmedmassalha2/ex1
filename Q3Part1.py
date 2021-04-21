import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt


def solveQ3Part1():
    src = cv.imread("Q3/Dylan.jpg")
    target=cv.imread("Q3/frames.jpg")
    srcTri = np.array( [[0,0],[640,0],[640,480]] ).astype(np.float32)
    
    ######################Affine transform#####################################
    dstTriAffin = np.array( [[551,220],[844,66],[901,299]] ).astype(np.float32)
    warp_mat = cv.estimateAffine2D(srcTri, dstTriAffin)[0]
    warp_dst_affin = cv.warpAffine(src, warp_mat, (target.shape[1], target.shape[0]))
    ###########################################################################
    
    
    ######################Proj transform#####################################
    srcTri = np.array( [[0,0],[640,0],[640,480],[0,480]] ).astype(np.float32)
    dstTriProj = np.array( [[195,55],[495,159],[431,498],[37,182]] ).astype(np.float32)
    warp_mat_proj = cv.getPerspectiveTransform(srcTri, dstTriProj)
    warp_dst_affin_proj=cv.warpPerspective(src,warp_mat_proj,(target.shape[1], target.shape[0]))
    
    cv.imwrite('warpedImageOverFrames.jpg', target +warp_dst_affin+warp_dst_affin_proj)
    cv.imwrite('warpedImageOverBlackBackground.jpg',warp_dst_affin+warp_dst_affin_proj)
    
    plt.imshow(warp_dst_affin+warp_dst_affin_proj)
    plt.show()
    plt.imshow(target +warp_dst_affin+warp_dst_affin_proj)
    plt.show()
    print("Results saved in warpedImageOverBlackBackground.jpg and warpedImageOverFrames.jpg\n")
    