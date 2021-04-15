import cv2 as cv
import numpy as np

src = cv.imread("Q3/Dylan.jpg")
print(src.shape[1])
srcTri = np.array( [[0, 0], [0,476], [638,0]] ).astype(np.float32)
dstTri = np.array( [[195,55], [37,182], [494,48]] ).astype(np.float32)
warp_mat = cv.getAffineTransform(srcTri, dstTri)
warp_dst = cv.warpAffine(src, warp_mat, (src.shape[1], src.shape[0]))

h, status = cv.findHomography(srcTri, dstTri)


cv.imwrite('Warp.jpg', warp_dst)

im_dst = cv.warpPerspective(src, h,(src.shape[1], src.shape[0]))
cv.imwrite('Warp2.jpg', im_dst)