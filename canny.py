import numpy as np
import cv2
from findBoundary import findBoundary
# from findBoundary import findBoundaryWithinBox
import pdb

# Finds the edge boundary points and returns the outer boundary as a series of points

# $$$ This is the original one!
def FindEdgeImage(img):
    img = cv2.GaussianBlur(img,(5,5),10)
    canny_edge = cv2.Canny(img,50,100)
    dilation_kernel = np.ones((11, 11), np.uint8)
    closed_canny = cv2.morphologyEx(canny_edge, cv2.MORPH_CLOSE, dilation_kernel)
    closed_canny = cv2.morphologyEx(closed_canny, cv2.MORPH_CLOSE, dilation_kernel)
    margin_size = 5
    height, width  = closed_canny.shape
    closed_canny_modified = closed_canny[margin_size:(height-margin_size), margin_size:(width-margin_size)]
    # pdb.set_trace()
    a = findBoundary(closed_canny_modified)
    return a

def imageDerivative(input_image):
    sobelx = cv2.Sobel(input_image,cv2.CV_64F,1,0,ksize=5)
    sobely = cv2.Sobel(input_image,cv2.CV_64F,0,1,ksize=5)
    final_image = sobelx + sobely
    # pdb.set_trace()
    return final_image
# storeLeftBoundaryPoints, storeRightBoundaryPoints = findBoundary(closed_canny_modified)
#
#     lenLeft = len(storeLeftBoundaryPoints)
#     lenRight = len(storeRightBoundaryPoints)
#
#     lenIteration = max(lenLeft,lenRight)
#
#     for i in range(0,lenIteration):
#         if(storeLeftBoundaryPoints[1,i]<y+h):


# def FindEdgeImage(img, y, h):
#     img = cv2.GaussianBlur(img,(5,5),10)
#     canny_edge = cv2.Canny(img,50,100)
#     dilation_kernel = np.ones((11, 11), np.uint8)
#     closed_canny = cv2.morphologyEx(canny_edge, cv2.MORPH_CLOSE, dilation_kernel)
#     closed_canny = cv2.morphologyEx(closed_canny, cv2.MORPH_CLOSE, dilation_kernel)
#     margin_size = 5
#     height, width  = closed_canny.shape
#     closed_canny_modified = closed_canny[margin_size:(height-margin_size), margin_size:(width-margin_size)]
#     # pdb.set_trace()
#     a = findBoundary(closed_canny_modified, y, h)
#     return a

def PlotPoints(a,img, x, y):
    # a = a.transpose()
    # a[0,:] = a[0,:]
    # a[1,:] = a[1,:]
    num = len(a[0,:])
    for i in range(0,num):
        cv2.line(img, (a[0][i], a[1][i]), (a[0][i], a[1][i]), (255,0,0),4)
    return img