import numpy as np
import cv2
from findBoundary import findBoundary

# Finds the edge boundary points and returns the outer boundary as a series of points
def FindEdgeImage(img):
    img = cv2.GaussianBlur(img,(5,5),10)
    canny_edge = cv2.Canny(img,50,100)
    dilation_kernel = np.ones((11, 11), np.uint8)
    closed_canny = cv2.morphologyEx(canny_edge, cv2.MORPH_CLOSE, dilation_kernel)
    closed_canny = cv2.morphologyEx(closed_canny, cv2.MORPH_CLOSE, dilation_kernel)
    a = findBoundary(closed_canny)
    return a

def PlotPoints(a,img, x, y):
    a[0,:] = a[0,:] + x
    a[1,:] = a[1,:] + y
    num = len(a[0,:])
    for i in range(0,num):
        cv2.line(img, (a[0][i], a[1][i]), (a[0][i], a[1][i]), (255,0,0),6)
    return img