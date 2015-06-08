import numpy as np
import cv2
import pdb
from findBoundary import findBoundary

# img = cv2.imread('sampleFaceImage.png',0)
img = cv2.imread('downscaled_image.png')
img = cv2.GaussianBlur(img,(5,5),0)
canny_edge = cv2.Canny(img,50,100)
dilation_kernel = np.ones((5,5), np.uint8)
closed_canny = cv2.morphologyEx(canny_edge, cv2.MORPH_CLOSE, dilation_kernel)

a = findBoundary(closed_canny)
num = len(a[0,:])
for i in range(0,num):
    cv2.line(img, (a[0][i], a[1][i]), (a[0][i], a[1][i]), (255,0,0),6)
cv2.imshow('img', img)
cv2.waitKey(0)
cv2.destroyAllWindows()