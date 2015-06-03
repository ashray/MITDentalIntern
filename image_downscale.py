
import numpy as np
import cv2
import pdb
import math


img = cv2.imread('sampleFaceImage3.JPG')
#img = cv2.imread('./photo/sampleFaceImage4.JPG')
#img = cv2.imread('sampleFaceImage2.jpg')
#img = cv2.imread('./photo/sampleFaceImage3.JPG')
#----For Debugging-------
# img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

height, width, depth= img.shape
height, width = width, height
maxDimension = 1000
if (height>width):
    dim = (maxDimension, int(((width*maxDimension)/height)))
    img = cv2.resize(img, dim)#, interpolation = cv2.INTER_AREA)
else:
    dim = (int(((height*maxDimension)/width)), maxDimension)
# pdb.set_trace();
img = cv2.resize(img, dim)#, interpolation = cv2.INTER_AREA)

cv2.imshow('img',img)
cv2.waitKey(0)
cv2.destroyAllWindows()

cv2.imwrite('downscaled_image.png',img)