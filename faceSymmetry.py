import cv2
import numpy as np
import pdb
from findBoundary import findBoundary
from image_downscale import image_downscale
from violaJones import *
from canny import *

img = cv2.imread('sampleFaceImage3.jpg')
img = image_downscale(img, 400)
img_copy = img.copy()
gray = colGray(img)

# midPoint = []
# count_face, count_mouth, count_nose, count = 0, 0, 0, 0
midPoint, x, y, w, h = faceFeatureDetector(img)#,count,count_face,count_mouth,count_nose)
a = FindEdgeImage(img_copy[y:y + h, x:x+w])
img_new = PlotPoints(a,img, x, y)

cv2.imshow('img', img_new)

len_list = len(midPoint) / 2

midPointNP = np.asarray(midPoint)

midPointDebug = midPointNP.reshape(len_list, 2)


xbf, ybf = draw_line(img, midPointDebug)
sum_image1, sum_image2 = skin_detector(img_copy, x, y, w, h, xbf, ybf)
percentageDifference = math.fabs(sum_image1[0] - sum_image2[0]) / max(sum_image1[0], sum_image2[0])
print "Percentage asymmetry ", percentageDifference * 100

cv2.imshow('img', img)
cv2.waitKey(0)
cv2.destroyAllWindows()