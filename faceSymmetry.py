import cv2
import numpy as np
import pdb
from image_downscale import image_downscale
from violaJones import *
from canny import *
from symmetryMidpoints import symmetryMidpoints

# Accept original image as input
img = cv2.imread('sampleFaceImage.png')
img = image_downscale(img, 400)
img_copy = img.copy()
gray = colGray(img)

# midPoint = []

# Standard face detector; returns
# x,y,w and h = face bounding rectangle , midPoint = an array of old symmetry line points, intersection_x, intersection_y = points for normal to the symmetry line

midPoint, x, y, w, h, intersection_x, intersection_y = faceFeatureDetector(img)

# Finds the edge boundary points and returns the outer boundary as a series of points in variable 'a'
a = FindEdgeImage(img_copy[max((y-h),0):y + 2*h, max((x-w),0):x+2*w])

# Calculates distance between corresponding points on face curve to get an array of points for drawing central face symmetry line
# midpoints = symmetryMidpoints(a,img,0,0)
midpoints = symmetryMidpoints(a,img,x,y)

# Draws central symmetry line using new symmetryMidpoints
# xbf_temp, ybf_temp, vx_temp, vy_temp = draw_line(img,midpoints)

xbf_temp, ybf_temp, vx_temp, vy_temp = draw_line(img,midpoints)
cv2.imshow('new midpoints fitline',img)

img = PlotPoints(midpoints,img_copy, 0, 0)
# ------------------------------------------------------------
# Use fitline to fit these points on a straight line. Verify using photoshop if that is the actual centre. Also check if there is a shift of 8 points or not.
#
# Then get the slope and input it to the perpendicular function used below
# ------------------------------------------------------------

cv2.imshow('new midpoints',img)
cv2.waitKey(0)
print a.shape
# pdb.set_trace()

# Plot the face curve
# img_new = PlotPoints(a,img, x, y)
margin_size = 8
height, width, depth  = img_copy.shape
img_copy_cropped = img_copy[margin_size:(height-margin_size), margin_size:(width-margin_size)]

img_new = PlotPoints(a,img_copy_cropped, 0, 0)

cv2.imshow('img', img_new)
cv2.waitKey(0)
# len_list = len(midPoint) / 2
#
# midPointNP = np.asarray(midPoint)
#
# SymmetryLinePoints = midPointNP.reshape(len_list, 2)
#
# xbf, ybf, vx, vy= draw_line(img_new, SymmetryLinePoints)
# # sum_image1, sum_image2, img_new = skin_detector(img_new, x, y, w, h, xbf, ybf, intersection_x, intersection_y, vx, vy)
#
# [vx_perpen,vy_perpen] = Perpendicular([vx,vy])

# The upper bounding line is defined by points intersection_x, intersection_y and direction vectors vx_perpen, vy_perpen

# cv2.line(img_copy,(intersection_x,intersection_y),(intersection_x+(100*vx_perpen),intersection_y+100*vy_perpen), (255, 224, 0), 6)
# cv2.line(img_copy,(intersection_x,intersection_y),(intersection_x-(100*vx_perpen),intersection_y-100*vy_perpen), (255, 224, 0), 6)
# sub_image1 = img_copy[y:y + h, x:xbf]
# sub_image2 = img_copy[y:y + h, xbf:x + w]
# sub_image1 is the left side of the image
# sub_image2 is the right side of the image
# a has the boundary points of the face

[vx_perpen,vy_perpen] = Perpendicular([vx_temp,vy_temp])
# cv2.line(img_copy,(intersection_x,intersection_y),(intersection_x+(100*vx_perpen),intersection_y+100*vy_perpen), (255, 224, 0), 6)
# cv2.line(img_copy,(intersection_x,intersection_y),(intersection_x-(100*vx_perpen),intersection_y-100*vy_perpen), (255, 224, 0), 6)
linePoint1 = [(intersection_x+(100*vx_perpen)),(intersection_y+100*vy_perpen)]
linePoint2 = [intersection_x-(100*vx_perpen),(intersection_y-100*vy_perpen)]

[leftIntersectionPoint, rightIntersectionPoint] = FaceSymmetryLineIntersection(a, linePoint1, linePoint2)
cv2.line(img_copy_cropped, (leftIntersectionPoint[0],leftIntersectionPoint[1]), (leftIntersectionPoint[0],leftIntersectionPoint[1]), (0,255,0),10)
cv2.line(img_copy_cropped, (rightIntersectionPoint[0],rightIntersectionPoint[1]), (rightIntersectionPoint[0],rightIntersectionPoint[1]), (0,255,0),10)
cv2.imshow('img_new_cropped', img_copy_cropped)

#percentageDifference = math.fabs(sum_image1[0] - sum_image2[0]) / max(sum_image1[0], sum_image2[0])
#print "Percentage asymmetry ", percentageDifference * 100

# cv2.imshow('img', img_new)
cv2.waitKey(0)
cv2.destroyAllWindows()