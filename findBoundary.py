import numpy as np
import cv2
import pdb

import cv2
import numpy as np
import pdb
from image_downscale import image_downscale
from violaJones import *
from canny import *

# Finds the outermost boundary in the edge image, originally written to find outer face boundary after canny operator

# $$$ This is the original findBoundary which works beautifully ! Writing a similar one with minor change to restrict the curve to only within the face bounding box
def findBoundary(inputImage):
    height, width =inputImage.shape
    margin = 3
    temp = 0
    croppedImage = inputImage[margin:(height - margin), margin:(width - margin)]
    newHeight, newWidth = croppedImage.shape
    storeLeftBoundaryPoints = np.arange(2 * int(newHeight)).reshape((2, int(newHeight)))
    for i in range(0, newHeight):
        foundPoint = 0
        for j in range(0, newWidth):
            if croppedImage[i, j] != 0:
                storeLeftBoundaryPoints[:, i] = [j, i]
                foundPoint = 1
                break
            elif foundPoint == 0:
                storeLeftBoundaryPoints[:,i] = [0,0]
    storeRightBoundaryPoints = np.arange(2 * int(newHeight)).reshape((2, int(newHeight)))
    for i in range(0, newHeight):
        foundPoint = 0
        for j in range(newWidth-1,0, -1):
            if croppedImage[i, j] != 0:
                storeRightBoundaryPoints[:, i] = [j, i]
                foundPoint = 1
                break
            elif foundPoint == 0:
                storeRightBoundaryPoints[:,i] = [0,0]
    storeBoundaryPoints = np.hstack((storeLeftBoundaryPoints, storeRightBoundaryPoints))
    return storeBoundaryPoints

# def findBoundary(inputImage):
#     height, width =inputImage.shape
#     margin = 3
#     temp = 0
#     croppedImage = inputImage[margin:(height - margin), margin:(width - margin)]
#     newHeight, newWidth = croppedImage.shape
#     storeLeftBoundaryPoints = np.arange(2 * int(newHeight)).reshape((2, int(newHeight)))
#     for i in range(0, newHeight):
#         foundPoint = 0
#         for j in range(0, newWidth):
#             if croppedImage[i, j] != 0:
#                 storeLeftBoundaryPoints[:, i] = [j, i]
#                 foundPoint = 1
#                 break
#             elif foundPoint == 0:
#                 storeLeftBoundaryPoints[:,i] = [0,0]
#     storeRightBoundaryPoints = np.arange(2 * int(newHeight)).reshape((2, int(newHeight)))
#     for i in range(0, newHeight):
#         foundPoint = 0
#         for j in range(newWidth-1,0, -1):
#             if croppedImage[i, j] != 0:
#                 storeRightBoundaryPoints[:, i] = [j, i]
#                 foundPoint = 1
#                 break
#             elif foundPoint == 0:
#                 storeRightBoundaryPoints[:,i] = [0,0]
#     # storeBoundaryPoints = np.hstack((storeLeftBoundaryPoints, storeRightBoundaryPoints))
#     return storeLeftBoundaryPoints, storeRightBoundaryPoints


# def findBoundary(inputImage):
#     # pdb.set_trace()
#     contours, hierarchy = cv2.findContours(inputImage,cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)
#     # pdb.set_trace()
#     a,b,c = contours[2].shape
#     #Typically b turns out to be 1, which creates a problem for us
#
#     len_contours = len(contours)
#     # print 'len contours',len_contours
#     # a=contours[3]
#
#     # Make this as infinite number ideally
#     min_value = 100000
#     j = 15
#     for i in range(0, len_contours-1):
#         a,b,c = contours[i].shape
#         contour_formatted = contours[i].reshape(a,c)
#         # print 'a shape',contour_formatted.shape
#         # print 'contour data',contour_formatted[:,1]
#         if min(contour_formatted[:,1])<min_value:
#             j = i
#             min_value = min(contour_formatted[:,1])
#     # for i in range (0,len_contours-1):
#
#     a,b,c = contours[j].shape
#     contour_formatted = contours[j].reshape(a,c)
#     return contour_formatted
#     # pdb.set_trace()


# img = cv2.imread('sampleFaceImage.png')
# img = image_downscale(img, 400)
# img_copy = img.copy()
# gray = colGray(img)
