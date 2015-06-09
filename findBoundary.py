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
#     storeBoundaryPoints = np.hstack((storeLeftBoundaryPoints, storeRightBoundaryPoints))
#     return storeBoundaryPoints

def findBoundary(inputImage):
    pdb.set_trace()
    contours, hierarchy = cv2.findContours(inputImage,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    a,b,c = contours[0].shape
    #Typically b turns out to be 1, which creates a problem for us
    return contours[0].reshape(a,c)
    # pdb.set_trace()


# img = cv2.imread('sampleFaceImage.png')
# img = image_downscale(img, 400)
# img_copy = img.copy()
# gray = colGray(img)
