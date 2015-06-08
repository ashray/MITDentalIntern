import numpy as np
import cv2
import pdb

# Finds the outermost boundary in the edge image, originally written to find outer face boundary after canny operator
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
