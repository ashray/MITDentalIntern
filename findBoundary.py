import numpy as np
import cv2
import pdb


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
        # temp = temp + 1
        # pdb.set_trace()

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

# img = cv2.imread('sampleFaceImage.png',0)
img = cv2.imread('downscaled_image.png')
img = cv2.GaussianBlur(img,(5,5),0)
canny_edge = cv2.Canny(img,50,100)
dilation_kernel = np.ones((5,5), np.uint8)
closed_canny = cv2.morphologyEx(canny_edge, cv2.MORPH_CLOSE, dilation_kernel)

a = findBoundary(closed_canny)
# print a[:,150]
num = len(a[0,:])
for i in range(0,num):
    cv2.line(img, (a[0][i], a[1][i]), (a[0][i], a[1][i]), (255,0,0),6)

cv2.imshow('img', img)
cv2.waitKey(0)
cv2.destroyAllWindows()