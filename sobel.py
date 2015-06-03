import cv2
import numpy as np
import pdb
from matplotlib import pyplot as plt

img = cv2.imread('sampleFaceImage.png',0)

kernel = np.ones((5,5),np.float32)/25
img = cv2.filter2D(img,-1,kernel)

#laplacian = cv2.Laplacian(img,cv2.CV_64F)
sobelx = cv2.Sobel(img,cv2.CV_64F,1,0,ksize=5)
sobely = cv2.Sobel(img,cv2.CV_64F,0,1,ksize=5)

# plt.subplot(2,1,1),plt.imshow(img,cmap = 'gray')
# plt.title('Original'), plt.xticks([]), plt.yticks([])
# plt.subplot(2,1,2),plt.imshow(laplacian,cmap = 'gray')
# plt.title('laplacian'), plt.xticks([]), plt.yticks([])

# plt.subplot(2,2,3),plt.imshow(sobelx,cmap = 'gray')
# plt.title('Sobel X'), plt.xticks([]), plt.yticks([])
# plt.subplot(2,2,4),plt.imshow(sobely,cmap = 'gray')
# plt.title('Sobel Y'), plt.xticks([]), plt.yticks([])

cv2.imshow('laplacian',laplacian)
cv2.imshow('sobelx',sobelx)
cv2.imshow('sobely', sobely)
cv2.waitKey(0)

cv2.destroyAllWindows()
#pdb.set_trace();
#plt.show()