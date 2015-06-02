
import numpy as np
import cv2
import pdb
import math

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml') 
mouth_cascade = cv2.CascadeClassifier('haarcascade_mcs_mouth.xml')
nose_cascade = cv2.CascadeClassifier('haarcascade_mcs_nose.xml')

img = cv2.imread('sampleFaceImage3.JPG')
img2 = img.copy()
#img = cv2.imread('sampleFaceImage2.jpg')
#img = cv2.imread('sampleFaceImage3.JPG')
#----For Debugging-------
# img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

height, width, depth= img.shape
height, width = width, height
maxDimension = 400
if (height>width):
    dim = (maxDimension, int(((width*maxDimension)/height)))
    img = cv2.resize(img, dim)#, interpolation = cv2.INTER_AREA)
else:
    dim = (int(((height*maxDimension)/width)), maxDimension)
# pdb.set_trace();
img = cv2.resize(img, dim)#, interpolation = cv2.INTER_AREA)
# gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

#img2 = cv2.resize(img, img2, np.array([height, width]), b, b, cv2.INTER_LINEAR)

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
cv2.equalizeHist(gray, gray)

faces = face_cascade.detectMultiScale(gray, 1.1,2)
midPoint = np.arange(8)
#midPoint = [0 for x in range(6)]

mask = np.zeros(img.shape[:2],dtype = np.uint8) # mask initialized to PR_BG
bgdmodel = np.zeros((1,65),np.float64)
fgdmodel = np.zeros((1,65),np.float64)
for (x,y,w,h) in faces:
    cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
    rect = (max((x-w/2), 0),max((y-h/2),0),2*w,2*h)
    #cv2.equalizeHist(img,rect)
    cv2.grabCut(img,mask,rect,bgdmodel,fgdmodel,1,cv2.GC_INIT_WITH_RECT)


mask2 = np.where((mask==1) + (mask==3),255,0).astype('uint8')
output = cv2.bitwise_and(img,img,mask=mask2)

cv2.imshow('input image',img)
cv2.imshow('output image',output)
cv2.waitKey(0)
cv2.destroyAllWindows()



