
import numpy as np
import cv2
import pdb
import math

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml') 
mouth_cascade = cv2.CascadeClassifier('haarcascade_mcs_mouth.xml')
nose_cascade = cv2.CascadeClassifier('haarcascade_mcs_nose.xml')

img = cv2.imread('sampleFaceImage.png')
#img = cv2.imread('sampleFaceImage2.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
cv2.equalizeHist(gray, gray)

faces = face_cascade.detectMultiScale(gray, 1.1,2)
midPoint = np.arange(8)
#midPoint = [0 for x in range(6)]
#midpoint = np.arange(6)

for (x,y,w,h) in faces:
    cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
    #img = cv2.rectangle(img,(y,x),(y+h,x+w),(255,0,0),2)
    #pdb.set_trace()
    midPoint[0] = x+w/2;
    midPoint[1] = y+h/2;
    roi_gray = gray[y:y+h, x:x+w]
    roi_color = img[y:y+h, x:x+w]
    eyes = eye_cascade.detectMultiScale(roi_gray)
    temp1 = 0;
    temp2 = 0;
    count = 0;
    for (ex,ey,ew,eh) in eyes:
        cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)
        temp1 = temp1 + ex + ew/2;
        temp2 = temp2 + ey + eh/2;
        count = count+1;
    #midPoint[2] = temp1/count;
    #midPoint[3] = temp2/count;
    midPoint[2],midPoint[3]=midPoint[0],midPoint[1]

# If the mouth detection does not work, adjust the second parameter inside the #detectMultiScale. It acts as a measure of confidence.

    mouth = mouth_cascade.detectMultiScale(roi_gray,4,2)
    for (mx,my,mw,mh) in mouth:
        cv2.rectangle(roi_color,(mx,my),(mx+mw,my+mh),(0,0,255),2)
    midPoint[6] = mx + mw/2
    midPoint[7] = my + mh/2
    #midPoint[6],midPoint[7]=midPoint[0],midPoint[1]

    nose = nose_cascade.detectMultiScale(roi_gray)
    for (nx,ny,nw,nh) in nose:
        cv2.rectangle(roi_color,(nx,ny),(nx+nw,ny+nh),(0,255,0),2)
    midPoint[4] = nx+nw/2;
    midPoint[5] = ny+nh/2;

#pdb.set_trace()
#for i in range (0,5):
midPointDebug = midPoint.reshape(4,2);
[vx,vy,x,y] = cv2.fitLine(midPointDebug, cv2.cv.CV_DIST_L1, 0, 0.01, 0.01)
x = np.around(x)
y = np.around(y)
distance = 100;
#x2 = x+distance*vy;
#y2 = y+distance*vx;
x2,x3=x,x
y2=y+70
y3=y-60
#theta = math.atan(vy/vx)
#x2 = x + (math.cos(theta))*distance
#y2 = y + (math.sin(theta))*distance
x2 = np.around(x2)
y2 = np.around(y2)
cv2.line(img,(x,y),(x2,y2),(255,0,0),2)
cv2.line(img,(x,y),(x3,y3),(255,0,0),2)
cv2.imshow('img',img)
cv2.waitKey(0)
cv2.destroyAllWindows()
#pdb.set_trace()
