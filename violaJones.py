
import numpy as np
import cv2
import pdb

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml') 
mouth_cascade = cv2.CascadeClassifier('haarcascade_mcs_mouth.xml')
nose_cascade = cv2.CascadeClassifier('haarcascade_mcs_nose.xml')

img = cv2.imread('sampleFaceImage.png')
#img = cv2.imread('sampleFaceImage2.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
cv2.equalizeHist(gray, gray)

faces = face_cascade.detectMultiScale(gray, 1.1,2)

midPoint = [[0 for x in range(2)] for x in range(3)]

for (x,y,w,h) in faces:
    cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
    #img = cv2.rectangle(img,(y,x),(y+h,x+w),(255,0,0),2)
    #pdb.set_trace()
    midPoint[0][0] = x+w/2;
    midPoint[0][1] = y+h/2;
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
    midPoint[1][0] = temp1/count;
    midPoint[1][1] = temp2/count;
# If the mouth detection does not work, adjust the second parameter inside the detectMultiScale. It acts as a measure of confidence.
    mouth = mouth_cascade.detectMultiScale(roi_gray,4,2)
    for (mx,my,mw,mh) in mouth:
        cv2.rectangle(roi_color,(mx,my),(mx+mw,my+mh),(0,0,255),2)

    nose = nose_cascade.detectMultiScale(roi_gray)
    for (nx,ny,nw,nh) in nose:
        cv2.rectangle(roi_color,(nx,ny),(nx+nw,ny+nh),(0,255,0),2)
    midPoint[2][0] = nx+nw/2;
    midPoint[2][1] = ny+nh/2;

pdb.set_trace()
symmetricalLine = cv2.fitLine(midPoint, cv2.cv.CV_DIST_L1, 0, 0.01, 0.01)
cv2.imshow('img',img)
cv2.waitKey(0)
cv2.destroyAllWindows()
#pdb.set_trace()
