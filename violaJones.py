
import numpy as np
import cv2
import pdb

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml') 
mouth_cascade = cv2.CascadeClassifier('haarcascade_mcs_mouth.xml')
nose_cascade = cv2.CascadeClassifier('haarcascade_mcs_nose.xml')

img = cv2.imread('sampleFaceImage.png')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
cv2.equalizeHist(gray, gray)

faces = face_cascade.detectMultiScale(gray, 1.1,2)
for (x,y,w,h) in faces:
    cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
    #img = cv2.rectangle(img,(y,x),(y+h,x+w),(255,0,0),2)
    #pdb.set_trace()

    roi_gray = gray[y:y+h, x:x+w]
    #pdb.set_trace()
    roi_color = img[y:y+h, x:x+w]
    eyes = eye_cascade.detectMultiScale(roi_gray)
    for (ex,ey,ew,eh) in eyes:
        cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)

 #   mouth = mouth_cascade.detectMultiScale(roi_gray)
  #  for (mx,my,mw,mh) in mouth:
   #     cv2.rectangle(roi_color,(mx,my),(mx+mw,my+mh),(0,255,0),2)

    nose = nose_cascade.detectMultiScale(roi_gray)
    for (nx,ny,nw,nh) in nose:
        cv2.rectangle(roi_color,(nx,ny),(nx+nw,ny+nh),(0,255,0),2)

#pdb.set_trace()
cv2.imshow('img',img)
cv2.waitKey(0)
cv2.destroyAllWindows()
#pdb.set_trace()
