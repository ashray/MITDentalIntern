
import numpy as np
import cv2
import pdb
import math

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml') 
mouth_cascade = cv2.CascadeClassifier('haarcascade_mcs_mouth.xml')
nose_cascade = cv2.CascadeClassifier('haarcascade_mcs_nose.xml')

#img = cv2.imread('sampleFaceImage.png')
#img = cv2.imread('sampleFaceImage2.jpg')
img = cv2.imread('sampleFaceImage3.JPG')
#----For Debugging-------
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

height, width= img.shape
height, width = width, height
if (height>width):
    dim = (100, int(((width*100)/height)))
    img = cv2.resize(img, dim)#, interpolation = cv2.INTER_AREA)
else:
    dim = (int(((height*100)/width)), 100)
# pdb.set_trace();
img = cv2.resize(img, dim)#, interpolation = cv2.INTER_AREA)
# gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


cv2.imshow("img", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
#img2 = cv2.resize(img, img2, np.array([height, width]), b, b, cv2.INTER_LINEAR)

# gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# cv2.equalizeHist(gray, gray)

# faces = face_cascade.detectMultiScale(gray, 1.1,2)
# midPoint = np.arange(8)
# #midPoint = [0 for x in range(6)]

# for (x,y,w,h) in faces:
#     cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
#     #img = cv2.rectangle(img,(y,x),(y+h,x+w),(255,0,0),2)
#     #pdb.set_trace()
#     midPoint[0] = x+w/2;
#     midPoint[1] = y+h/2;
#     #pdb.set_trace()

#     cv2.line(img,(midPoint[0],midPoint[1]),(midPoint[0],midPoint[1]),(255,0,0),10)

#     roi_gray = gray[y:y+h, x:x+w]
#     roi_color = img[y:y+h, x:x+w]
#     eyes = eye_cascade.detectMultiScale(roi_gray)
#     temp1 = 0;
#     temp2 = 0;
#     count = 0;
#     for (ex,ey,ew,eh) in eyes:
#         cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)
#         temp1 = temp1 + ex + ew/2;
#         temp2 = temp2 + ey + eh/2;
#         count = count+1;
#         #pdb.set_trace();
#     midPoint[2] = x+ temp1/count;
#     midPoint[3] = y+ temp2/count;



#     cv2.line(img,(midPoint[2],midPoint[3]),(midPoint[2],midPoint[3]),(0,255,0),10)
#     #midPoint[2],midPoint[3]=midPoint[0],midPoint[1]

# # If the mouth detection does not work, adjust the second parameter inside the #detectMultiScale. It acts as a measure of confidence.

#     mouth = mouth_cascade.detectMultiScale(roi_gray,4,2)
#     for (mx,my,mw,mh) in mouth:
#         cv2.rectangle(roi_color,(mx,my),(mx+mw,my+mh),(0,0,255),2)
#     	midPoint[6] = mx + mw/2
#     	midPoint[7] = my + mh/2
# 	midPoint[6] = midPoint[6]+x;
#         midPoint[7] = midPoint[7]+y;
#         cv2.line(img,(midPoint[6],midPoint[7]),(midPoint[6],midPoint[7]),(0,0,255),10)

#     #midPoint[6],midPoint[7]=midPoint[0],midPoint[1]

    
    

#     nose = nose_cascade.detectMultiScale(roi_gray)
#     for (nx,ny,nw,nh) in nose:
#         cv2.rectangle(roi_color,(nx,ny),(nx+nw,ny+nh),(0,255,0),2)
#     	midPoint[4] = nx+nw/2;
#     	midPoint[5] = ny+nh/2;
#         midPoint[4] = midPoint[4]+x;
#         midPoint[5] = midPoint[5]+y;
#         cv2.line(img,(midPoint[4],midPoint[5]),(midPoint[4],midPoint[5]),(255,224,0),10)

# #for i in range (0,5):
# midPointDebug = midPoint.reshape(4,2);
# [vx,vy,x,y] = cv2.fitLine(midPointDebug, cv2.cv.CV_DIST_L1, 0, 0.01, 0.01)
# #-----Verify this does not cause any major error-------
# #pdb.set_trace();

# x = np.around(x)
# y = np.around(y)
# #-------------------------------------------------------
# #Hack just to get 2 points from the direction vectors so that we can plot the line
# distance = 100;
# x2 = x+distance*vx;
# y2 = y+distance*vy;

# x3 = x+(-1)*distance*vx;
# y3 = y+(-1)*distance*vy;

# x2 = np.around(x2)
# y2 = np.around(y2)

# cv2.line(img,(x3,y3),(x2,y2),(255,0,0),1)

# cv2.imshow('img',img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
# #pdb.set_trace()
