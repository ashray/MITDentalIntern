#!/usr/bin/python
#!/usr/local/bin/python
import numpy as np
import cv2
import pdb
import math
from shapely.geometry import LineString
from IntersectionPoint import *


def colGray(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    cv2.equalizeHist(gray, gray)
    return gray


def faceFeatureDetector(img):
    gray = colGray(img)
    count_face, count_mouth, count_nose, count = 0, 0, 0, 0
    midPoint = []
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')
    mouth_cascade = cv2.CascadeClassifier('haarcascade_mcs_mouth.xml')
    nose_cascade = cv2.CascadeClassifier('haarcascade_mcs_nose.xml')

    # eye_cascade = cv2.CascadeClassifier('eyes22x5.xml')
    # mouth_cascade = cv2.CascadeClassifier('mouth.xml')
    # nose_cascade = cv2.CascadeClassifier('nose18x15.xml')

    faces = face_cascade.detectMultiScale(gray, 1.3, 2)

    for (x, y, w, h) in faces:
        count_face = count_face + 1
        if count_face == 1:
            # cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
            midPoint.append(x + w / 2)
            midPoint.append(y + h / 2)
            print midPoint[0:len(midPoint)]
            # pdb.set_trace()
            # Plot center of face
            cv2.line(img, (midPoint[0], midPoint[1]), (midPoint[0], midPoint[1]), (255, 0, 0), 6)
            curr_len = len(midPoint)
            intersection_x, intersection_y = midPoint[curr_len-2], midPoint[curr_len-1]

            roi_gray = gray[y:y + h, x:x + w]
            roi_color = img[y:y + h, x:x + w]

            eyes = eye_cascade.detectMultiScale(roi_gray)
            temp1 = 0
            temp2 = 0

            for (ex, ey, ew, eh) in eyes:
                # cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)
                temp1 = temp1 + ex + ew / 2
                temp2 = temp2 + ey + eh / 2
                count = count + 1
                # pdb.set_trace();

            if count == 2:
                midPoint.append(x + temp1 / count)
                midPoint.append(y + temp2 / count)
                curr_len = len(midPoint)
                # Plot center of eyes
                cv2.line(img, (midPoint[curr_len - 2], midPoint[curr_len - 1]), (midPoint[curr_len - 2], midPoint[curr_len - 1]), (0, 255, 0), 6)
                intersection_x, intersection_y = midPoint[curr_len-2], midPoint[curr_len-1]
                print midPoint[0:len(midPoint)]

            else:
                print 'Eyes != 2'


            # cv2.line(img,(midPoint[2],midPoint[3]),(midPoint[2],midPoint[3]),(0,255,0),10)
            # If the mouth detection does not work, adjust the second parameter inside the #detectMultiScale. It acts as a measure of confidence.

            mouth = mouth_cascade.detectMultiScale(roi_gray, 4, 2)
            for (mx, my, mw, mh) in mouth:
                count_mouth = count_mouth + 1

            if count_mouth == 1:
                # cv2.rectangle(roi_color, (mx, my), (mx + mw, my + mh), (0, 0, 255), 2)
                midPoint.append(mx + mw / 2 + x)
                midPoint.append(my + mh / 2 + y)
                curr_len = len(midPoint)
                # Plot center of mouth
                cv2.line(img, (midPoint[curr_len - 2], midPoint[curr_len - 1]), (midPoint[curr_len - 2], midPoint[curr_len - 1]), (0, 0, 255), 6)
                print midPoint[0:len(midPoint)]

            else:
                print 'Mouth != 1'

            nose = nose_cascade.detectMultiScale(roi_gray)
            for (nx, ny, nw, nh) in nose:
                count_nose = count_nose + 1

            if count_nose == 1:
                # cv2.rectangle(roi_color, (nx, ny), (nx + nw, ny + nh), (0, 255, 0), 2)
                midPoint.append(nx + nw / 2 + x)
                midPoint.append(ny + nh / 2 + y)
                curr_len = len(midPoint)
                # Plot center of nose
                cv2.line(img, (midPoint[curr_len - 2], midPoint[curr_len - 1]), (midPoint[curr_len - 2], midPoint[curr_len - 1]), (255, 224, 0), 6)
                print midPoint[0:len(midPoint)]

            else:
                print 'Nose != 1'

        else:
            print 'Face != 1'



    return midPoint, x, y, w, h, intersection_x, intersection_y


# Resize midPoint numpy array
def draw_line(img, midPointDebug):
    row, col = midPointDebug.shape
    if row<col:
        midPointDebug = np.transpose(midPointDebug)
    else:
        pass
    [vx, vy, xbf, ybf] = cv2.fitLine(midPointDebug, cv2.cv.CV_DIST_L1, 0, 0.01, 0.01)
    # -----Verify this does not cause any major error-------
    # pdb.set_trace();

    xbf = np.around(xbf)
    ybf = np.around(ybf)
    # -------------------------------------------------------
    # Hack just to get 2 points from the direction vectors so that we can plot the line
    distance = 400
    x2 = xbf + distance * vx
    y2 = ybf + distance * vy

    x3 = xbf + (-1) * distance * vx
    y3 = ybf + (-1) * distance * vy

    x2 = np.around(x2)
    y2 = np.around(y2)

    cv2.line(img, (x3, y3), (x2, y2), (255, 0, 0), 1)

    return xbf, ybf, vx, vy

# def skin_detector(img_copy, x, y, w, h, xbf, ybf, intersection_x, intersection_y, vx, vy):

    # sub_image1 = img_copy[y:y + h, x:xbf]
    # sub_image2 = img_copy[y:y + h, xbf:x + w]
    #
    # # cv2.imshow('sub image 1', sub_image1)
    # # cv2.imshow('sub image 2', sub_image2)
    #
    # # define the upper and lower boundaries of the HSV pixel
    # # intensities to be considered 'skin'
    # lower = np.array([0, 48, 80], dtype="uint8")
    # upper = np.array([20, 255, 255], dtype="uint8")
    #
    # converted_sub1 = cv2.cvtColor(sub_image1, cv2.COLOR_BGR2HSV)
    # converted_sub2 = cv2.cvtColor(sub_image2, cv2.COLOR_BGR2HSV)
    # converted_ori = cv2.cvtColor(img_copy, cv2.COLOR_BGR2HSV)
    #
    # # cv2.imshow('sub image 1', converted_sub1)
    # # cv2.imshow('sub image 2', converted_sub2)
    #
    # skinMask_sub1 = cv2.inRange(converted_sub1, lower, upper)
    # skinMask_sub2 = cv2.inRange(converted_sub2, lower, upper)
    # skinMask_ori = cv2.inRange(converted_ori, lower, upper)
    #
    # kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
    # skinMask_sub1 = cv2.erode(skinMask_sub1, kernel, iterations=2)
    # skinMask_sub1 = cv2.dilate(skinMask_sub1, kernel, iterations=2)
    # skinMask_sub2 = cv2.erode(skinMask_sub2, kernel, iterations=2)
    # skinMask_sub2 = cv2.dilate(skinMask_sub2, kernel, iterations=2)
    #
    # skinMask_ori = cv2.erode(skinMask_ori, kernel, iterations=2)
    # skinMask_ori = cv2.dilate(skinMask_ori, kernel, iterations=2)
    #
    # # cv2.imshow('sub image 1', skinMask_sub1)
    # # cv2.imshow('sub image 2', skinMask_sub2)
    #
    # #        cv2.equalizeHist(sub_image1,sub_image1)
    # #        cv2.equalizeHist(sub_image2,sub_image2)
    #
    # sum_image1 = cv2.sumElems(skinMask_sub1)
    # sum_image2 = cv2.sumElems(skinMask_sub2)
    # # pdb.set_trace();
    # print 'Sum image 1', sum_image1
    # print 'Sum image 2', sum_image2
    #
    
    
#    for i in range (y,y+h):
#        for j in range (x,x+w):
#            if skinMask_ori[i,j]==0:
#                left_min=i
#                break

    # [vx_perpen,vy_perpen] = Perpendicular([vx,vy])
    #
    # cv2.line(img_copy,(intersection_x,intersection_y),(intersection_x+(100*vx_perpen),intersection_y+100*vy_perpen), (255, 224, 0), 6)
    # cv2.line(img_copy,(intersection_x,intersection_y),(intersection_x-(100*vx_perpen),intersection_y-100*vy_perpen), (255, 224, 0), 6)
    #
    # return sum_image1, sum_image2, img_copy