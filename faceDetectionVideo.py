#!/usr/bin/env python

import numpy as np
import cv2
import os
import pdb
from image_downscale import image_downscale
from violaJones import *
from canny import *
from symmetryMidpoints import symmetryMidpoints
from symmetryCalculation import symmetryCalculationIntensity
from helperFunctions import *
from faceMorpher import *
from docopt import docopt

# help_message = '''
# USAGE: facedetect.py [--cascade <cascade_fn>] [--nested-cascade <cascade_fn>] [<video_source>]
# '''

def detect(img, cascade):
    rects = cascade.detectMultiScale(img, scaleFactor=1.3, minNeighbors=4, minSize=(30, 30), flags = cv2.CASCADE_SCALE_IMAGE)
    if len(rects) == 0:
        return []
    rects[:,2:] += rects[:,:2]
    return rects

def draw_rects(img, rects, color):
    for x, y, w, h in rects:
        cv2.rectangle(img, (x, y), (w, h), color, 2)

# def faceDetectionVideo():
if __name__ == '__main__':
    # array_name = ["L1.mov","L2.mov","L3.mov","L4.mov","L5.mov","L6.mov","L7.mov","L8.mov","L9.mov", "R1.mov", "R2.mov"
    #     "R3.mov", "", "L10.mov", "L11.mov"]
    array_name = ["R2.mov"]
#     "L7.mov",
#     "L8.mov",
#     "L9.mov",
#     "L10.mov",
#     "R1.mov",
# ]
    for loop_iterator in range(1):
        face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
        eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')
        mouth_cascade = cv2.CascadeClassifier('haarcascade_mcs_mouth.xml')
        nose_cascade = cv2.CascadeClassifier('haarcascade_mcs_nose.xml')
        video_path = "/Users/me/Desktop/MITREDX/MITDentalIntern/photo/" + array_name[loop_iterator]
        cam = cv2.VideoCapture(video_path)
        # Change the while True to while there are still frames to read from!!
        # all_points = np.arange(2*77*400).reshape(77,2,400)
        frame_limit = 180
        all_points = np.zeros((77,2,frame_limit),dtype=np.int)
        j = 0
        feature_landmarks_central = 56
        feature_landmarks_one = [18, 21, 30, 58, 59]
        feature_landmarks_two = [25, 22, 40, 54, 65]
        while j < frame_limit:
            ret, img = cam.read()
            # if (img==):
            #     break

            cv2.imwrite('temporary_image.png',img)
            directoryLocation = os.path.dirname(os.path.abspath(__file__))
            imageLocation = directoryLocation + '/temporary_image.png'
            img,points = landmark_locator(imageLocation,  width=500, height=600, fps=10)
            #eyebrows
            cv2.line(img, (points[18][0],points[18][1]), (points[18][0],points[18][1]), (255,224,0),5)
            cv2.line(img, (points[21][0],points[21][1]), (points[21][0],points[21][1]), (255,224,0),5)
            cv2.line(img, (points[22][0],points[22][1]), (points[22][0],points[22][1]), (255,224,0),5)
            cv2.line(img, (points[25][0],points[25][1]), (points[25][0],points[25][1]), (255,224,0),5)
            #eyes
            cv2.line(img, (points[30][0],points[30][1]), (points[30][0],points[30][1]), (255,0,0),5)
            cv2.line(img, (points[40][0],points[40][1]), (points[40][0],points[40][1]), (255,0,0),5)
            #nose
            cv2.line(img, (points[54][0],points[54][1]), (points[54][0],points[54][1]), (0,255,0),5)
            cv2.line(img, (points[56][0],points[56][1]), (points[56][0],points[56][1]), (0,255,0),5)
            cv2.line(img, (points[58][0],points[58][1]), (points[58][0],points[58][1]), (0,255,0),5)
            #mouth
            cv2.line(img, (points[59][0],points[59][1]), (points[59][0],points[59][1]), (0,0,255),5)
            cv2.line(img, (points[65][0],points[65][1]), (points[65][0],points[65][1]), (0,0,255),5)


            midpoints = np.zeros((6,2),dtype=np.int)
            midpoints[0][0] = points[feature_landmarks_central][0]
            midpoints[0][1] = points[feature_landmarks_central][1]

            for i in range(1,len(feature_landmarks_one)+1):
                midpoints[i][0] = points[feature_landmarks_one[i-1]][0] + (points[feature_landmarks_two[i-1]][0] - points[feature_landmarks_one[i-1]][0])/2
                midpoints[i][1] = points[feature_landmarks_one[i-1]][1] + (points[feature_landmarks_two[i-1]][1] - points[feature_landmarks_one[i-1]][1])/2

            # pdb.set_trace()
            [vx, vy, xbf, ybf] = cv2.fitLine(midpoints, cv2.cv.CV_DIST_L1, 0, 0.01, 0.01)

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
            cv2.imshow("img",img)


            # a - point number(in the 77 range)
            # b - 0 or 1 for x and y
            # c - frame number0

            # the way we read the all_points
            # all_points[point location(in 77)][]
            for i in range (0,77):
                # dstack
                cv2.line(img, (points[i][0],points[i][1]), (points[i][0],points[i][1]), (0,255,0),5)
                all_points[i][0][j] = points[i][0]
                all_points[i][1][j] = points[i][1]
                # if j==1:
                #     store_values = points
                #     # store_values = np.dstack((store_values, points))
                # else:
                #     store_values = np.vstack((points, store_values))
                # c = np.dstack(())
                # all_points[:][0] = np.append(all_points[:][0],points[i][0])
                # all_points[:][1] = np.append(all_points[:][1],points[i][1])
            # all_points.astype(int)

            # vis = img.copy()
            # img2 = img.copy()
            # gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            # gray = cv2.equalizeHist(gray)
            # rects = detect(gray, face_cascade)
            # draw_rects(vis, rects, (0, 255, 0))
            #
            # x,y,w,h=rects[0][0],rects[0][1],rects[0][2],rects[0][3]
            # # pdb.set_trace()
            # w = w-x
            # h = h-y
            # # -------------Complete the code below later to not search on full frame----------
            #
            # x,y,w,h=rects[0][0],rects[0][1],rects[0][2],rects[0][3]
            # # pdb.set_trace()
            # w = w-x     # actual width
            # h = h-y     # actual height
            # # face_midpoint_x = x + w/2
            # # face_midpoint_y = y + h/2
            # # cv2.line(vis, (face_midpoint_x,face_midpoint_y), (face_midpoint_x,face_midpoint_y), (0,0,255),10)
            #
            # # -------------Complete the code below later to not search on full frame----------
            # # try:
            # #     rects[0][0]=max(rects[0][0]-10,0)
            # #     rects[0][1]=max(rects[0][1]-10,0)
            # #     rects[0][2]=max(rects[0][2]+10,0)
            # #     rects[0][3]=max(rects[0][3]+10,0)
            # # except IndexError:
            # #     rects[0][0]=0
            # #     rects[0][1]=0
            # #     rects[0][2]=0
            # #     rects[0][3]=0
            # #     print 'Face not detected'
            # #     break
            # #
            # # face_roi = vis[rects[0][1]:rects[0][3], rects[0][0]:rects[0][2]]
            # # ----------------------------------------------------------------------
            #
            # y_max = y+h
            #
            # a = FindEdgeImage(img2[max((y-h),0):y + 2*h, max((x-w),0):x+2*w])
            # # a_new =
            # width,length = a.shape
            # k=0
            # anew = np.arange(width*length).reshape(width, length)
            # if width==2:
            #     for g in range(length):
            #         if a[1][g] > y_max or a[1][g] < y:
            #             a[1][g] = 0
            #             a[0][g] = 0
            #         else:
            #             anew[1][k] = a[1][g]
            #             anew[0][k] = a[0][g]
            #             k=k+1
            # else:
            #     print "Width not 2"
            #     pdb.set_trace()
            #
            # # CHANGE THIS CODE!!LIKE REALLY, THIS IS BAD! REALLY BAD!!!!
            # anew2 = np.arange(width*k).reshape(width, k)
            # for m in range(k):
            #     anew2[1][m] = anew[1][m]
            #     anew2[0][m] = anew[0][m]
            # margin_size = 8
            # # pdb.set_trace()
            # anew2 = anew2+margin_size
            # midpoints = symmetryMidpoints((anew2),img2,x,y)
            # # midpoints = midpoints + margin_size
            # # midpoints = midpoints+margin_size
            # height, width, depth = img.shape
            # # img_cropped = vis[margin_size:(height-margin_size), margin_size:(width-margin_size)]
            # xbf_temp, ybf_temp, vx_temp, vy_temp = draw_line(vis,midpoints)
            #
            # # # another symmetry line using direction vectors of the bestfit line but passing through the face bounding box center
            # # distance = 400
            # # x1 = face_midpoint_x + distance * vx_temp
            # # y1 = face_midpoint_y + distance * vy_temp
            # #
            # # x2 = face_midpoint_x + (-1) * distance * vx_temp
            # # y2 = face_midpoint_y + (-1) * distance * vy_temp
            # # cv2.line(vis, (x1,y1), (x2,y2), (0,0,255),1)
            #
            # # vis = PlotPoints(midpoints,vis, 0, 0)
            # vis = PlotPoints(anew2,vis, 0, 0)
            #
            # # [vx_perpen,vy_perpen] = Perpendicular([vx_temp,vy_temp])
            #
            # distance = 400
            # x1 = points[56][0] + distance * vx_temp
            # y1 = points[56][1] + distance * vy_temp
            # x2 = points[56][0] + (-1) * distance * vx_temp
            # y2 = points[56][1] + (-1) * distance * vy_temp
            # cv2.line(vis, (x1,y1), (x2,y2), (0,0,255),1)
            #
            # # cv2.imshow('new midpoints',vis)
            # # cv2.waitKey(0)
            # os.remove(imageLocation)
            # # pdb.set_trace()
            # print 'Face landmark points', points
            np.save(array_name[loop_iterator], all_points)
            print j
            j = j+1
            # break

    #point no. 56 = nose base center point '/Users/me/Desktop/MITREDX/MITDentalIntern/photo/sampleFaceImage11.JPG'
    pdb.set_trace()
    cv2.destroyAllWindows()

