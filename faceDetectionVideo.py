#!/usr/bin/env python

import numpy as np
import cv2
import cProfile
import pdb
from image_downscale import image_downscale
from violaJones import *
from canny import *
from symmetryMidpoints import symmetryMidpoints
from symmetryCalculation import symmetryCalculationIntensity
from helperFunctions import *


def detect(img, cascade):
    rects = cascade.detectMultiScale(img, scaleFactor=1.3, minNeighbors=4, minSize=(30, 30), flags = cv2.CASCADE_SCALE_IMAGE)
    if len(rects) == 0:
        return []
    rects[:,2:] += rects[:,:2]
    return rects

def draw_rects(img, rects, color):
    for x, y, w, h in rects:
        cv2.rectangle(img, (x, y), (w, h), color, 2)

if __name__ == '__main__':
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')
    mouth_cascade = cv2.CascadeClassifier('haarcascade_mcs_mouth.xml')
    nose_cascade = cv2.CascadeClassifier('haarcascade_mcs_nose.xml')
    video_path = "./photo/Equal.mov"
    cam = cv2.VideoCapture(video_path)
    # Change the while True to while there are still frames to read from!!
    while True:
        ret, img = cam.read()
        vis = img.copy()
        img2 = img.copy()
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)
        rects = detect(gray, face_cascade)
        draw_rects(vis, rects, (0, 255, 0))

        x,y,w,h=rects[0][0],rects[0][1],rects[0][2],rects[0][3]
        # pdb.set_trace()
        w = w-x
        h = h-y
        # -------------Complete the code below later to not search on full frame----------
        # try:
        #     rects[0][0]=max(rects[0][0]-10,0)
        #     rects[0][1]=max(rects[0][1]-10,0)
        #     rects[0][2]=max(rects[0][2]+10,0)
        #     rects[0][3]=max(rects[0][3]+10,0)
        # except IndexError:
        #     rects[0][0]=0
        #     rects[0][1]=0
        #     rects[0][2]=0
        #     rects[0][3]=0
        #     print 'Face not detected'
        #     break
        #
        # face_roi = vis[rects[0][1]:rects[0][3], rects[0][0]:rects[0][2]]
        # ----------------------------------------------------------------------

        y_max = y+h

        a = FindEdgeImage(img2[max((y-h),0):y + 2*h, max((x-w),0):x+2*w])
        # a_new =
        width,length = a.shape
        k=0
        anew = np.arange(width*length).reshape(width, length)
        if width==2:
            for g in range(length):
                if a[1][g] > y_max or a[1][g] < y:
                    a[1][g] = 0
                    a[0][g] = 0
                else:
                    anew[1][k] = a[1][g]
                    anew[0][k] = a[0][g]
                    k=k+1
        else:
            print "Width not 2"
            pdb.set_trace()

        # CHANGE THIS CODE!!LIKE REALLY, THIS IS BAD! REALLY BAD!!!!
        anew2 = np.arange(width*k).reshape(width, k)
        for m in range(k):
            anew2[1][m] = anew[1][m]
            anew2[0][m] = anew[0][m]
        margin_size = 8
        # pdb.set_trace()
        anew2 = anew2+margin_size
        midpoints = symmetryMidpoints((anew2),img2,x,y)
        # midpoints = midpoints + margin_size
        # midpoints = midpoints+margin_size
        height, width, depth = img.shape
        # img_cropped = vis[margin_size:(height-margin_size), margin_size:(width-margin_size)]
        xbf_temp, ybf_temp, vx_temp, vy_temp = draw_line(vis,midpoints)
        vis = PlotPoints(midpoints,vis, 0, 0)
        vis = PlotPoints(anew2,vis, 0, 0)

        # [vx_perpen,vy_perpen] = Perpendicular([vx_temp,vy_temp])
        cv2.imshow('new midpoints',vis)

        if 0xFF & cv2.waitKey(5) == 27:
            break
    cv2.destroyAllWindows()