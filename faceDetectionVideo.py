#!/usr/bin/env python

import numpy as np
import cv2
import cProfile
import pdb
# local modules
from image_downscale import image_downscale
from violaJones import *
from canny import *
from symmetryMidpoints import symmetryMidpoints
from symmetryCalculation import symmetryCalculationIntensity
from helperFunctions import *

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
    for x1, y1, x2, y2 in rects:
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)

# def faceDetectionVideo():
if __name__ == '__main__':
    # import sys, getopt
    # # print help_message
    # #
    # video_src = getopt.getopt(sys.argv[1:])
    # try:
    #     video_src = video_src[0]
    # except:
    video_src = 0
    # args = dict(args)
    # cascade_fn = args.get('--cascade', "../../data/haarcascades/haarcascade_frontalface_alt.xml")
    # nested_fn  = args.get('--nested-cascade', "../../data/haarcascades/haarcascade_eye.xml")
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')
    mouth_cascade = cv2.CascadeClassifier('haarcascade_mcs_mouth.xml')
    nose_cascade = cv2.CascadeClassifier('haarcascade_mcs_nose.xml')
    i=1
    while (i>0):
        if (i==10):
            cam = create_capture(video_src, fallback='synth:bg=../data/lena.jpg:noise=0.05')
            ret, img = cam.read()
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            gray = cv2.equalizeHist(gray)
            # t = clock()
            rects = detect(gray, face_cascade)
            print '10th frame captured'
            break
        else:
            i=i+1

    # pdb.set_trace()
    print 'starting while'
    pdb.set_trace()

    while True:
        vis = img.copy()
        img_copy = img.copy()
        draw_rects(vis, rects, (0, 255, 0))
        try:
            rects[0][0]=max(rects[0][0]-10,0)
            rects[0][1]=max(rects[0][1]-10,0)
            rects[0][2]=max(rects[0][2]+10,0)
            rects[0][3]=max(rects[0][3]+10,0)
        except IndexError:
            rects[0][0]=0
            rects[0][1]=0
            rects[0][2]=0
            rects[0][3]=0
            print 'Face not detected'
        # x1,y1,x2,y2=rects[0][0]-10,rects[0][1]-10,rects[0][2]+10,rects[0][3]+10
        # print 'initializing x,y'
        for x1, y1, x2, y2 in rects:
            roi = gray[y1:y2, x1:x2]
            vis_roi = vis[y1:y2, x1:x2]
            subrects = detect(roi.copy(), eye_cascade)
            draw_rects(vis_roi, subrects, (255, 0, 0))
            subrects = detect(roi.copy(), nose_cascade)
            draw_rects(vis_roi, subrects, (0, 255, 0))
            subrects = detect(roi.copy(), mouth_cascade)
            draw_rects(vis_roi, subrects, (0, 0, 255))
            # gray_roi = gray[y1-10:y2+10,x1-10:x2+10]
        # pdb.set_trace()
        # dt = clock() - t
            cam = create_capture(video_src, fallback='synth:bg=../data/lena.jpg:noise=0.05')
            ret, img = cam.read()
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            gray = cv2.equalizeHist(gray)
        # t = clock()
            rects = detect(gray, face_cascade)
        # draw_str(vis, (20, 20), 'time: %.1f ms' % (dt*1000))
        # cv2.imshow('facedetect', vis)
        # img = image_downscale(img, 400)
        a = FindEdgeImage(vis[max((y1-y2),0):y1 + 2*y2, max((x1-x2),0):x1+2*x2])
        midpoints = symmetryMidpoints(a,vis,x1,y1)
        margin_size = 8
        height, width, depth  = img.shape
        img_cropped = vis[margin_size:(height-margin_size), margin_size:(width-margin_size)]
        xbf_temp, ybf_temp, vx_temp, vy_temp = draw_line(img_cropped,midpoints)
        # cv2.imshow('new midpoints fitline',img_cropped)
            # height, width, depth  = img_copy.shape
            # img_copy_cropped = img_copy[margin_size:(height-margin_size), margin_size:(width-margin_size)]
            #
        img_cropped = PlotPoints(midpoints,img_cropped, 0, 0)
        img_cropped = PlotPoints(a,img_cropped, 0, 0)
        cv2.imshow('new midpoints',img_cropped)

        if 0xFF & cv2.waitKey(5) == 27:
            break
        # return x1,y1,x2,y2,img
    cv2.destroyAllWindows()
    cProfile.run('faceDetectionVideo.py')
# x1,y1,x2,y2,img = faceDetectionVideo()
print 'back from func call'
