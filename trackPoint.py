import cv2
from helperFunctions import *
from faceMorpher import landmark_locator
import matplotlib.pyplot as plt
import os


videoLocation_r = '/Users/me/Desktop/MITDentalData/Videos/9r.mov'
videoLocation_l = '/Users/me/Desktop/MITDentalData/Videos/9l.mov'
cam_r = cv2.VideoCapture(videoLocation_r)
cam_l = cv2.VideoCapture(videoLocation_l)
frame_limit,j = 80,0
frame_limit = int(cam_r.get(cv2.cv.CV_CAP_PROP_FRAME_COUNT))
poi = [59, 8]#poi[0], poi[1]
x, y =np.zeros((2,2,frame_limit),dtype=np.int), np.zeros((2,2,frame_limit),dtype=np.int)
t = np.arange(0, frame_limit, 1)
pdb.set_trace()
while j < frame_limit:
    ret, img = cam_r.read()

    cv2.imwrite('temporary_image.png',img)
    directoryLocation = os.path.dirname(os.path.abspath(__file__))
    imageLocation = directoryLocation + '/temporary_image.png'
    img,points = landmark_locator(imageLocation,  width=500, height=600, fps=10)
    x[0][0][j] = points[poi[0]][0] - points[52][0]
    print j, x[0][0]
    j+=1

plt.plot(t, x[0][0], '-r', label='Point poi[0], Right')

frame_limit,j = 80,0
frame_limit = int(cam_l.get(cv2.cv.CV_CAP_PROP_FRAME_COUNT))
t = np.arange(0, frame_limit, 1)
poi = [59, 8]
x, y =np.zeros((2,2,frame_limit),dtype=np.int), np.zeros((2,2,frame_limit),dtype=np.int)
while j < frame_limit:
    ret, img = cam_l.read()

    cv2.imwrite('temporary_image.png',img)
    directoryLocation = os.path.dirname(os.path.abspath(__file__))
    imageLocation = directoryLocation + '/temporary_image.png'
    img,points = landmark_locator(imageLocation,  width=500, height=600, fps=10)
    x[0][0][j] = points[poi[0]][0] - points[52][0] #+ vx*dist/2
    print j, x[0][0] #, y[0][1], x[1][0], y[1][1]
    j+=1

plt.plot(t, x[0][0], '-b', label='Point poi[0], Left')
plt.show()
