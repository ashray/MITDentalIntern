import cv2
import numpy as np
import pdb
from canny import *
from numpy import linalg as LA


def symmetryMidpoints(a, img, x, y):
    
    a[0,:] = a[0,:] + x
    a[1,:] = a[1,:] + y
    num = len(a[0,:])
    midpoints = np.arange(num).reshape((2,num/2))
    for i in range(0,num):
	midpoints.append((1/2)*distance...