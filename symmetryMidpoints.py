import cv2
import numpy as np
import pdb
import math
from canny import *
from numpy import linalg as LA

# Calculates distance between corresponding points on face curve to get an array of points for drawing central face symmetry line

def symmetryMidpoints(a, img, x, y):
    print "Shape of a ", a.shape
    # a[0,:] = a[0,:] + x
    # a[1,:] = a[1,:] + y
    num = len(a[0,:])
    print 'len of a ', num
    # correspondingPointsDistance = np.arange(num).reshape((2,num/2))


    # midPointNP = np.asarray(midPoint)
    #
    # SymmetryLinePoints = midPointNP.reshape(len_list, 2)
    loopIterationCount = num/2
    correspondingPointsDistance = []
    # num = num - 1
    for i in range(0,(loopIterationCount-1)):
        # correspondingPointsDistance.append((1/2)*(math.sqrt((a[0,i]-a[0,num-1-i])*(a[0,i]-a[0,num-1-i]) + (a[1,i]-a[1,num-1-i])*(a[1,i]-a[1,num-1-i]))))   #complete this
        correspondingPointsDistance.append((1/2)*(a[0,i]-a[0,num-1-i]))
    # num = num + 1
    lenDistanceArray = len(correspondingPointsDistance)
    print 'Length of the correspondingPointsDistance array is ', lenDistanceArray
    # correspondingPointsDistanceArray = np.arange(lenDistanceArray)
    correspondingPointsDistanceArray = np.asarray(correspondingPointsDistance)

    # print 'len of correspondingPointsDistance', lenDistanceArray
    # lenDistanceArray = lenDistanceArray/2
    # print 'len of midpoints2', lenDistanceArray
    # correspondingPointsDistance = correspondingPointsDistanceArray.reshape(lenDistanceArray,2)
    # print 'len of correspondingPointsDistance', lenDistanceArray

    # correspondingPointsDistance has the distance between consecutive points stored in it
    # we will need to add correspondingPointsDistance/2 to the left side of the face edges(which we get in a)
    symmetryPoints = np.arange((2*loopIterationCount))
    symmetryPoints = symmetryPoints.reshape(2,loopIterationCount)
    symmetryPoints[0, 0:(loopIterationCount-1)] = a[0,0:loopIterationCount-1] + correspondingPointsDistanceArray/2
    symmetryPoints[1, 0:(loopIterationCount-1)] = a[1,0:(loopIterationCount-1)]
    return symmetryPoints
