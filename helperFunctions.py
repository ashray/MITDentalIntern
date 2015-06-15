import numpy as np
import math
import pdb

def find_x(y, point1, point2):
    x1 = point1[0]
    y1 = point1[1]
    x2 = point2[0]
    y2 = point2[1]

    # for a vertical line. Return either of the two points
    if abs(x2-x1) < 2:
        return x1
    elif abs(y2-y1) == 0:
        return "Can not find intersection"
    else:
        m = (y2-y1)/(x2-x1)
        x = (y-(y1-((y2-y1)/(x2-x1))*x1))/m
        return x
    # y = mx + c

def dot_product(arr1, arr2):
    return arr1[0]*arr2[0] + arr1[1]*arr2[1]