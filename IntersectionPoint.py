import numpy as np
import math
import pdb

def Perpendicular(a) :
    b = np.empty_like(a)
    b[0] = -a[1]
    b[1] = a[0]
    return b

# line segment a given by endpoints a1, a2
# line segment b given by endpoints b1, b2
# return
def LineSegmentIntersection(a1,a2, b1,b2) :
    da = a2-a1
    db = b2-b1
    dp = a1-b1
    dap = Perpendicular(da)
    denom = np.dot( dap, db)
    num = np.dot( dap, dp )
    return (num / denom.astype(float))*db + b1

def lineMagnitude (x1, y1, x2, y2):
    lineMagnitude = math.sqrt(math.pow((x2 - x1), 2)+ math.pow((y2 - y1), 2))
    return lineMagnitude

# Calc minimum distance from a point and a line segment (i.e. consecutive vertices in a polyline).
def DistancePointLine (px, py, x1, y1, x2, y2):
    # http://local.wasp.uwa.edu.au/~pbourke/geometry/pointline/source.vba
    LineMag = lineMagnitude(x1, y1, x2, y2)

    if LineMag < 0.00000001:
        DistancePointLine = 9999
        return DistancePointLine

    u1 = (((px - x1) * (x2 - x1)) + ((py - y1) * (y2 - y1)))
    u = u1 / (LineMag * LineMag)

    if (u < 0.00001) or (u > 1):
        # closest point does not fall within the line segment, take the shorter distance
        # to an endpoint
        ix = lineMagnitude(px, py, x1, y1)
        iy = lineMagnitude(px, py, x2, y2)
        if ix > iy:
            DistancePointLine = iy
        else:
            DistancePointLine = ix
    else:
        # Intersecting point is on the line, use the formula
        ix = x1 + u * (x2 - x1)
        iy = y1 + u * (y2 - y1)
        DistancePointLine = lineMagnitude(px, py, ix, iy)
    return DistancePointLine

# def assignWeights(curve_midpoints,feature_midpoints):
#     curve_midpoints_list_left = np.array(curve_midpoints[0,:]).tolist()
#     curve_midpoints_list_right = np.array(curve_midpoints[1,:]).tolist()
#     feature_midpoints_list = np.array(feature_midpoints).tolist()
#     for j in range(0,(len(feature_midpoints_list)-1),2):
#         for i in range (0,50):
#             curve_midpoints_list_left.append(feature_midpoints_list[j])
#             curve_midpoints_list_right.append(feature_midpoints_list[j+1])
#     curve_midpoints_list = curve_midpoints_list_left + curve_midpoints_list_right
#     curve_midpoints2 = np.asarray(curve_midpoints_list)
#
#     pdb.set_trace()
#
#     return curve_midpoints2

# We give an array of points as input and we want to return the point which is the closest to that line
def CurveLineIntersection(curve_points, linePoint1, linePoint2):
    # x coordinate in curve
    row, col = curve_points.shape
    if col<row:
        curve_points = np.transpose(curve_points)
    else:
        pass
    # this will make sure that col>row, which means that it'll be like 2*x
    point_count = len(curve_points[0, :])
    # min should be as large as possible. Also chosen_point_index should be a value different from 1000 at the end of the program
    min = 10000
    chosen_point_index = 1000
    for i in range(0, point_count-1):
        point = curve_points[:,i]
        dist = DistancePointLine(point[0], point[1], linePoint1[0], linePoint1[1], linePoint2[0], linePoint2[1])
        if dist<min:
            chosen_point_index = i
            min = dist
    return curve_points[:,chosen_point_index]

def FaceSymmetryLineIntersection(face_boundary_points, linePoint1, linePoint2):
    row, col = face_boundary_points.shape
    if col<row:
        face_boundary_points = np.transpose(face_boundary_points)
    else:
        pass
    points_count = len(face_boundary_points[0,:])
    left_boundary = face_boundary_points[:,0:((points_count/2)-1)]
    right_boundary = face_boundary_points[:, (points_count/2):(points_count-1)]
    left_intersection_point = CurveLineIntersection(left_boundary, linePoint1, linePoint2)
    right_intersection_point = CurveLineIntersection(right_boundary, linePoint1, linePoint2)
    return left_intersection_point, right_intersection_point