import numpy as np
import cv2
import pdb
from IntersectionPoint import LineSegmentIntersection
from helperFunctions import find_x

#Finds symmetry percentage considering face boundary and central symmetry line
# We need the input image, the face curve points(faceCurve) and the two intersection points(of the eye line)
def symmetryCalculationIntensity(face_boundary_points,input_img, eye_line_point1,eye_line_point2, x_symmetry, y_symmetry, vx_symmetry, vy_symmetry):
    # We first need to get the points of the symmetry line
    symmetry_point1 = [(x_symmetry - 200*vx_symmetry),(y_symmetry - 200*vy_symmetry)]
    symmetry_point2 = [(x_symmetry + 200*vx_symmetry),(y_symmetry + 200*vy_symmetry)]

    # we come down in y axis
    row, col, depth = input_img.shape
    if col<row:
        input_img = np.transpose(input_img)
    else:
        pass
    # theta = 0

    # We know that eye_line_point1 is going to be on the left side of the face
    points_count = len(face_boundary_points[0,:])
    left_boundary = face_boundary_points[:,0:((points_count/2)-1)]
    right_boundary = face_boundary_points[:, (points_count/2):(points_count-1)]

    # We need to scan all the points on the left boundary which have y value greater than eye_line_point1
    sum = 0
    symmetry_perpendicular_intersection = LineSegmentIntersection(symmetry_point1, symmetry_point2, eye_line_point1, eye_line_point2)

    sum_left = 0
    sum_right = 0
    for i in range(0, (points_count/2) - 1):
        # The above loop will run if eye_line_point1[1]< eye_line_point2[1]
        if eye_line_point1[1]<eye_line_point2[1]:
            if face_boundary_points[1, i] < eye_line_point1[1]:
                pass
            elif (face_boundary_points[1,i] < symmetry_perpendicular_intersection[1]) and (eye_line_point1[1] < face_boundary_points[1,i]):
                xf = find_x(face_boundary_points[1],eye_line_point1, eye_line_point2)
                sum_left = sum_left + np.sum(input_img[face_boundary_points[0,i]:xf, face_boundary_points[1,i]])
            elif ((symmetry_perpendicular_intersection[1]<face_boundary_points[1,i]) and (face_boundary_points[1,i]<eye_line_point2[1])):
                xf = find_x(face_boundary_points[1], eye_line_point1, eye_line_point2)
                xg = find_x(face_boundary_points[1], symmetry_point1, symmetry_point2)
                sum_left = sum_left + np.sum(input_img[face_boundary_points[0,i]:xg, face_boundary_points[1,i]])
                sum_right = sum_right + np.sum(input_img[xg:xf, face_boundary_points[1,i]])
            elif face_boundary_points[1]>eye_line_point2[1]:
                xg = find_x(face_boundary_points[1], symmetry_point1, symmetry_point2)
                sum_left = sum_left + np.sum(input_img[face_boundary_points[0,i]:xg, face_boundary_points[1,i]])
                sum_right = sum_right + np.sum(input_img[xg:face_boundary_points[1,i], face_boundary_points[1,i]])

        # For the case where eye_line_point1[1]>=eye_line_point2[1]:
        else

