import numpy as np
import cv2
import pdb

#Finds symmetry percentage considering face boundary and central symmetry line
# We need the input image, the face curve points(faceCurve) and the two intersection points(of the eye line)
def symmetryCalculationIntensity(face_boundary_points,input_img, eye_line_point1,eye_line_point2):
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
    for i in range(0, (points_count/2) - 1):
        if face_boundary_points[1, i] < left_boundary[1]:
            pass
        else:
            b = np.where(np.array(face_boundary_points[1,:])==left_boundary[1])
            c = face_boundary_points[0,b]
            d = np.where(np.array(face_boundary_points[1,:])==right_boundary[1])
            e = face.
            for j in range(face_boundary_points[1, b], )
            sum =