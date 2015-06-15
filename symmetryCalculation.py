import numpy as np
import cv2
import pdb
from IntersectionPoint import LineSegmentIntersection
from helperFunctions import find_x

#Finds symmetry percentage considering face boundary and central symmetry line
# We need the input image, the face curve points(faceCurve) and the two intersection points(of the eye line)
def symmetryCalculationIntensity(face_boundary_points,input_img, eye_line_point1_woNP,eye_line_point2_woNP, x_symmetry, y_symmetry, vx_symmetry, vy_symmetry):
    # We first need to get the points of the symmetry line
    symmetry_point1_woNP = [(x_symmetry - 200*vx_symmetry),(y_symmetry - 200*vy_symmetry)]
    symmetry_point2_woNP = [(x_symmetry + 200*vx_symmetry),(y_symmetry + 200*vy_symmetry)]

    # symmetry_point1_woNP = x_symmetry - 200*vx_symmetry,y_symmetry - 200*vy_symmetry
    # symmetry_point2_woNP = x_symmetry + 200*vx_symmetry,y_symmetry + 200*vy_symmetry

    # symmetry_point1 = np.array([(x_symmetry - 200*vx_symmetry),(y_symmetry - 200*vy_symmetry)])
    # symmetry_point2 = np.array([(x_symmetry + 200*vx_symmetry),(y_symmetry + 200*vy_symmetry)])

    # symmetry_point1 = np.array(symmetry_point1_woNP)
    # symmetry_point2 = np.array(symmetry_point2_woNP)

    symmetry_point1 = symmetry_point1_woNP
    symmetry_point2 = symmetry_point2_woNP

    # eye_line_point1 = np.array([(eye_line_point1[0]),(eye_line_point1[1])])
    # eye_line_point2 = np.array([(eye_line_point2[0]),(eye_line_point2[1])])

    # eye_line_point1 = np.array(eye_line_point1_woNP)
    # eye_line_point2 = np.array(eye_line_point2_woNP)
    #
    # # so now a hack - which also didn't work !
    # eye_line_point1 = np.transpose(eye_line_point1)
    # eye_line_point2 = np.transpose(eye_line_point2)

    eye_line_point1 = eye_line_point1_woNP
    eye_line_point2 = eye_line_point2_woNP

    # we come down in y axis
    row, col, depth = input_img.shape
    if col<row:
        input_img = np.transpose(input_img)
    else:
        pass
    # theta = 0
    # pdb.set_trace()
    # We know that eye_line_point1 is going to be on the left side of the face
    points_count = len(face_boundary_points[0,:])
    left_boundary = face_boundary_points[:,0:((points_count/2)-1)]
    right_boundary = face_boundary_points[:, (points_count/2):(points_count-1)]

    # We need to scan all the points on the left boundary which have y value greater than eye_line_point1
    sum = 0
    symmetry_point1 = np.asarray(symmetry_point1)
    symmetry_point2 = np.asarray(symmetry_point2)
    symmetry_point1 = symmetry_point1.astype(int)
    symmetry_point2 = symmetry_point2.astype(int)

    # symmetry_perpendicular_intersection = LineSegmentIntersection([int(symmetry_point1[0]), int(symmetry_point1[1])], [int(symmetry_point2[0]), int(symmetry_point2[1])], eye_line_point1, eye_line_point2)
    eye_line_point1 = np.asarray(eye_line_point1)
    eye_line_point2 = np.asarray(eye_line_point2)
    eye_line_point1 = eye_line_point1.reshape((2,1))
    eye_line_point2 = eye_line_point2.reshape((2,1))
    eye_line_point1 = eye_line_point1.astype(int)
    eye_line_point2 = eye_line_point2.astype(int)
    # pdb.set_trace()
    symmetry_perpendicular_intersection = LineSegmentIntersection(symmetry_point1, symmetry_point2, eye_line_point1, eye_line_point2)

    symmetry_perpendicular_intersection = np.asarray(symmetry_perpendicular_intersection)
    symmetry_perpendicular_intersection = symmetry_perpendicular_intersection.astype(int)

    sum_left = 0
    sum_right = 0
    for i in range(0, (points_count/2) - 1):
        # The above loop will run if eye_line_point1[1]< eye_line_point2[1]
        if (eye_line_point1[1]<eye_line_point2[1]) or (eye_line_point1[1] == eye_line_point2[1]):
            if face_boundary_points[1, i] < eye_line_point1[1]:
                print 'pass ',i
                pass
            elif (face_boundary_points[1,i] < symmetry_perpendicular_intersection[1]) and (eye_line_point1[1] < face_boundary_points[1,i]):
                xf = find_x(face_boundary_points[1],eye_line_point1, eye_line_point2)
                sum_left = sum_left + np.sum(input_img[face_boundary_points[0,i]:xf, face_boundary_points[1,i]])
                print 'cond 1'
            elif ((symmetry_perpendicular_intersection[1]<face_boundary_points[1,i]) and (face_boundary_points[1,i]<eye_line_point2[1])):
                xf = find_x(face_boundary_points[1], eye_line_point1, eye_line_point2)
                xg = find_x(face_boundary_points[1], symmetry_point1, symmetry_point2)
                sum_left = sum_left + np.sum(input_img[face_boundary_points[0,i]:xg, face_boundary_points[1,i]])
                sum_right = sum_right + np.sum(input_img[xg:xf, face_boundary_points[1,i]])
                print 'cond 2'

            elif face_boundary_points[1,i]>eye_line_point2[1]:
                xg = find_x(face_boundary_points[1], symmetry_point1, symmetry_point2)
                sum_left = sum_left + np.sum(input_img[face_boundary_points[0,i]:xg, face_boundary_points[1,i]])
                sum_right = sum_right + np.sum(input_img[xg:right_boundary[0,i], face_boundary_points[1,i]])
                # sum_right = sum_right + np.sum(input_img[xg:face_boundary_points[1,i], face_boundary_points[1,i]])
                print 'cond 3'

            # total_sum = sum_left + sum_right
    #         # pdb.set_trace()set_trace
    # left_percentage = (sum_left/total_sum)*100
    # right_percentage = (sum_right/total_sum)*100



        # For the case where eye_line_point1[1]>=eye_line_point2[1]:
        else:
            print 'Case 2 - left tilted face'
            if face_boundary_points[1, i] < eye_line_point2[1]:
                pass
            elif (face_boundary_points[1,i] < symmetry_perpendicular_intersection[1]) and (eye_line_point2[1] < face_boundary_points[1,i]) :
                xf = find_x(face_boundary_points[1], eye_line_point1, eye_line_point2)
                sum_right = sum_right + np.sum(input_img[xf:right_boundary[0,i], face_boundary_points[1,i]])
                #
                # xg = find_x(face_boundary_points[1], symmetry_point1, symmetry_point2)
                # sum_left = sum_left + np.sum(input_img[face_boundary_points[0,i]:xg, face_boundary_points[1,i]])
                # sum_right = sum_right + np.sum(input_img[xg:xf, face_boundary_points[1,i]])
            elif ((symmetry_perpendicular_intersection[1]<face_boundary_points[1,i]) and (face_boundary_points[1,i]<eye_line_point1[1])):
                xf = find_x(face_boundary_points[1],eye_line_point1, eye_line_point2)
                xg = find_x(face_boundary_points[1], symmetry_point1, symmetry_point2)
                sum_left = sum_left + np.sum(input_img[xf:xg, face_boundary_points[1,i]])
                sum_right = sum_right + np.sum(input_img[xg:right_boundary[0,i], face_boundary_points[1,i]])
            elif face_boundary_points[1,i]>eye_line_point1[1]:
                xg = find_x(face_boundary_points[1], symmetry_point1, symmetry_point2)
                # pdb.set_trace()
                sum_left = sum_left + np.sum(input_img[face_boundary_points[0,i]:xg, face_boundary_points[1,i]])
                sum_right = sum_right + np.sum(input_img[xg:right_boundary[0,i], face_boundary_points[1,i]])

    total_sum = sum_left + sum_right
    # pdb.set_trace()
    left_percentage = (sum_left/total_sum)*100
    right_percentage = (sum_right/total_sum)*100

    return left_percentage, right_percentage

