import numpy as np
import cv2
import pdb
from IntersectionPoint import LineSegmentIntersection, Perpendicular, CurveLineIntersection, DistancePointLine, lineMagnitude
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
            elif (face_boundary_points[1, i] < symmetry_perpendicular_intersection[1]) and (
                        eye_line_point1[1] < face_boundary_points[1, i]):
                xf = find_x(face_boundary_points[1, i], eye_line_point1, eye_line_point2)
                sum_left = sum_left + np.sum(input_img[face_boundary_points[0, i]:xf, face_boundary_points[1, i]])
                print 'cond 1'
            elif ((symmetry_perpendicular_intersection[1] < face_boundary_points[1, i]) and (
                        face_boundary_points[1, i] < eye_line_point2[1])):
                xf = find_x(face_boundary_points[1, i], eye_line_point1, eye_line_point2)
                xg = find_x(face_boundary_points[1, i], symmetry_point1, symmetry_point2)
                sum_left = sum_left + np.sum(input_img[face_boundary_points[0, i]:xg, face_boundary_points[1, i]])
                sum_right = sum_right + np.sum(input_img[xg:xf, face_boundary_points[1, i]])
                print 'cond 2'

            elif face_boundary_points[1, i] > eye_line_point2[1]:
                xg = find_x(face_boundary_points[1, i], symmetry_point1, symmetry_point2)
                sum_left = sum_left + np.sum(input_img[face_boundary_points[0, i]:xg, face_boundary_points[1, i]])
                sum_right = sum_right + np.sum(input_img[xg:right_boundary[0, i], face_boundary_points[1, i]])
                # sum_right = sum_right + np.sum(input_img[xg:face_boundary_points[1,i], face_boundary_points[1,i]])
                print 'cond 3'
                # total_sum = sum_left + sum_right
                #         # pdb.set_trace()set_trace
                # left_percentage = (sum_left/total_sum)*100
                # right_percentage = (sum_right/total_sum)*100

            # total_sum = sum_left + sum_right
    #         # pdb.set_trace()set_trace
    # left_percentage = (sum_left/total_sum)*100
    # right_percentage = (sum_right/total_sum)*100



        # For the case where eye_line_point1[1]>=eye_line_point2[1]:
        else:
            print 'Case 2 - left tilted face'
            if face_boundary_points[1, i] < eye_line_point2[1]:
                pass
            elif (face_boundary_points[1, i] < symmetry_perpendicular_intersection[1]) and (
                        eye_line_point2[1] < face_boundary_points[1, i]):
                xf = find_x(face_boundary_points[1, i], eye_line_point1, eye_line_point2)
                sum_right = sum_right + np.sum(input_img[xf:right_boundary[0, i], face_boundary_points[1, i]])
                #
                # xg = find_x(face_boundary_points[1], symmetry_point1, symmetry_point2)
                # sum_left = sum_left + np.sum(input_img[face_boundary_points[0,i]:xg, face_boundary_points[1,i]])
                # sum_right = sum_right + np.sum(input_img[xg:xf, face_boundary_points[1,i]])
            elif ((symmetry_perpendicular_intersection[1] < face_boundary_points[1, i]) and (
                        face_boundary_points[1, i] < eye_line_point1[1])):
                xf = find_x(face_boundary_points[1, i], eye_line_point1, eye_line_point2)
                xg = find_x(face_boundary_points[1, i], symmetry_point1, symmetry_point2)
                sum_left = sum_left + np.sum(input_img[xf:xg, face_boundary_points[1, i]])
                sum_right = sum_right + np.sum(input_img[xg:right_boundary[0, i], face_boundary_points[1, i]])
            elif face_boundary_points[1, i] > eye_line_point1[1]:
                xg = find_x(face_boundary_points[1, i], symmetry_point1, symmetry_point2)
                # pdb.set_trace()
                print "face_boundary_points[0,i] = ", face_boundary_points[0,i]
                print "xg = ", xg
                sum_left = sum_left + np.sum(input_img[face_boundary_points[0,i]:xg, face_boundary_points[1,i]])
                sum_right = sum_right + np.sum(input_img[xg:right_boundary[0,i], face_boundary_points[1,i]])

    total_sum = sum_left + sum_right
    # pdb.set_trace()
    left_percentage = (sum_left/total_sum)*100
    right_percentage = (sum_right/total_sum)*100

    return left_percentage, right_percentage

def symmetryCalculationBoundaryDifference(face_boundary_points, input_img, eye_line_point1, eye_line_point2, x_symmetry,
                                y_symmetry, vx_symmetry, vy_symmetry):
    # We first need to get the points of the symmetry line
    symmetry_point1 = [(x_symmetry - 200 * vx_symmetry), (y_symmetry - 200 * vy_symmetry)]
    symmetry_point2 = [(x_symmetry + 200 * vx_symmetry), (y_symmetry + 200 * vy_symmetry)]

    # we come down in y axis
    # row, col, depth = input_img.shape
    # if col < row:
    #     input_img = np.transpose(input_img)
    # else:
    #     pass

    points_count = len(face_boundary_points[0, :])
    left_boundary = face_boundary_points[:, 0:((points_count / 2) - 1)]
    right_boundary = face_boundary_points[:, (points_count / 2):(points_count - 1)]

    # We need to scan all the points on the left boundary which have y value greater than eye_line_point1
    sum = 0
    symmetry_point1 = np.asarray(symmetry_point1)
    symmetry_point2 = np.asarray(symmetry_point2)
    symmetry_point1 = symmetry_point1.astype(int)
    symmetry_point2 = symmetry_point2.astype(int)

    eye_line_point1 = np.asarray(eye_line_point1)
    eye_line_point2 = np.asarray(eye_line_point2)
    eye_line_point1 = eye_line_point1.reshape((2, 1))
    eye_line_point2 = eye_line_point2.reshape((2, 1))
    eye_line_point1 = eye_line_point1.astype(int)
    eye_line_point2 = eye_line_point2.astype(int)
    # pdb.set_trace()
    symmetry_perpendicular_intersection = LineSegmentIntersection(symmetry_point1, symmetry_point2, eye_line_point1,
                                                                  eye_line_point2)

    symmetry_perpendicular_intersection = np.asarray(symmetry_perpendicular_intersection)
    symmetry_perpendicular_intersection = symmetry_perpendicular_intersection.astype(int)


    sum_left = 0
    sum_right = 0

    y_init = symmetry_perpendicular_intersection[1]
    y_jump = 1
    # from face_boundary_points find out the maximum value of y_max and replace it below
    y_max = max(face_boundary_points[1,:])
    y_value = y_init
    perpendicular_vectors = Perpendicular([vx_symmetry, vy_symmetry])
    DistanceDifference = []
    # print "Reached symmetryCalculationBoundaryDifference"
    for i in range(y_init, y_max, y_jump):
        x_value = find_x(y_value, symmetry_point1, symmetry_point2)
        perpendicular_point_1 = [(x_value - 200 * perpendicular_vectors[0]), (y_value - 200 * perpendicular_vectors[1])]
        perpendicular_point_2 = [(x_value + 200 * perpendicular_vectors[0]), (y_value + 200 * perpendicular_vectors[1])]
        left_intersection_point = CurveLineIntersection(left_boundary, perpendicular_point_1, perpendicular_point_2)
        right_intersection_point = CurveLineIntersection(right_boundary, perpendicular_point_1, perpendicular_point_2)
        dist_left = np.linalg.norm(left_intersection_point-np.asarray([x_value, y_value]).reshape(1,2))
        dist_right = np.linalg.norm(right_intersection_point-np.asarray([x_value, y_value]).reshape(1,2))
        leftRightDifference = dist_left - dist_right
        DistanceDifference.append(leftRightDifference)
    # print "Reached symmetryCalculationBoundaryDifference after for"
    pdb.set_trace()

    sumNegative = 0
    for number in DistanceDifference:
        if number < 0:
            sumNegative += number
    # print "Reached symmetryCalculationBoundaryDifference after second for"

    sumPositive = 0
    for number in DistanceDifference:
        if number >= 0:
            sumPositive += number

    # sumNegative = sum(number2 for number2 in DistanceDifference if number2 < 0)
    positivePercentage = 100*(sumPositive/(sumPositive+sumNegative))
    negativePercentage = 100*(sumNegative/(sumPositive+sumNegative))
    pdb.set_trace()
    return negativePercentage, positivePercentage

def symmetryCalculationLandmarkPoints(points, x_symmetry,y_symmetry,vx_symmetry,vy_symmetry):
    # cv2.line(img, (points[18][0],points[18][1]), (points[18][0],points[18][1]), (255,224,0),5)
    # cv2.line(img, (points[21][0],points[21][1]), (points[21][0],points[21][1]), (255,224,0),5)
    # cv2.line(img, (points[22][0],points[22][1]), (points[22][0],points[22][1]), (255,224,0),5)
    # cv2.line(img, (points[25][0],points[25][1]), (points[25][0],points[25][1]), (255,224,0),5)
    #
    # cv2.line(img, (points[30][0],points[30][1]), (points[30][0],points[30][1]), (255,0,0),5)
    # cv2.line(img, (points[40][0],points[40][1]), (points[40][0],points[40][1]), (255,0,0),5)
    #
    # cv2.line(img, (points[54][0],points[54][1]), (points[54][0],points[54][1]), (0,255,0),5)
    # cv2.line(img, (points[56][0],points[56][1]), (points[56][0],points[56][1]), (0,255,0),5)
    # cv2.line(img, (points[58][0],points[58][1]), (points[58][0],points[58][1]), (0,255,0),5)
    #
    # cv2.line(img, (points[59][0],points[59][1]), (points[59][0],points[59][1]), (0,0,255),5)
    # cv2.line(img, (points[65][0],points[65][1]), (points[65][0],points[65][1]), (0,0,255),5)

    # Difference in landmark points
    difference_array = []
    difference_array2 = []
    symmetry_point1 = np.asarray([(x_symmetry - 200 * vx_symmetry), (y_symmetry - 200 * vy_symmetry)])

    symmetry_point2 = np.asarray([(x_symmetry + 200 * vx_symmetry), (y_symmetry + 200 * vy_symmetry)])
    Dist_18 = DistancePointLine(points[18][0],points[18][1], symmetry_point1[0], symmetry_point1[1], symmetry_point2[0], symmetry_point2[1])
    Dist_25 = DistancePointLine(points[25][0],points[25][1], symmetry_point1[0], symmetry_point1[1], symmetry_point2[0], symmetry_point2[1])
    difference_array.append((Dist_18-Dist_25)/Dist_18)
    Dist_21 = DistancePointLine(points[21][0],points[21][1], symmetry_point1[0], symmetry_point1[1], symmetry_point2[0], symmetry_point2[1])
    Dist_22 = DistancePointLine(points[22][0],points[22][1], symmetry_point1[0], symmetry_point1[1], symmetry_point2[0], symmetry_point2[1])
    difference_array.append((Dist_21-Dist_22)/Dist_21)
    Dist_30 = DistancePointLine(points[30][0],points[30][1], symmetry_point1[0], symmetry_point1[1], symmetry_point2[0], symmetry_point2[1])
    Dist_40 = DistancePointLine(points[40][0],points[40][1], symmetry_point1[0], symmetry_point1[1], symmetry_point2[0], symmetry_point2[1])
    difference_array.append((Dist_30-Dist_40)/Dist_30)
    Dist_58 = DistancePointLine(points[58][0],points[58][1], symmetry_point1[0], symmetry_point1[1], symmetry_point2[0], symmetry_point2[1])
    Dist_54 = DistancePointLine(points[54][0],points[54][1], symmetry_point1[0], symmetry_point1[1], symmetry_point2[0], symmetry_point2[1])
    difference_array.append((Dist_58-Dist_54)/Dist_58)
    Dist_59 = DistancePointLine(points[59][0],points[59][1], symmetry_point1[0], symmetry_point1[1], symmetry_point2[0], symmetry_point2[1])
    Dist_65 = DistancePointLine(points[65][0],points[65][1], symmetry_point1[0], symmetry_point1[1], symmetry_point2[0], symmetry_point2[1])
    difference_array.append((Dist_59-Dist_65)/Dist_59)

    # Difference in perpendicular projection on the symmetry line
    temp1 = []
    temp2 = []
    perpendicular_vectors = Perpendicular([vx_symmetry, vy_symmetry])
    temp1 = np.asanyarray([(points[18][0] - 200 * perpendicular_vectors[0]), (points[18][1] - 200 * perpendicular_vectors[1])])
    temp2 = np.asarray([(points[18][0] + 200 * perpendicular_vectors[0]), (points[18][1] + 200 * perpendicular_vectors[1])])
    point1 = LineSegmentIntersection(temp1, temp2, symmetry_point1, symmetry_point2)
    temp1 = np.asarray([(points[25][0] - 200 * perpendicular_vectors[0]), (points[25][1] - 200 * perpendicular_vectors[1])])
    temp2 = np.asarray([(points[25][0] + 200 * perpendicular_vectors[0]), (points[25][1] + 200 * perpendicular_vectors[1])])
    point2 = LineSegmentIntersection(temp1, temp2, symmetry_point1, symmetry_point2)
    difference_array2.append(lineMagnitude(point1[0], point1[1], point2[0], point2[1]))
    temp1 = np.asarray([(points[21][0] - 200 * perpendicular_vectors[0]), (points[21][1] - 200 * perpendicular_vectors[1])])
    temp2 = np.asarray([(points[21][0] + 200 * perpendicular_vectors[0]), (points[21][1] + 200 * perpendicular_vectors[1])])
    point1 = LineSegmentIntersection(temp1, temp2, symmetry_point1, symmetry_point2)
    temp1 = np.asarray([(points[22][0] - 200 * perpendicular_vectors[0]), (points[22][1] - 200 * perpendicular_vectors[1])])
    temp2 = np.asarray([(points[22][0] + 200 * perpendicular_vectors[0]), (points[22][1] + 200 * perpendicular_vectors[1])])
    point2 = LineSegmentIntersection(temp1, temp2, symmetry_point1, symmetry_point2)
    difference_array2.append(lineMagnitude(point1[0], point1[1], point2[0], point2[1]))
    temp1 = np.asarray([(points[30][0] - 200 * perpendicular_vectors[0]), (points[30][1] - 200 * perpendicular_vectors[1])])
    temp2 = np.asarray([(points[30][0] + 200 * perpendicular_vectors[0]), (points[30][1] + 200 * perpendicular_vectors[1])])
    point1 = LineSegmentIntersection(temp1, temp2, symmetry_point1, symmetry_point2)
    temp1 = np.asarray([(points[40][0] - 200 * perpendicular_vectors[0]), (points[40][1] - 200 * perpendicular_vectors[1])])
    temp2 = np.asarray([(points[40][0] + 200 * perpendicular_vectors[0]), (points[40][1] + 200 * perpendicular_vectors[1])])
    point2 = LineSegmentIntersection(temp1, temp2, symmetry_point1, symmetry_point2)
    difference_array2.append(lineMagnitude(point1[0], point1[1], point2[0], point2[1]))
    temp1 = np.asarray([(points[58][0] - 200 * perpendicular_vectors[0]), (points[58][1] - 200 * perpendicular_vectors[1])])
    temp2 = np.asarray([(points[58][0] + 200 * perpendicular_vectors[0]), (points[58][1] + 200 * perpendicular_vectors[1])])
    point1 = LineSegmentIntersection(temp1, temp2, symmetry_point1, symmetry_point2)
    temp1 = np.asarray([(points[54][0] - 200 * perpendicular_vectors[0]), (points[54][1] - 200 * perpendicular_vectors[1])])
    temp2 = np.asarray([(points[54][0] + 200 * perpendicular_vectors[0]), (points[54][1] + 200 * perpendicular_vectors[1])])
    point2 = LineSegmentIntersection(temp1, temp2, symmetry_point1, symmetry_point2)
    difference_array2.append(lineMagnitude(point1[0], point1[1], point2[0], point2[1]))
    temp1 = np.asarray([(points[59][0] - 200 * perpendicular_vectors[0]), (points[59][1] - 200 * perpendicular_vectors[1])])
    temp2 = np.asarray([(points[59][0] + 200 * perpendicular_vectors[0]), (points[59][1] + 200 * perpendicular_vectors[1])])
    point1 = LineSegmentIntersection(temp1, temp2, symmetry_point1, symmetry_point2)
    temp1 = np.asarray([(points[65][0] - 200 * perpendicular_vectors[0]), (points[65][1] - 200 * perpendicular_vectors[1])])
    temp2 = np.asarray([(points[65][0] + 200 * perpendicular_vectors[0]), (points[65][1] + 200 * perpendicular_vectors[1])])
    point2 = LineSegmentIntersection(temp1, temp2, symmetry_point1, symmetry_point2)
    difference_array2.append(lineMagnitude(point1[0], point1[1], point2[0], point2[1]))

    return difference_array, difference_array2
# 18(viewer left), 25(right) outer points of eyebrows
# 21(left), 22(right) inner points of eyebrows
# 30(left), 40(right) eye edge detection(towards nose)
# 58(left), 54(right) edge points of nose
# 59(left), 65(right) lip edge points

    # for i in range(0, (points_count / 2) - 1):
    #     # The above loop will run if eye_line_point1[1]< eye_line_point2[1]
    #     if (eye_line_point1[1] < eye_line_point2[1]) or (eye_line_point1[1] == eye_line_point2[1]):
    #         if face_boundary_points[1, i] < eye_line_point1[1]:
    #             print 'pass ', i
    #             pass
    #         elif (face_boundary_points[1, i] < symmetry_perpendicular_intersection[1]) and (
    #                     eye_line_point1[1] < face_boundary_points[1, i]):
    #             xf = find_x(face_boundary_points[1, i], eye_line_point1, eye_line_point2)
    #             sum_left = sum_left + np.sum(input_img[face_boundary_points[0, i]:xf, face_boundary_points[1, i]])
    #             print 'cond 1'
    #         elif ((symmetry_perpendicular_intersection[1] < face_boundary_points[1, i]) and (
    #                     face_boundary_points[1, i] < eye_line_point2[1])):
    #             xf = find_x(face_boundary_points[1, i], eye_line_point1, eye_line_point2)
    #             xg = find_x(face_boundary_points[1, i], symmetry_point1, symmetry_point2)
    #             sum_left = sum_left + np.sum(input_img[face_boundary_points[0, i]:xg, face_boundary_points[1, i]])
    #             sum_right = sum_right + np.sum(input_img[xg:xf, face_boundary_points[1, i]])
    #             print 'cond 2'
    #
    #         elif face_boundary_points[1, i] > eye_line_point2[1]:
    #             xg = find_x(face_boundary_points[1, i], symmetry_point1, symmetry_point2)
    #             sum_left = sum_left + np.sum(input_img[face_boundary_points[0, i]:xg, face_boundary_points[1, i]])
    #             sum_right = sum_right + np.sum(input_img[xg:right_boundary[0, i], face_boundary_points[1, i]])
    #             # sum_right = sum_right + np.sum(input_img[xg:face_boundary_points[1,i], face_boundary_points[1,i]])
    #             print 'cond 3'
    #             # total_sum = sum_left + sum_right
    #             #         # pdb.set_trace()set_trace
    #             # left_percentage = (sum_left/total_sum)*100
    #             # right_percentage = (sum_right/total_sum)*100
    #
    #
    #
    #     # For the case where eye_line_point1[1]>=eye_line_point2[1]:
    #     else:
    #         print 'Case 2 - left tilted face'
    #         if face_boundary_points[1, i] < eye_line_point2[1]:
    #             pass
    #         elif (face_boundary_points[1, i] < symmetry_perpendicular_intersection[1]) and (
    #                     eye_line_point2[1] < face_boundary_points[1, i]):
    #             xf = find_x(face_boundary_points[1, i], eye_line_point1, eye_line_point2)
    #             sum_right = sum_right + np.sum(input_img[xf:right_boundary[0, i], face_boundary_points[1, i]])
    #             #
    #             # xg = find_x(face_boundary_points[1], symmetry_point1, symmetry_point2)
    #             # sum_left = sum_left + np.sum(input_img[face_boundary_points[0,i]:xg, face_boundary_points[1,i]])
    #             # sum_right = sum_right + np.sum(input_img[xg:xf, face_boundary_points[1,i]])
    #         elif ((symmetry_perpendicular_intersection[1] < face_boundary_points[1, i]) and (
    #                     face_boundary_points[1, i] < eye_line_point1[1])):
    #             xf = find_x(face_boundary_points[1, i], eye_line_point1, eye_line_point2)
    #             xg = find_x(face_boundary_points[1, i], symmetry_point1, symmetry_point2)
    #             sum_left = sum_left + np.sum(input_img[xf:xg, face_boundary_points[1, i]])
    #             sum_right = sum_right + np.sum(input_img[xg:right_boundary[0, i], face_boundary_points[1, i]])
    #         elif face_boundary_points[1, i] > eye_line_point1[1]:
    #             xg = find_x(face_boundary_points[1, i], symmetry_point1, symmetry_point2)
    #             # pdb.set_trace()
    #             print "face_boundary_points[0,i] = ", face_boundary_points[0, i]
    #             print "xg = ", xg
    #             sum_left = sum_left + np.sum(input_img[face_boundary_points[0, i]:xg, face_boundary_points[1, i]])
    #             sum_right = sum_right + np.sum(input_img[xg:right_boundary[0, i], face_boundary_points[1, i]])

    # total_sum = sum_left + sum_right
    # # pdb.set_trace()
    # left_percentage = (sum_left / total_sum) * 100
    # right_percentage = (sum_right/total_sum) * 100

    # return left_percentage, right_percentage