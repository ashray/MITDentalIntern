import cv2
import numpy as np
import pdb
from image_downscale import image_downscale
from violaJones import *
from canny import *
from symmetryMidpoints import symmetryMidpoints
from symmetryCalculation import symmetryCalculationIntensity, symmetryCalculationBoundaryDifference, symmetryCalculationLandmarkPoints
from faceMorpher import landmark_locator
import os
from scipy.interpolate import interp1d
from helperFunctions import *
from faceMorpher import *
from scipy.ndimage.interpolation import rotate
# from scipy.stsci.image import translate
from scipy.ndimage.interpolation import shift
import matplotlib.pyplot as plt
import Tkinter
import time
import tkMessageBox
import aligner
from aligner import *

def faceSymmetry():
    # Accept original image as input
    # img = cv2.imread('./photo/sampleFaceImage0.jpg')
    # img = cv2.imread('temporary_image.png')
    # imageLocation = '/Users/me/Desktop/MITREDX/MITDentalIntern/photo/sampleFaceImage.png'
    # img = cv2.imread('sampleFaceImage3.JPG')

    cam = cv2.VideoCapture(0)



    if cam.isOpened():
        ret, img_o = cam.read()
    else:
        ret = False

    while ret:
        height, width, depth = img_o.shape
        img_o = img_o[:, width/8:7*width/8]
        cv2.imshow('Preview',img_o)
        ret, img_o = cam.read()

        key = cv2.waitKey(20)
        if key == 27:
            break


    time.sleep(2)
    cam.release()
    cv2.imwrite('temporary_image.png',img_o)
    directoryLocation = os.path.dirname(os.path.abspath(__file__))
    imageLocation = directoryLocation + '/temporary_image.png'
    point = locator.face_points(imageLocation)

    cv2.imshow('Original Image',img_o)
    cv2.moveWindow('Original Image', 0,0)
    # img = image_downscale(img, 400)
    img,points = landmark_locator(imageLocation,  width=500, height=600, fps=10)
    # size = [500, 600]
    # aligner.resize_align(img_o, point, size)
    # img_copy = img.copy()
    # img_copy_2 = img.copy()
    gray = colGray(img)



    # print "Line 25"
    # midPoint = []

    # Standard face detector; returns
    # x,y,w and h = face bounding rectangle , midPoint = an array of old symmetry line points, intersection_x, intersection_y = points for normal to the symmetry line

    midPoint, x, y, w, h, intersection_x, intersection_y = faceFeatureDetector(img)

    # x1,y1,x2,y2,img = faceDetectionVideo()
    # img = image_downscale(img, 400)
    # a = FindEdgeImage(img[max((y1-y2),0):y1 + 2*y2, max((x1-x2),0):x1+2*x2])
    # Finds the edge boundary points and returns the outer boundary as a series of points in variable 'a'

    # $$$
    # a = FindEdgeImage(img_copy[max((y-h),0):y + 2*h, max((x-w),0):x+2*w])
    # aBound = np.arange(2 * len(a)).reshape((2, len(a)))
    #
    # for i in range (0, len(a)/2):
    #     if(a[1,i]<y+h):
    #         aBound[1,i]=a[1,i]
    #     else:
    #         break



    # a = FindEdgeImage(img_copy[max((y-h),0):y + 2*h, max((x-w),0):x+2*w], y, h)

    # Calculates distance between corresponding points on face curve to get an array of points for drawing central face symmetry line
    # midpoints = symmetryMidpoints(a,img,0,0)

    # $$$
    # midpoints = symmetryMidpoints(a,img,x,y)

    # midpoints = symmetryMidpoints(aBound,img,x,y)
    # margin_size = 8
    # Draws central symmetry line using new symmetryMidpoints
    # xbf_temp, ybf_temp, vx_temp, vy_temp = draw_line(img,midpoints)
    # midpoints_features = assignWeights(midpoints,midPoint)
    # height, width, depth  = img.shape
    # img_cropped = img[margin_size:(height-margin_size), margin_size:(width-margin_size)]
    # xbf_temp, ybf_temp, vx_temp, vy_temp = draw_line(img_cropped,midpoints)

    # print "Line 67"
    # cv2.imshow('new midpoints fitline',img_cropped)

    # pdb.set_trace()

    # height, width, depth  = img_copy.shape
    # img_copy_cropped = img_copy[margin_size:(height-margin_size), margin_size:(width-margin_size)]

    # img = PlotPoints(midpoints,img_copy_cropped, 0, 0)
    # ------------------------------------------------------------
    # Use fitline to fit these points on a straight line. Verify using photoshop if that is the actual centre. Also check if there is a shift of 8 points or not.
    #
    # Then get the slope and input it to the perpendicular function used below
    # ------------------------------------------------------------

    # cv2.imshow('new midpoints',img)
    # cv2.waitKey(0)
    # print a.shape
    # print "Line 85"
    # pdb.set_trace()

    # Plot the face curve
    # img_new = PlotPoints(a,img, x, y)

    # height, width, depth  = img_copy.shape
    # img_copy_cropped = img_copy[margin_size:(height-margin_size), margin_size:(width-margin_size)]


    # img_new = PlotPoints(a,img_copy_cropped, 0, 0)

    # cv2.imshow('img', img_new)
    # cv2.waitKey(0)
    # len_list = len(midPoint) / 2
    #
    # midPointNP = np.asarray(midPoint)
    #
    # SymmetryLinePoints = midPointNP.reshape(len_list, 2)
    #
    # xbf, ybf, vx, vy= draw_line(img_new, SymmetryLinePoints)
    # # sum_image1, sum_image2, img_new = skin_detector(img_new, x, y, w, h, xbf, ybf, intersection_x, intersection_y, vx, vy)
    #
    # [vx_perpen,vy_perpen] = Perpendicular([vx,vy])

    # The upper bounding line is defined by points intersection_x, intersection_y and direction vectors vx_perpen, vy_perpen

    # cv2.line(img_copy,(intersection_x,intersection_y),(intersection_x+(100*vx_perpen),intersection_y+100*vy_perpen), (255, 224, 0), 6)
    # cv2.line(img_copy,(intersection_x,intersection_y),(intersection_x-(100*vx_perpen),intersection_y-100*vy_perpen), (255, 224, 0), 6)
    # sub_image1 = img_copy[y:y + h, x:xbf]
    # sub_image2 = img_copy[y:y + h, xbf:x + w]
    # sub_image1 is the left side of the image
    # sub_image2 is the right side of the image
    # a has the boundary points of the face
    # print 'Working till before perpen calc'
    # [vx_perpen,vy_perpen] = Perpendicular([vx_temp,vy_temp])
    # cv2.line(img_copy,(intersection_x,intersection_y),(intersection_x+(100*vx_perpen),intersection_y+100*vy_perpen), (255, 224, 0), 6)
    # cv2.line(img_copy,(intersection_x,intersection_y),(intersection_x-(100*vx_perpen),intersection_y-100*vy_perpen), (255, 224, 0), 6)

    # intersection_x , intersection_y is the mid point of the two eyes. Perpendicular line passes through these points
    # linePoint1 = [(intersection_x+(100*vx_perpen)),(intersection_y+100*vy_perpen)]
    # linePoint2 = [intersection_x-(100*vx_perpen),(intersection_y-100*vy_perpen)]

    # [leftIntersectionPoint, rightIntersectionPoint] = FaceSymmetryLineIntersection(a, linePoint1, linePoint2)
    # leftIntersectionPoint = np.array(leftIntersectionPoint)
    # rightIntersectionPoint = np.array(rightIntersectionPoint)
    # cv2.line(img_copy_cropped, (leftIntersectionPoint[0],leftIntersectionPoint[1]), (leftIntersectionPoint[0],leftIntersectionPoint[1]), (0,255,0),10)
    # cv2.line(img_copy_cropped, (rightIntersectionPoint[0],rightIntersectionPoint[1]), (rightIntersectionPoint[0],rightIntersectionPoint[1]), (0,255,0),10)
    # cv2.imshow('img_new_cropped', img_copy_cropped)

    # print 'Line 135'
    # cv2.waitKey(0)
    weightageToDistance=0.8

    # pdb.set_trace()

    # difference1, difference2 = symmetryCalculationLandmarkPoints(points, int(xbf_temp),int(ybf_temp),int(vx_temp),int(vy_temp))
    # xbf_temp = points[56,0]
    # ybf_temp = points[56,1]

    feature_landmarks_central = 56
    feature_landmarks_one = [18, 21, 30, 58, 59]
    feature_landmarks_two = [25, 22, 40, 54, 65]

    midpoints = np.zeros((6,2),dtype=np.int)
    midpoints[0][0] = points[feature_landmarks_central][0]
    midpoints[0][1] = points[feature_landmarks_central][1]

    for i in range(1,len(feature_landmarks_one)+1):
        midpoints[i][0] = points[feature_landmarks_one[i-1]][0] + (points[feature_landmarks_two[i-1]][0] - points[feature_landmarks_one[i-1]][0])/2
        midpoints[i][1] = points[feature_landmarks_one[i-1]][1] + (points[feature_landmarks_two[i-1]][1] - points[feature_landmarks_one[i-1]][1])/2

    # pdb.set_trace()
    [vx_temp, vy_temp, xbf_temp, ybf_temp] = cv2.fitLine(midpoints, cv2.cv.CV_DIST_L1, 0, 0.01, 0.01)


    xbf = np.around(xbf_temp)
    ybf = np.around(ybf_temp)
    height, width, depth  = img.shape

    x_center = width/2
    y_center= height/2









    # xbf = xbf + image_shift_x
    # ybf = ybf + image_shift_y

    # img.shape = img.width + image_shift_x
    # img.height = img.height + image_shift_y

    # translated_image = shift(img, image_shift_x)

    distance = 400
    x2 = xbf + distance * vx_temp
    y2 = ybf + distance * vy_temp

    x3 = xbf + (-1) * distance * vx_temp
    y3 = ybf + (-1) * distance * vy_temp

    slopeN = y3 - y2
    slopeD = x3 - x2

    x2 = np.around(x2)
    y2 = np.around(y2)

    x3 = np.around(x3)
    y3 = np.around(y3)
    cv2.line(img, (x3, y3), (x2, y2), (255, 0, 0), 1)

    x_line = find_x(y_center, (x2,y2),(x3,y3))

    image_shift_x = width/2-x_line
    image_shift_y = abs(height/2-ybf)

    M = np.float32([[1,0,image_shift_x],[0,1,0]])
    translated_image = cv2.warpAffine(img, M, (width, height))
    cv2.line(translated_image,(xbf,ybf),(xbf,ybf),(0,0,255),1)

    height, width, depth  = translated_image.shape

    x_center = width/2
    y_center= height/2

    cv2.line(translated_image,(x_center,y_center),(x_center,y_center),(0,0,255),3)

    # -------------------------------------------------------
    # Hack just to get 2 points from the direction vectors so that we can plot the line

    # cv2.imshow("translated",translated_image)
    # cv2.moveWindow('translated', width,0)
    # cv2.waitKey(0)

    # Pushing the line to be in the first or the second quadrant
    # if vy_temp < 0 and vx_temp < 0:
    #     vy_temp = abs(vy_temp)
    #     vx_temp = abs(vx_temp)
    #
    # elif vx_temp > 0 and vy_temp < 0:
    #     vy_temp = (-1)*vy_temp
    #     # vx_temp = (-1)*vx_temp

    if vx_temp == 0:
        angle = 0
    else:
        # slope = slopeN/slopeD
        angle = math.atan2(vy_temp,vx_temp)
        angle_degree = math.degrees(angle)
        print 'vx_temp',vx_temp
        print 'vy_temp',vy_temp
        print 'angle',angle_degree
        # if angle_degree<0:

        if angle_degree <0 :

            angle_degree = 90 + angle_degree
            rot_mat = cv2.getRotationMatrix2D((width/2,height/2),angle_degree,1.0)
        else:
            angle_degree = 90 - angle_degree
            rot_mat = cv2.getRotationMatrix2D((width/2,height/2),-1*angle_degree,1.0)
    # cv2.line(img, (x3, y3), (x2, y2), (255, 0, 0), 1)
    #
    # def rotateImage(image, angle):
    # image_center = tuple(np.array(translated_image.shape)/2)

    img_rotated = cv2.warpAffine(translated_image, rot_mat, (width, height))
  # return result



    # img_rotated = rotate(translated_image, angle_degree, axes=(1,0))

    height, width, depth  = img_rotated.shape
    img_left = img_rotated[0:height, 0:width/2]
    flipped_left = cv2.flip(img_left,1)
    symmetry_left = np.concatenate((img_left, flipped_left), axis=1)
    cv2.imshow('flipped left',symmetry_left)
    cv2.moveWindow('flipped left', width, 0)

    img_right = img_rotated[0:height, width/2:width]
    flipped_right = cv2.flip(img_right,1)
    symmetry_right = np.concatenate((flipped_right, img_right), axis=1)
    cv2.imshow('flipped right',symmetry_right)
    cv2.moveWindow('flipped right', 2*width, 0)

    # cv2.imshow('rotated image',img_rotated)
    # cv2.moveWindow('rotated image' , 200,200)
    time.sleep(3)



    # cubic_x = [0,0,0,0,0,0,0,0,0,0,0,0,0,0]
    # cubic_y = [0,0,0,0,0,0,0,0,0,0,0,0,0,0]

    cubic_x = np.zeros(14, dtype=int)
    cubic_y = np.zeros(14, dtype=int)

    l=0
    while l<13:
        cv2.line(img, (points[l][0],points[l][1]), (points[l][0],points[l][1]), (255,224,0),5)
        cubic_x[l] = points[l][0]
        cubic_y[l] = points[l][1]
        l=l+1
    f = interp1d(cubic_x, cubic_y, kind='cubic')
    # plt.plot(cubic_x, f(cubic_y),'-')
    # plt.show()

    cv2.imshow("Symmetry Line",img)


    difference1, difference2 = symmetryCalculationLandmarkPoints(points, xbf_temp,ybf_temp,vx_temp,vy_temp)
    if np.mean(difference1)>0:
        dominant = "left"
    #     Basically the distance of left is more than right, hence left is more dominant/larger
    else:
        dominant = "right"
    print "Asymmetry because of difference in distances" + str(difference1)
    print "Assymetry because of vertical missalignment " + str(difference2)
    print "Percentage asymmetry " + str((weightageToDistance*np.absolute(np.mean(difference1))+(1-weightageToDistance)*np.mean(difference2)))
    print "Dominant side is " + dominant + "(Assuming strictly frontal input image)"
    tkMessageBox.showinfo( "Hello !", "You are "+ str((weightageToDistance*np.absolute(np.mean(difference1))+(1-weightageToDistance)*np.mean(difference2)))+ " % asymmetric. Your dominant side is " +dominant+".")

    # x1 = points[56,0] + (1) * 400 * vx_temp
    # y1 = points[56,1] + (1) * 400 * vy_temp
    # x2 = points[56,0] + (-1) * 400 * vx_temp
    # y2 = points[56,1] + (-1) * 400 * vy_temp
    # cv2.line(img, (x1,y1), (x2,y2), (0,0,255),1)

    # symmetry_point1 = np.asarray([(xbf_temp - 50 * vx_temp), (ybf_temp - 50 * vy_temp)])
    # symmetry_point2 = np.asarray([(xbf_temp + 50 * vx_temp), (ybf_temp + 50 * vy_temp)])
    # perpendicular_vectors = Perpendicular([vx_temp, vy_temp])
    # temp1 = np.asarray([(points[59][0] - 200 * perpendicular_vectors[0]), (points[59][1] - 200 * perpendicular_vectors[1])])
    # temp2 = np.asarray([(points[59][0] + 200 * perpendicular_vectors[0]), (points[59][1] + 200 * perpendicular_vectors[1])])
    # point1 = LineSegmentIntersection(temp1, temp2, symmetry_point1, symmetry_point2)
    # temp1 = np.asarray([(points[65][0] - 200 * perpendicular_vectors[0]), (points[65][1] - 200 * perpendicular_vectors[1])])
    # temp2 = np.asarray([(points[65][0] + 200 * perpendicular_vectors[0]), (points[65][1] + 200 * perpendicular_vectors[1])])
    # point2 = LineSegmentIntersection(temp1, temp2, symmetry_point1, symmetry_point2)
    # cv2.line(img, (point1[0],point1[1]), (point1[0],point1[1]), (0,255,255),1)
    # cv2.line(img, (point2[0],point2[1]), (point2[0],point2[1]), (0,255,255),1)
    # # cv2.line(img, (symmetry_point1[0],symmetry_point1[1]), (symmetry_point1[0],symmetry_point1[1]), (0,255,255),10)
    # # cv2.line(img, (symmetry_point2[0],symmetry_point2[1]), (symmetry_point2[0],symmetry_point2[1]), (0,255,255),10)
    # # cv2.line(img, (point2[0],point2[1]), (point2[0],point2[1]), (0,255,255),10)
    #
    # cv2.imshow("Image with landmark points", img)
    # pdb.set_trace()
    # left_percentage, right_percentage = symmetryCalculationIntensity(a,img_copy_2,leftIntersectionPoint,rightIntersectionPoint,xbf_temp,ybf_temp,vx_temp,vy_temp)
    # left_percentage, right_percentage = symmetryCalculationBoundaryDifference(a,img_copy_2,leftIntersectionPoint,rightIntersectionPoint,xbf_temp,ybf_temp,vx_temp,vy_temp)
    # print 'Left % = ',left_percentage,' Right % = ',right_percentage
    #percentageDifference = math.fabs(sum_image1[0] - sum_image2[0]) / max(sum_image1[0], sum_image2[0])
    #print "Percentage asymmetry ", percentageDifference * 100

    # cv2.imshow('img', img_new)
    time.sleep(3)
    # cv2.destroyAllWindows()


top = Tkinter.Tk()
top.attributes('-fullscreen', True)
# def helloCallBack():
#    tkMessageBox.showinfo( "Hello Python", "Hello World")

B = Tkinter.Button(top, text ="Take a Picture !", command = faceSymmetry)

B.pack()
top.mainloop()