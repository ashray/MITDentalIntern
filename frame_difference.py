import numpy as np
import pdb
import math
import matplotlib.pyplot as plt
import os.path
from faceMorpher import *
from sklearn import svm

videoLocation = '/Users/me/Desktop/MITDentalData/Videos/'
side = ['e','l','r']
for i in range (1,15):
    for j in range (0,3):
        videoName = videoLocation+str(i)+side[j]+".mov"
        if (os.path.isfile(videoName)==False):
            print 'Did not find ',videoName
            videoName = videoLocation+str(i)+side[j+1]+".mov"
        if (os.path.isfile(videoName)==False):
            print 'Did not find ',videoName
            videoName = videoLocation+str(i)+side[j+2]+".mov"
        if (os.path.isfile(videoName)==False):
            pass

        cam = cv2.VideoCapture(videoName)
        frame_limit = int(cam.get(cv2.cv.CV_CAP_PROP_FRAME_COUNT))
        k=0
        while k<frame_limit:
            ret, img = cam.read()
            cv2.imwrite('temporary_image.png',img)
            directoryLocation = os.path.dirname(os.path.abspath(__file__))
            imageLocation = directoryLocation + '/temporary_image.png'
            img,points = landmark_locator(imageLocation,  width=500, height=600, fps=10)
            np.save(videoLocation+str(i)+side[j]+".mov", points)
            print 'Generated .npy for ',videoLocation+str(i)+side[j]+".mov", 'frame',k
            k = k+1


        j = j+1
    i=i+1





#
# if __name__ == '__main__':
#     frame_limit = 75
#     feature_vector = np.zeros((4,2,frame_limit-1),dtype=np.int)
#     directoryPath = "/Users/me/Desktop/MITDentalData/Videos/"
#     array_name = [
#         "1l.mov",
#         "2l.mov",
#         "3l.mov",
#         "4l.mov",
#         "1r.mov",
#         "2r.mov",
#         "3r.mov",
#         "4r.mov"
#     ]
#
#     np.save(array_name[0], feature_vector)
#
#     # array_name = [
#     #     "L1.mov",
#     #     "L2.mov",
#     #     "L3.mov",
#     #     "L4.mov",
#     #     "L8.mov",
#     #     "L9.mov",
#     #     "L10.mov",
#     #     "R1.mov",
#     #     "R2.mov",
#     #     "R3.mov",
#     #     "R4.mov",
#     #     "R5.mov",
#     #     "R6.mov",
#     #     "R7.mov"
#     # ]
#     # array_name_diff = [
#     #     "L1_diff.mov",
#     #     "L2_diff.mov",
#     #     "L3_diff.mov",
#     #     "L4_diff.mov",
#     #     "L8_diff.mov",
#     #     "L9_diff.mov",
#     #     "L10_diff.mov",
#     #     "R1_diff.mov",
#     #     "R2_diff.mov",
#     #     "R3_diff.mov",
#     #     "R4_diff.mov",
#     #     "R5_diff.mov",
#     #     "R6_diff.mov",
#     #     "R7_diff.mov"
#     #
#     # ]
#     output = [0,0,0,0,1,1,1,1]
#     svm_input = []
#     poi = [5, 7, 59, 65]
#
#     # store_x_values = []
#     # store_y_values = []
#     for loop_iterator in range(8):
#         print loop_iterator
#         read_npy = np.load(directoryPath+array_name[loop_iterator] + '.npy')
#         print loop_iterator, array_name[loop_iterator]
#         j = 0
#
#         while j < frame_limit-1:
#             for i in range (0,4):
#                 # if j==179:
#                 #     feature_vector[i][0][j] = read_npy[i,0,j] #- read_npy[i,0,j+1]
#                 #     feature_vector[i][1][j] = read_npy[i,1,j] #- read_npy[i,1,j+1]
#                 # else:
#                 # store_x_values = read_npy
#                 feature_vector[i][0][j] = read_npy[poi[i],0,j]
#                 feature_vector[i][1][j] = read_npy[poi[i],1,j]
#             # np.save(array_name_diff[loop_iterator], feature_vector)
#             # print j
#             j = j+1
#
#         # fourier_values = np.zeros((77,2,frame_limit-1))
#         # var = np.zeros(77)
#         # mean = np.zeros(77)
#         # third_moment = np.zeros(77)
#         # svm_input_temp = np.zeros(4*2)
#         # for i in range(0,4):
#         #     # var[i] = np.var(feature_vector[i,0,:])
#         #     # mean[i] = np.mean(feature_vector[i,0,:])
#         #     # third_moment[i] = sum((feature_vector[i,0,:] - np.mean(feature_vector[i,0,:]))**3)/len(feature_vector[i,0,:])
#         #     # svm_input_temp[2*i] = var[i]
#         #     # svm_input_temp[2*i + 1] = third_moment[i]
#         #     svm_input_temp[i,0] = feature_vector[i][0]
#         #     svm_input_temp[i,1] = feature_vector[i][1]
#         # svm_input.append(svm_input_temp.tolist())
#         svm_input.append(feature_vector.tolist())
#             # fourier_values[i,0,:] = np.fft.fft(np.sqrt(feature_vector[i,0,:]**2 + feature_vector[i,1,:]**2))
#             # fourier_values[i,1,:] = np.fft.fft(np.arctan2(feature_vector[i,1,:], feature_vector[i,0,:]))
#     # svm_input = zip(var, third_moment)
#     # pdb.set_trace()
#     # X = [[0, 0], [1, 1]]
#     # Y = [0, 1]
#     clf = svm.SVC()
#     clf.fit(svm_input[5:11], output[5:11])
#     clf.predict(svm_input[:])
#
#     pdb.set_trace()