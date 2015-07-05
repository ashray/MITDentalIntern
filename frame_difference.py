import numpy as np
import pdb
import math
import matplotlib.pyplot as plt
from sklearn import svm

if __name__ == '__main__':
    frame_limit = 180
    feature_vector = np.zeros((77,2,frame_limit-1),dtype=np.int)
    array_name = [
        "L1.mov",
        "L2.mov",
        "L3.mov",
        "L4.mov",
        "L8.mov",
        "L9.mov",
        "L10.mov",
        "R1.mov",
        "R2.mov",
        "R3.mov",
        "R4.mov",
        "R5.mov",
        "R6.mov",
        "R7.mov"
    ]
    array_name_diff = [
        "L1_diff.mov",
        "L2_diff.mov",
        "L3_diff.mov",
        "L4_diff.mov",
        "L8_diff.mov",
        "L9_diff.mov",
        "L10_diff.mov",
        "R1_diff.mov",
        "R2_diff.mov",
        "R3_diff.mov",
        "R4_diff.mov",
        "R5_diff.mov",
        "R6_diff.mov",
        "R7_diff.mov"

    ]
    output = [0,0,0,0,0,0,0,1,1,1,1,1,1,1]
    svm_input = []

    store_x_values = np.zeros[]
    store_y_values = []
    for loop_iterator in range(14):
        # print loop_iterator
        read_npy = np.load(array_name[loop_iterator] + '.npy')
        print loop_iterator, array_name[loop_iterator]
        j = 0

        while j < frame_limit-1:
            for i in range (0,77):
                # if j==179:
                #     feature_vector[i][0][j] = read_npy[i,0,j] #- read_npy[i,0,j+1]
                #     feature_vector[i][1][j] = read_npy[i,1,j] #- read_npy[i,1,j+1]
                # else:
                store_x_values = read_npy
                feature_vector[i][0][j] = read_npy[i,0,j+1] - read_npy[i,0,j]
                feature_vector[i][1][j] = read_npy[i,1,j+1] - read_npy[i,1,j]
            np.save(array_name_diff[loop_iterator], feature_vector)
            # print j
            j = j+1

        # fourier_values = np.zeros((77,2,frame_limit-1))
        var = np.zeros(77)
        mean = np.zeros(77)
        third_moment = np.zeros(77)
        svm_input_temp = np.zeros(77*2)
        for i in range(0,77):
            var[i] = np.var(feature_vector[i,0,:])
            mean[i] = np.mean(feature_vector[i,0,:])
            third_moment[i] = sum((feature_vector[i,0,:] - np.mean(feature_vector[i,0,:]))**3)/len(feature_vector[i,0,:])
            svm_input_temp[2*i] = var[i]
            svm_input_temp[2*i + 1] = third_moment[i]
        svm_input.append(svm_input_temp.tolist())
            # fourier_values[i,0,:] = np.fft.fft(np.sqrt(feature_vector[i,0,:]**2 + feature_vector[i,1,:]**2))
            # fourier_values[i,1,:] = np.fft.fft(np.arctan2(feature_vector[i,1,:], feature_vector[i,0,:]))
    # svm_input = zip(var, third_moment)
    # pdb.set_trace()
    # X = [[0, 0], [1, 1]]
    # Y = [0, 1]
    clf = svm.SVC()
    clf.fit(svm_input[5:11], output[5:11])
    clf.predict(svm_input[:])

    pdb.set_trace()