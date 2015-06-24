import numpy as np
import pdb

if __name__ == '__main__':
    frame_limit = 180
    feature_vector = np.zeros((77,2,frame_limit),dtype=np.int)
    array_name = [
        "L8.mov",
        "L9.mov",
        "L10.mov",
        "R1.mov",
    ]
    array_name_diff = [
        "L8_diff.mov",
        "L9_diff.mov",
        "L10_diff.mov",
        "R1_diff.mov",
    ]
    for loop_iterator in range(4):
        read_npy = np.load(array_name[loop_iterator] + '.npy')
        j = 0

        while j < frame_limit:
            for i in range (0,77):
                if j==179:
                    feature_vector[i][0][j] = read_npy[i,0,j] #- read_npy[i,0,j+1]
                    feature_vector[i][1][j] = read_npy[i,1,j] #- read_npy[i,1,j+1]
                else:
                    feature_vector[i][0][j] = read_npy[i,0,j] - read_npy[i,0,j+1]
                    feature_vector[i][1][j] = read_npy[i,1,j] - read_npy[i,1,j+1]

            np.save(array_name_diff[loop_iterator], feature_vector)
            print j
            j = j+1

    pdb.set_trace()
