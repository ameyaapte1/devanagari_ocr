import sys, os
import cv2
import numpy as np
def char_id_y(file_obj):
    line = file_obj.readline()
    try:
        unicode_ = int(line[-5:])
        return unicode_%2325
    except:
        return -1
def char_from_id(char_id):
    unicode_ = char_id + 2325
    return unichr(unicode_)

def get_y(file_obj):
    line  = file_obj.readline()
    try:
        code = int(line[-4:])
        return code
    except:
        return -1
def get_flattened_image(file_obj, resolution):
    x,y = resolution
    feature_vector_list = []
    for i in xrange(y):
        line = file_obj.readline()
        line = line.replace("0","1")
        line = line.replace(" ", "0")
        li = map(int, list(line[:-1]))
        feature_vector_list.append(li)
    #print feature_vector_list
    return np.float32(np.array(feature_vector_list, dtype='float32').flatten())\
            .reshape(-1,18*14)

def oneshot_vector(activate):
    vector = np.zeros(37,dtype='float32')
    vector[activate] = 1.000
    return vector.reshape(-1,37)

def extract_from_file(filename):
    feat_file = open(filename, "r")
    if feat_file == None:
        print "File nahi milali"
        sys.exit(1)
    X = np.array([],dtype = "float32").reshape(-1,18*14)
    Y = np.array([],dtype = "float32").reshape(-1,1)
    while True:
        code = get_y(feat_file)
        if code == -1:
            break
        Y = np.vstack((Y, code))
        a = get_flattened_image(feat_file,(18,14))
        X = np.vstack((X,a))
    return X,Y


