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
        code = char_id_y(feat_file)
        if code == -1:
            break
        Y = np.vstack((Y, code))
        a = get_flattened_image(feat_file,(18,14))
        X = np.vstack((X,a))
    return X,Y
folders = ["images/", "rotated_img/", "degraded_img/"]
train_X = np.array([],dtype = "float32").reshape(-1,18*14)
train_Y = np.array([],dtype = "float32").reshape(-1,1)
for folder in folders:
    files = os.listdir(folder)
    for fil in files:
        if ".dat" not in fil:
            continue
        print folder+fil
        X,Y = extract_from_file(folder+fil)
        train_X = np.vstack((train_X, X))
        train_Y = np.vstack((train_Y, Y))
print train_X
print train_Y, train_Y.dtype
svm = cv2.ml.SVM_create()
svm.setKernel(cv2.ml.SVM_LINEAR)
svm.setType(cv2.ml.SVM_C_SVC)
svm.setC(2.67)
svm.setGamma(5.383)
svm.train(train_X, cv2.ml.ROW_SAMPLE,train_Y)
svm.save('trained_data.dat')
"""
######     Now training      ########################

trainData = np.float32() # n X 1 Feature Vector
responses = np.float32() # m X 1 Output vector(Actual Output)

svm = cv2.SVM()
svm.train(trainData,responses, params=svm_params)
svm.save('svm_data.dat')

######     Now testing      ########################

testData = np.float32() # Feature vector extracted from test images
result = svm.predict_all(testData)

#######   Check Accuracy   ########################
mask = result == responses
correct = np.count_nonzero(mask)
print correct * 100.0 / result.size
"""
