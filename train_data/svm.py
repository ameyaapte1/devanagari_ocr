from svm_lib import *
import os
import numpy as np
import cv2
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
train_X = np.float32(train_X)
train_Y = np.int32(train_Y)
print train_Y, train_Y.dtype
svm = cv2.ml.SVM_create()
svm.setKernel(cv2.ml.SVM_RBF)
svm.setType(cv2.ml.SVM_C_SVC)
svm.setC(2.67)
svm.setGamma(5.383)
svm.train(train_X, cv2.ml.ROW_SAMPLE,train_Y)
svm.save('trained_data.dat')
print "Training...Done"
