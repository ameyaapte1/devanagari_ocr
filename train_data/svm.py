import sys
import cv2
import numpy as np
def char_id_y(file_obj):
    line = file_obj.readline()
    unicode_ = int(line[-4:])
    return unicode_%2325
def get_flattened_image(file_obj, resolution):
    x,y = resolution
    feature_vector_list = []
    for i in xrange(y):
        line = file_obj.readline()
        line = line.replace("0","1")
        line = line.replace(" ", "0")
        li = map(int, list(line[:-1]))
        feature_vector_list.append(li)
    print feature_vector_list
    return np.array(feature_vector_list, dtype=float).flatten()
#svm_params = dict( kernel_type = cv2.SVM_LINEAR,
#                            svm_type = cv2.SVM_C_SVC,
#                                                C=2.67, gamma=5.383 )

feat_file = open(sys.argv[1], "r")
if feat_file == None:
    print "File nahi milali"
    sys.exit(1)
code = char_id_y(feat_file)
a = get_flattened_image(feat_file,(18,14))
print np.float32(a).reshape(-1,18*14)
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
