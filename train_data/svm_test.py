from svm_lib import *
import numpy as np
import sys

svm = cv2.ml.SVM_load('trained_data.dat')
test_X, test_Y = extract_from_file(sys.argv[1])
test_X = np.float32(test_X)
test_Y = np.float32(test_Y)
result = svm.predict(test_X)
print result
print test_Y
mask = result[1] == test_Y
correct = np.count_nonzero(mask)
for x in result[1]:
    print char_from_id(x) ,
print correct * 100.0 / 26
