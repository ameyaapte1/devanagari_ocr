import cv2
import numpy as np
def char_id_y(file_obj):
    pass
def get_flattened_image(file_obj, resolution):
    pass

svm_params = dict( kernel_type = cv2.SVM_RBF,
                            svm_type = cv2.SVM_C_SVC,
                                                C=2.67, gamma=5.383 )

filename='' #Set the filename
img = cv2.imread(filename,0)

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
