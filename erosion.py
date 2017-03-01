import cv2
import numpy as np
img = cv2.imread('binary.jpg',cv2.IMREAD_GRAYSCALE)
#img=255-img
kernel = np.ones((3,3),np.uint8)
#gradient = cv2.morphologyEx(img, cv2.MORPH_GRADIENT, kernel)
dilation = cv2.erode(img,kernel,iterations = 1)
cv2.imwrite("erosion.jpg",dilation)
