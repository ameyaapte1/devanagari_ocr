import cv2
import numpy as np
import sys

img = cv2.imread(sys.argv[1],0)
bin_img = img.copy()

prev_i = 0
prev_j = 0
step=int(sys.argv[2])

i=0
j=0
print img.shape
while i < img.shape[0]:
    
    i = i + img.shape[0]/step
    if(i > img.shape[0]):
        i = img.shape[0]

    while j < img.shape[1]:
        j = j + img.shape[1]/step
        if(j > img.shape[1]):
            j = img.shape[1]
        if(np.amin(img[prev_i:i,prev_j:j]) > 96):
            bin_img[prev_i:i,prev_j:j] = 255
        else:
            blur = cv2.GaussianBlur(img[prev_i:i,prev_j:j],(5,5),0)
            ret3,th = cv2.threshold(blur,254,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
            bin_img[prev_i:i,prev_j:j] = th
            
        prev_j=j
        
    prev_i=i
    prev_j=j=0
# Otsu's thresholding after Gaussian filtering
bin_img[bin_img>=128]=255
bin_img[bin_img<128]=0
cv2.imwrite(sys.argv[3],bin_img)
