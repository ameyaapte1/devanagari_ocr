import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread('./orig_image.jpg',0)

# Otsu's thresholding after Gaussian filtering
blur = cv2.GaussianBlur(img,(5,5),0)
ret3,bin_img = cv2.threshold(blur,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)

cv2.imwrite('bin_img.jpg',bin_img)
lines=list()
words=list()

start = 0
end = 0
flag = False
k = 0

horizontal_sum = bin_img.sum(axis = 1)

for index, item in enumerate(horizontal_sum):
    
    if item > 0 and flag == False:
        start = index;
        flag = True
        
    elif item == 0 and flag == True:
        end = index;
        lines.append([start,end])
        flag = False
        
for i in lines:
    vertical_sum = bin_img[i[0]:i[1]].sum(axis = 0)
    
    for index, item in enumerate(vertical_sum):
        if item > 0 and flag == False:
            start = index;
            flag = True

        elif item == 0 and flag == True:
            end = index;
            words.append([i[0],i[1],start,end])
            flag = False
            
for index, item in enumerate(lines):
    cv2.imwrite('line_' + str(index).zfill(3) + '.jpg',bin_img[item[0]:item[1]])
    
for index, item in enumerate(words):
    cv2.imwrite('word_' + str(index).zfill(3) + '.jpg',bin_img[item[0]:item[1],item[2]:item[3]])

    #Use the following function to get horizontal/vertical histogram of any image
def word_plot(orientation , word_list):
    dist = []
    for word in word_list:
        if orientation == 0:
            vertical_sum = bin_img[word[0]:word[1],word[2]:word[3]].sum(axis=0)
            plt.plot(vertical_sum)
            mean = np.zeros(word[3]-word[2])
            median = np.zeros(word[3]-word[2])
            mean.fill(np.mean(vertical_sum))
            median.fill(np.median(vertical_sum))
            plt.plot(mean)
            plt.plot(median)

        else:
            horizontal_sum = bin_img[word[0]:word[1],word[2]:word[3]].sum(axis=1)
            plt.plot(horizontal_sum)
            mean = np.zeros(word[1]-word[0])
            median = np.zeros(word[1]-word[0])
            variance = np.zeros(word[1]-word[0])
            mean.fill(np.mean(horizontal_sum))
            median.fill(np.median(horizontal_sum))
            plt.plot(mean)
            plt.plot(median)
    plt.show()
        
