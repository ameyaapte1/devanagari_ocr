# vim: tabstop=8 expandtab shiftwidth=4 softtabstop=4

import cv2
import numpy as np
import sys
from matplotlib import pyplot as plt

img = cv2.imread(sys.argv[1],0)

horizontal_histogram = (255*img.shape[1])-img.sum(axis=1)

plt.plot(horizontal_histogram,'ro')
plt.show()



upper_start=-1
lower_start=-1
find_upper=True

for loc,i in enumerate(horizontal_histogram):
    if( i/255 > img.shape[1]/2 and upper_start==-1 and find_upper and horizontal_histogram[loc]<=horizontal_histogram[loc+1] and horizontal_histogram[loc+1] >= horizontal_histogram[loc+2]):
        upper_start=loc+1
        print upper_start
    if( upper_start!=-1 and find_upper and horizontal_histogram[loc]>=horizontal_histogram[loc+1] and horizontal_histogram[loc+1] <= horizontal_histogram[loc+2]):
        upper_end = loc+1
        print upper_end
        find_upper = False
    if(i/255 < 10 and lower_start==-1 and not find_upper):
        lower_start=loc
        break

stroke_width = upper_end-upper_start

cv2.imwrite(sys.argv[1].replace(".jpg","")+"_upper.jpg",img[:upper_start])
cv2.imwrite(sys.argv[1].replace(".jpg","")+"_lower.jpg",img[lower_start:])
cv2.imwrite(sys.argv[1].replace(".jpg","")+"_middle.jpg",img[upper_end:lower_start])

img = img[upper_end:lower_start]
vertical_histogram = (255*img.shape[0])-img.sum(axis=0)

#vertical_seg=[]
vertical_break=[]
#vertical_break.append(0)

percentile=[]
for i in range(0,50,2):
	percentile.append(np.percentile(vertical_histogram,i))
flag=False
print np.percentile(vertical_histogram,25)
for loc,data in enumerate(vertical_histogram < np.percentile(vertical_histogram,np.argmax(np.gradient(percentile))*2)):
	if(data):
		if(not flag):
			vertical_break.append(loc)
			flag=data
	else:
		if(flag):
			vertical_break.append(loc)
			flag=data
'''
for loc,data in enumerate(vertical_histogram):
    if(data/255 < stroke_width):
        vertical_seg.append(loc)
for i in range(len(vertical_seg)-1):
    if(vertical_seg[i+1] - vertical_seg[i] > 5):
        vertical_break.append(vertical_seg[i+1])
k=1
print vertical_break
'''
for i in vertical_break:
    img[:,i] = 0
cv2.imwrite("charachter.jpg",img)

plt.plot(vertical_histogram)
plt.show()


