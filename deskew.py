import cv2
import numpy as np
import sys

img = cv2.imread(sys.argv[1],0)
img = 255 - img
if(img.shape[1]<50):
    sys.exit(0)
hough_lines = np.zeros(img.shape)

minLineLength = img.shape[1]
maxLineGap = 1
#longest_line_index=0
#longest_line=0
confiedence = 100
lines = None
longest_line=[0,0]

while(lines is None):
    lines = cv2.HoughLinesP(img,2,np.pi/180,confiedence,minLineLength,maxLineGap)
    #print len(lines)
    confiedence -= 5
    if(confiedence == 50):
        break
if(not(lines is None)):
#lines = cv2.HoughLinesP(img,2,np.pi/180,75,minLineLength,maxLineGap)
    for loc,line in enumerate(lines):
        for x1,y1,x2,y2 in line:
            #dist = np.sqrt(abs((x2-x1)^2-(y2-y1)^2))
            #if(dist > longest_line):
            dist = np.sqrt(abs((x2-x1)^2-(y2-y1)^2))
            angle = np.arctan((y2-y1)/float(x2-x1)) * 180.0/np.pi
            if(dist > longest_line[0] and angle < 30 and angle > -30):
                longest_line[0] = dist
                longest_line[1] = angle
'''
                longest_line = dist
                longest_line_index = loc
x1,y1,x2,y2=lines[longest_line_index][0]
angle = np.arctan((y2-y1)/float(x2-x1)) * 180.0/np.pi
cv2.line(hough_lines,(x1,y1),(x2,y2),(255,255,255),1)
'''
rows,cols = img.shape
rot = cv2.getRotationMatrix2D((cols/2,rows/2),longest_line[1],1)
rotated = 255 - cv2.warpAffine(img,rot,(cols,rows),cv2.INTER_CUBIC)

blur = cv2.GaussianBlur(rotated,(5,5),0)
ret3,th = cv2.threshold(blur,254,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)

cv2.imwrite(sys.argv[2],th)
cv2.imwrite("hough_debug.jpg",hough_lines)
