import cv2
import numpy as np
import sys

def get_rect_rank(rect):
    x_mean=(rect[0]+rect[2])/2
    y_mean=(rect[1]+rect[3])/2
    rank = (y/50)*5000+x
    return rank
sys.argv.append("test.tif")
img = cv2.imread(sys.argv[1],cv2.IMREAD_GRAYSCALE)
img[img>=128]=255
img[img<128]=0
contour_img= img.copy()
word_img= img.copy()
img = 255 - img
image, contours, hierarchy = cv2.findContours(img,cv2.RETR_CCOMP,cv2.CHAIN_APPROX_SIMPLE)
contour_rects = []
#out_img = cv2.drawContours(out_img, contours, -1, (0,0,0), 3)
k=0
for (i, j) in zip(contours, hierarchy[0]):
    if cv2.contourArea(i) > 25 and j[3] == -1 :
        """ Minimum Area Rectangle
        rect = cv2.minAreaRect(i)
        box = cv2.boxPoints(rect)
        box = np.int0(box)
        out_img = cv2.drawContours(out_img,[box],0,(0,0,255),2)
        """
        
        x,y,w,h = cv2.boundingRect(i)
        contour_rects.append([x,y,x+w,y+h])
        
        
#print contour_rects
contour_rects.sort(key=lambda x:get_rect_rank(x))
#print contour_rects
for x,y,x_w,y_h in contour_rects:
        #cv2.imwrite('word_' + str(k).zfill(3) + '.jpg',out_img[y:y+h,x:x+w])
        contour_img = cv2.rectangle(contour_img,(x,y),(x_w,y_h),(0,0,0),2)
        k += 1
        cv2.putText(contour_img,str(k),(x,y),cv2.FONT_HERSHEY_SIMPLEX, 1, 2)
cv2.imwrite("contours.jpg",contour_img)
'''
for x,y,x_w,y_h in contour_rects:
    word = word_img[y:y_h,x:x_w].copy()
    k += 1
    cv2.imwrite('word_' + str(k).zfill(3) + '.jpg',word)
'''
