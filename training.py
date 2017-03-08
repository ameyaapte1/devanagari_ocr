import cv2
import string

img = cv2.imread('test.tif')
height=img.shape[0]
width=img.shape[1]
word_file = open('test.box',"r")
rawdata = word_file.read()
word=[]
for i in rawdata.split("\n"):
    if(len(i) > 1):
        temp = i.split(' ')
        if(len(temp[0].translate(None,string.whitespace)) != 0):
            word.append([temp[0].decode("utf8"),int(temp[-5]),height-int(temp[-4]),int(temp[-3]),height-int(temp[-2])])

for i in word:
    img = cv2.rectangle(img,(i[1],i[2]),(i[3],i[4]),(0,0,0),2)
cv2.imwrite("test.tif",img)    
