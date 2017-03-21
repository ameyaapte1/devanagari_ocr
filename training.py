import cv2
import string
import sys
import codecs
from main import *
sys.argv.append('../test.tif')
sys.argv.append('../test.box')
img = cv2.imread(sys.argv[1],cv2.IMREAD_GRAYSCALE)
height=img.shape[0]
width=img.shape[1]
word_file = open(sys.argv[2],"r")
rawdata = word_file.read()
word=[]
contour_img = img.copy()
feat_file=codecs.open("feat.dat","w","utf8")
for i in rawdata.split("\n"):
    if(len(i) > 1):
        temp = i.split(' ')
        if(len(temp[0].translate(None,string.whitespace)) != 0):
            word.append([temp[0].decode("utf8"),int(temp[-5]),height-int(temp[-4]),int(temp[-3]),height-int(temp[-2])])

for loc,data in enumerate(word):
    diff = abs(data[3]-data[1])
    w=[data[3]-diff,data[4],data[1]+diff,data[2]]
    contour_img = cv2.rectangle(contour_img,(w[0],w[1]),(w[2],w[3]),(0,0,0),2)
    word_img = get_word_image(img,w)
    ch_prop=get_character_properties(word_img)
    feat_file.write(" word-"+str(loc).zfill(3)+" "+data[0]+"\n")
    write_feature_vector(ch_prop,get_word_image(img,w),feat_file)
cv2.imwrite("contour.jpg",contour_img)
