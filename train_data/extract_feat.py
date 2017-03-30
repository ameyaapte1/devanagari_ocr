import cv2
import string
import sys
import codecs
from main import *
import os
import subprocess
import numpy as np
count = 0
errors = []
folder = sys.argv[1] + '/'
resolution = (18,14)
images = os.listdir(folder)
for filee in images:
    filee = folder+filee
    extension = filee[filee.rfind('.'):]
    without_extension = filee[:filee.rfind('.')]
    if extension != '.tif':
        continue
    img = cv2.imread(filee,cv2.IMREAD_GRAYSCALE)
    height=img.shape[0]
    width=img.shape[1]
    word_file = open(without_extension+ '.box',"r")
    rawdata = word_file.read()
    word=[]
    contour_img = img.copy()
    feat_file=codecs.open(without_extension + "_feat.dat","w","utf8")
    print "Processing %s ..."%(without_extension)
    for i in rawdata.split("\n"):
        if(len(i) > 1):
            temp = i.split(' ')
            if(len(temp[0].translate(None,string.whitespace)) != 0):
                word.append([temp[0].decode("utf8"),int(temp[-5]),height-int(temp[-4]),int(temp[-3]),height-int(temp[-2])])
    #print filee, without_extension
    for loc,data in enumerate(word):
        diff = abs(data[3]-data[1])
        w=[data[3]-diff,data[4],data[1]+diff,data[2]]
        contour_img = cv2.rectangle(contour_img,(w[0],w[1]),(w[2],w[3]),(0,0,0),2)
        word_img = np.array(get_word_image(img,w))
        resized = cv2.resize(word_img,resolution,cv2.INTER_AREA)
        #ch_prop=get_character_properties(word_img)
        unicode_ = str(ord(data[0]))
        feat_file.write(" word-"+str(loc).zfill(3)+" "+data[0]+" "+unicode_+"\n")
        #feat_file.write("character: middle "+"\n")
        for col in resized:
            for row in col:
                feat_file.write("0" if row < 64 else " ")
            feat_file.write("\n")
        #feat_file.write("\n\n")
"""
        try:
            write_feature_vector(ch_prop,get_word_image(img,w),feat_file)
        except:
            count+=1
            errors.append(filee)
"""
cv2.imwrite("contour.jpg",contour_img)
