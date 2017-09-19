from main import *
def generate_feat(image):    
    import cv2
    import string
    import sys
    import codecs
    import numpy as np
    sys.argv.append(image)
    box_file = (image[:-4]+'.box')
    print box_file 
    img = cv2.imread(image,cv2.IMREAD_GRAYSCALE)
    height=img.shape[0]
    width=img.shape[1]
    word_file = open(box_file,"r")
    rawdata = word_file.read()
    word=[]
    char_height=[]
    feat_file=codecs.open("feat.dat","w","utf8")
    for i in rawdata.split("\n"):
        if(len(i) > 1):
            temp = i.split(' ')
            if(len(temp[0].translate(None,string.whitespace)) != 0):
                word.append([temp[0].decode("utf8"),int(temp[-5]),height-int(temp[-4]),int(temp[-3]),height-int(temp[-2])])
                char_height.append(int(temp[-2])-int(temp[-4]))
    avg_height = int(np.mean(char_height))
    print avg_height
    res = np.zeros((len(word),16,20))
    for loc,data in enumerate(word):
        diff = abs(data[3]-data[1])
        if avg_height - (data[2] - data[4]) > 0:
            data[4] -= avg_height - (data[2] - data[4])
        w=[data[3]-diff,data[4],data[1]+diff,data[2]]
        word_img = get_word_image(img,w)
        resized = cv2.resize(word_img,(20,16),cv2.INTER_AREA)
        for x in range(len(resized)):
            for y in range(len(resized[x])):
                if resized[x][y] > 127:
                    resized[x][y] = 255.
                else:
                    resized[x][y] = 0.
        res[loc]=resized
        '''
        ch_prop=get_character_properties(word_img)
        feat_file.write(" word-"+str(loc).zfill(3)+" "+data[0]+" upper_start:"+str(ch_prop[0])+"upper_end:"+str(ch_prop[1])+"lower:"+str(ch_prop[2])+"\n")
        write_feature_vector(ch_prop,get_word_image(img,w),feat_file)
        '''
    np.save(open(image[:-4]+'.dat','w'),res)
    #cv2.imwrite("contour.jpg",contour_img)
if __name__ == '__main__':
    foldername = 'images/'
    import os
    images = os.listdir(foldername)
    for image in images:
        if image[-4:] =='.tif':
            print image
            generate_feat(foldername + image)
