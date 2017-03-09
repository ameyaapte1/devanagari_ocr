from main import *
img = cv2.imread("../test.tif",cv2.IMREAD_GRAYSCALE)
img[img>=128]=255
img[img<128]=0
feat_file=open("feat.dat","w")
words=get_word_coordinates(img)

for loc,data in enumerate(words):
    deskewed = get_word_image(img,data)
    if(deskewed is not None):
        ch_prop=get_character_properties(deskewed)
        deskewed[ch_prop[0]]=255
        deskewed[ch_prop[1]]=255
        deskewed[ch_prop[2]]=255
        cv2.imwrite("word-"+str(loc).zfill(3)+".jpg",deskewed)
        feat_file.write("word-"+str(loc).zfill(3)+"\n")
        write_feature_vector(ch_prop,deskewed,feat_file)
    
