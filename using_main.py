from main import *
img = cv2.imread("binary.jpg",cv2.IMREAD_GRAYSCALE)
img[img>=128]=255
img[img<128]=0
feat_file=open("feat.dat","w")
words=get_word_coordinates(img)

for loc,data in enumerate(words):
    deskewed = get_deskewed_word(get_word_image(img,data))
    if(deskewed is not None):
        ch_prop=get_character_properties(deskewed)
        feat_file.write("word-"+str(loc).zfill(3)+"\n")
        write_feature_vector(ch_prop,deskewed,feat_file)
    
