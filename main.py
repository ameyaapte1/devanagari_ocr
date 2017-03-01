import cv2
import numpy as np
import sys
import os

def get_word_image(img,rect):
    x,y,x_w,y_h=rect
    return img[y:y_h,x:x_w]
def get_characters_image(word_img,vertical_break):
    characters=[]
    for i in range(0,len(vertical_break)-1,2):
        characters.append(word_img[:,vertical_break[i]:vertical_break[i+1]])
    return characters
def write_word(img,word_coordinates,file_name):
    x,y,x_w,y_h=word_coordinates
    cv2.imwrite(file_name,img[y:y_h,x:x_w])
def get_word_coordinates(input_img,debug=False): #Returns list of coordinates as a list. It contains [x,y,x+w,y+h]
    binary_img=input_img.copy()         #Original image is not modified

    binary_img[binary_img>=128]=255
    binary_img[binary_img<128]=0

    binary_img_area=binary_img.shape[0]*binary_img.shape[1]

    out_binary_img =binary_img.copy()
    debug_binary_img= binary_img.copy()

    binary_img = 255 - binary_img

    image, contours, hierarchy = cv2.findContours(binary_img,cv2.RETR_CCOMP,cv2.CHAIN_APPROX_SIMPLE)
    word_coord = []
    #out_binary_img = cv2.drawContours(out_binary_img, contours, -1, (0,0,0), 3)
    k=0
    cnt=0
    for (i, j) in zip(contours, hierarchy[0]):
        if cv2.contourArea(i) > binary_img_area/150000 and j[3] == -1:
            """ Minimum Area Rectangle
            rect = cv2.minAreaRect(i)
            box = cv2.boxPoints(rect)
            box = np.int0(box)
            out_binary_img = cv2.drawContours(out_binary_img,[box],0,(0,0,255),2)
            """
            
            x,y,w,h = cv2.boundingRect(i)
            word_coord.append([x,y,x+w,y+h])

            if(debug):
                #cv2.imwrite('word_' + str(k).zfill(3) + '.jpg',out_binary_img[y:y+h,x:x+w])
                debug_binary_img = cv2.rectangle(debug_binary_img,(x,y),(x+w,y+h),(0,0,0),2)
                cnt += 1
                cv2.putText(debug_binary_img,str(cnt),(x,y),cv2.FONT_HERSHEY_SIMPLEX, 1, 2)
    if(debug):
        cv2.imwrite("contours.jpg",debug_binary_img)
    return word_coord

def get_deskewed_word(input_word_img):    #Returns the deskewed word as a numpy array
    word_img = input_word_img.copy()

    word_img = 255-word_img
    
    minLineLength = word_img.shape[1]
    maxLineGap = 20
    #longest_line_index=0
    longest_line=[0,0]
    confiedence = 100
    lines = None

    while(lines is None):
        lines = cv2.HoughLinesP(word_img,3,np.pi/180,confiedence,minLineLength,maxLineGap)
        #print len(lines)
        confiedence -= 5
        if(confiedence == 50):
            break
    if(not(lines is None)):
        for loc,line in enumerate(lines):
            for x1,y1,x2,y2 in line:
                dist = np.sqrt(abs((x2-x1)^2-(y2-y1)^2))
                angle = np.arctan((y2-y1)/float(x2-x1)) * 180.0/np.pi
                if(dist > longest_line[0] and angle < 30 and angle > -30):
                    longest_line[0] = dist
                    longest_line[1] = angle
                    #longest_line_index = loc
        #x1,y1,x2,y2=lines[longest_line_index][0]
        #angle = np.arctan((y2-y1)/float(x2-x1)) * 180.0/np.pi

        rows,cols = word_img.shape
        rot = cv2.getRotationMatrix2D((cols/2,rows/2),longest_line[1],1)
        rotated = 255 - cv2.warpAffine(word_img,rot,(cols,rows),cv2.INTER_CUBIC)

        blur = cv2.GaussianBlur(rotated,(5,5),0)
        ret3,deskewed = cv2.threshold(blur,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        return deskewed
    else:
        return word
def get_character_coordinates(deskewed_img,crop_type):
 
    horizontal_histogram = (255*deskewed_img.shape[1])-deskewed_img.sum(axis=1)

    #plt.plot(horizontal_histogram,'ro')
    #plt.show()

    upper_start=-1
    lower_start=-1
    find_upper=True

    for loc,i in enumerate(horizontal_histogram):
        if( i/255 > deskewed_img.shape[1]/2 and upper_start==-1 and find_upper and horizontal_histogram[loc]<=horizontal_histogram[loc+1] and horizontal_histogram[loc+1] >= horizontal_histogram[loc+2]):
            upper_start=loc+1
            #print upper_start
        if( upper_start!=-1 and find_upper and horizontal_histogram[loc]>=horizontal_histogram[loc+1] and horizontal_histogram[loc+1] <= horizontal_histogram[loc+2]):
            upper_end = loc+1
            #print upper_end
            find_upper = False
        #and horizontal_histogram[loc]>=horizontal_histogram[loc+1] and horizontal_histogram[loc+1] <= horizontal_histogram[loc+2]
        if(i/255 < deskewed_img.shape[1]/4 and loc > deskewed_img.shape[0]*3/4 and lower_start==-1 and not find_upper ):
            lower_start=loc+1
            break
    #lower_start=horizontal_histogram[upper_end:].argmin()
    stroke_width = upper_end-upper_start
    #cv2.imwrite(sys.argv[1].replace(".jpg","")+"_"+str(k)+"_upper.jpg",deskewed_img[:upper_start])
    #cv2.imwrite(sys.argv[1].replace(".jpg","")+"_"+str(k)+"_lower.jpg",deskewed_img[lower_start:])
    #cv2.imwrite(sys.argv[1].replace(".jpg","")+"_middle.jpg",deskwewd[upper_end:lower_start])
    print upper_start
    print upper_end    
    if(crop_type==1):
        deskewed_img = deskewed_img[upper_end:]
    elif(crop_type==2):
        deskewed_img = deskewed_img[upper_end:lower_start]
    vertical_histogram = (255*deskewed_img.shape[0])-deskewed_img.sum(axis=0)

    vertical_seg=[]
    vertical_break=[]
    vertical_break.append(0)
    '''
    percentile=[]

    for i in range(0,50,2):
            percentile.append(np.percentile(vertical_histogram,i))
    '''
    flag=False
    #for loc,data in enumerate(vertical_histogram < np.percentile(vertical_histogram,np.argmax(np.gradient(percentile))*2)):
    for loc,data in enumerate(vertical_histogram > 255*stroke_width/2):
            if(data):
                    if(not flag):
                        if(loc-vertical_break[-1] > 5):
                            vertical_break.append(loc)
                        flag=True
            else:
                    if(flag):
                        if(loc-vertical_break[-1] > 5):
                            vertical_break.append(loc)
                        flag=False
    vertical_break.remove(0)
    '''
    for loc,data in enumerate(vertical_histogram):
        if(data/255 < stroke_width):
            vertical_seg.append(loc)
    for i in range(len(vertical_seg)-1):
        if(vertical_seg[i+1] - vertical_seg[i] > 3):
            vertical_break.append(vertical_seg[i])
            vertical_break.append(vertical_seg[i+1])
    #print vertical_break
    '''
    return vertical_break
    '''
    spacer_array=np.zeros((deskewed_img.shape[0],2))
    spacer_array += 255
    spacer_array[:,1]=0
    output = np.zeros((deskewed_img.shape[0],0))
    for i in range(len(vertical_break)-1):
        character=deskewed_img[:,vertical_break[i]:vertical_break[i+1]].copy()
        output=np.concatenate((output,deskewed_img[:,vertical_break[i]:vertical_break[i+1]],spacer_array),axis=1)
    if(not os.path.isfile("./word"+"_"+str(k).zfill(3)+".jpg")):
        k += 1
        cv2.imwrite("word"+"_"+str(k).zfill(3)+".jpg",output)    
    if(not os.path.isfile("./word"+"_"+str(k).zfill(3)+".feat")):
            k += 1
            fout = open("word"+"_"+str(k).zfill(3)+".feat","w")   
    #cv2.imwrite(sys.argv[2],deskewed_img)
    #plt.plot(vertical_histogram)
    #plt.show()
    '''

