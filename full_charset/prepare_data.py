import subprocess as sp
import os
import sys
#fc-list |grep TTF 
files = os.listdir('FONTS')
sucess = 0
fail = 0
folder = sys.argv[1]+'/'
degrade = rotate = False
fonts = sp.Popen('text2image --fonts_dir ./FONTS --list_available_fonts\
        '.split(), stdout = sp.PIPE, stderr=sp.STDOUT)
if "rotated_img" in folder:
    rotate = True
if "degrade" in folder:
    degrade = True
for fil in files:
    if fil[fil.rfind('.'):] != '.ttf':
        continue
    font = fonts.stdout.readline()[5:-1]
    font_name = font.replace(' ', '-')
    font = '"'+font+'"'
    #font_name = fil[:fil.rfind('.')]
    #font = font_name.replace('-',' ')
    if rotate:
        command = ['text2image', '--output_word_boxes', #'--xsize', '750', '--ysize', '1000',
                '--text', 'text.txt', '--bidirectional_rotation', '--outputbase', folder+font_name, '--font', font,
                '--fonts_dir', './FONTS/']
    elif degrade:
        command = ['text2image', '--output_word_boxes', #'--xsize', '750', '--ysize', '1000',
                '--text', 'text.txt', '--degrade_image', '--outputbase', folder+font_name, '--font', font,
                '--fonts_dir', './FONTS/']
    else:
         command = ['text2image', '--output_word_boxes', #'--xsize', '750', '--ysize', '1000',
                '--text', 'text.txt', '--outputbase', folder+font_name, '--font', font,
                '--fonts_dir', './FONTS/']
        
    print " ".join(command)
    if bool(sp.call(' '.join(command), shell = True)):
        fail+=1
    else:
        sucess+=1
print "failed fonts no =", fail
print "success fonts no =", sucess
print "succes rate", (float(sucess)/(sucess + fail))
