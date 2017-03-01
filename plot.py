import cv2
import numpy as np
import sys
from matplotlib import pyplot as plt
sys.argv.append('word_004.jpg')
img = cv2.imread(sys.argv[1],0)

horizontal_histogram = (255*img.shape[1])-img.sum(axis=1)
vertical_histogram = (255*img.shape[0])-img.sum(axis=0)


plt.plot(horizontal_histogram,'ro')
plt.show()

plt.plot(vertical_histogram,'bo')
plt.show()
