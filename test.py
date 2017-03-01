import cv2
import numpy as np
from matplotlib import pyplot as plt

img = cv2.imread('binary.jpg',0)
edges = cv2.Canny(img,100,200)


plt.subplot(122),plt.imshow(edges,cmap = 'gray')
plt.title('Edge Image'), plt.xticks([]), plt.yticks([])

plt.show()
