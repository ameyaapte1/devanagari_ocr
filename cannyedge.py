import cv2
import numpy as np

img = cv2.imread('binary.jpg',0)
edges = cv2.Canny(img,100,200)

cv2.imwrite("edges.jpg",255-edges)
