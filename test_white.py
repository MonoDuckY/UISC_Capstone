import cv2
import numpy as np

img = cv2.imread('data/raw data/complex_cyst_0001.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

print("Max pixel value:", np.max(gray))
# Count pixels > 240
print("Pixels > 240:", np.sum(gray > 240))
print("Pixels == 255:", np.sum(gray == 255))
