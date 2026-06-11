import cv2
import numpy as np

img = cv2.imread('data/raw data/test_0001.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

dst = cv2.cornerHarris(gray, 2, 3, 0.04)
dst = cv2.dilate(dst, None)

img_out = img.copy()
img_out[dst > 0.05 * dst.max()] = [0, 0, 255] # Red dots on corners

cv2.imwrite('data/processed data/harris_debug.jpg', img_out)
