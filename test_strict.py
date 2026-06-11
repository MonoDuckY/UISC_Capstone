import cv2
import numpy as np

img = cv2.imread('data/raw data/test_0001.jpg')

# Strict white threshold
mask = cv2.inRange(img, np.array([245, 245, 245]), np.array([255, 255, 255]))

print(f"Number of pixels > 245: {np.sum(mask > 0)}")

# Dilate slightly to make it visible
kernel = np.ones((3,3), np.uint8)
mask = cv2.dilate(mask, kernel, iterations=1)

img_out = img.copy()
img_out[mask > 0] = [0, 255, 255]

cv2.imwrite('data/processed data/strict_debug.jpg', img_out)
