import cv2
import numpy as np

img = cv2.imread('data/raw data/test_0001.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
h_img, w_img = gray.shape

# Get local minimum in a 7x7 window
kernel = np.ones((7, 7), np.uint8)
local_min = cv2.erode(gray, kernel)

# Overlays are bright (>180) and have a very dark pixel nearby (<50)
overlay_mask = (gray > 180) & (local_min < 50)

# Filter out edges
overlay_mask[:int(h_img*0.15), :] = False
overlay_mask[int(h_img*0.85):, :] = False
overlay_mask[:, :int(w_img*0.15)] = False
overlay_mask[:, int(w_img*0.85):] = False

img_out = img.copy()
img_out[overlay_mask] = [0, 255, 255]

cv2.imwrite('data/processed data/local_min_debug.jpg', img_out)

print(f"Number of overlay pixels found: {np.sum(overlay_mask)}")
