import cv2
import numpy as np

img = cv2.imread('data/raw data/test_0001.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
h_img, w_img = gray.shape

_, thresh = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)
contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

print(f"Total contours: {len(contours)}")
count = 0
for c in contours:
    x, y, w, h = cv2.boundingRect(c)
    
    # Must be in center region
    if x < w_img * 0.15 or x > w_img * 0.85:
        continue
    if y < h_img * 0.15 or y > h_img * 0.85:
        continue
        
    area = cv2.contourArea(c)
    if area < 5 or area > 500:
        continue
        
    aspect_ratio = float(w)/h if h > 0 else 0
    
    hull = cv2.convexHull(c)
    hull_area = cv2.contourArea(hull)
    solidity = float(area)/hull_area if hull_area > 0 else 0
    
    print(f"Center Contour: Area={area:.1f}, w={w}, h={h}, aspect={aspect_ratio:.2f}, solidity={solidity:.2f}")
    count += 1

print(f"Total center contours: {count}")
