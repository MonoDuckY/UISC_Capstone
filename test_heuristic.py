import cv2
import numpy as np

img = cv2.imread('data/raw data/test_0001.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
h_img, w_img = gray.shape

# Threshold for very bright pixels (accommodating JPEG compression)
_, thresh = cv2.threshold(gray, 220, 255, cv2.THRESH_BINARY)

contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

caliper_contours = []
for c in contours:
    area = cv2.contourArea(c)
    if area < 10 or area > 300: # Size filter
        continue
        
    x, y, w, h = cv2.boundingRect(c)
    aspect_ratio = float(w)/h if h > 0 else 0
    
    if aspect_ratio < 0.5 or aspect_ratio > 2.0: # Roughly square
        continue
        
    hull = cv2.convexHull(c)
    hull_area = cv2.contourArea(hull)
    if hull_area == 0:
        continue
        
    solidity = float(area)/hull_area
    if solidity > 0.6: # Calipers (+, x) have low solidity. Tissue has high solidity.
        continue
        
    # Location filter: Calipers are in the center, text is at the edges
    if x < w_img * 0.15 or x > w_img * 0.85:
        continue
    if y < h_img * 0.15 or y > h_img * 0.85:
        continue

    caliper_contours.append(c)

print(f"Found {len(caliper_contours)} calipers in the center region")

img_out = img.copy()
cv2.drawContours(img_out, caliper_contours, -1, (0, 255, 255), cv2.FILLED)
cv2.imwrite('data/processed data/calipers_heuristic.jpg', img_out)
