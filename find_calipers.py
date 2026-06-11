import cv2
import numpy as np

img = cv2.imread('data/raw data/test_0001.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Threshold for very bright pixels
_, thresh = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY)

contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

caliper_contours = []
for c in contours:
    area = cv2.contourArea(c)
    if area < 10 or area > 200: # Crosses are small, but not tiny noise
        continue
        
    x, y, w, h = cv2.boundingRect(c)
    aspect_ratio = float(w)/h if h > 0 else 0
    
    # Crosses should be roughly square
    if aspect_ratio < 0.7 or aspect_ratio > 1.3:
        continue
        
    hull = cv2.convexHull(c)
    hull_area = cv2.contourArea(hull)
    if hull_area == 0:
        continue
        
    solidity = float(area)/hull_area
    
    # A cross (+ or x) has a low solidity because it's not a solid block
    # Usually < 0.6
    print(f"Candidate: Area={area}, w={w}, h={h}, aspect={aspect_ratio:.2f}, solidity={solidity:.2f}")
    if solidity < 0.65:
        caliper_contours.append(c)

print(f"Found {len(caliper_contours)} calipers")

# Draw them
img_out = img.copy()
cv2.drawContours(img_out, caliper_contours, -1, (0, 255, 255), cv2.FILLED)
cv2.imwrite('data/processed data/calipers_detected.jpg', img_out)
