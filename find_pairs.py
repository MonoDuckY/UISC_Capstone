import cv2
import numpy as np

img = cv2.imread('data/raw data/test_0001.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
h_img, w_img = gray.shape

_, thresh = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)
contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

candidates = []
for c in contours:
    x, y, w, h = cv2.boundingRect(c)
    if x < w_img * 0.15 or x > w_img * 0.85: continue
    if y < h_img * 0.15 or y > h_img * 0.85: continue
    
    area = cv2.contourArea(c)
    if area < 5 or area > 300: continue
    
    hull = cv2.convexHull(c)
    hull_area = cv2.contourArea(hull)
    solidity = float(area)/hull_area if hull_area > 0 else 0
    if solidity > 0.6: continue # Must be cross-like
    
    candidates.append({'c': c, 'area': area, 'w': w, 'h': h, 'solidity': solidity})

print(f"Total candidates: {len(candidates)}")

# Find pairs with very similar area and dimensions
for i in range(len(candidates)):
    for j in range(i+1, len(candidates)):
        c1 = candidates[i]
        c2 = candidates[j]
        
        area_diff = abs(c1['area'] - c2['area']) / max(c1['area'], 1)
        w_diff = abs(c1['w'] - c2['w'])
        h_diff = abs(c1['h'] - c2['h'])
        
        if area_diff < 0.2 and w_diff <= 2 and h_diff <= 2:
            print(f"PAIR MATCH: Area1={c1['area']}, Area2={c2['area']}, w1={c1['w']}, w2={c2['w']}")
