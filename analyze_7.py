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
    if solidity > 0.6: continue 
    
    aspect_ratio = float(w)/h if h > 0 else 0
    
    candidates.append({'c': c, 'area': area, 'w': w, 'h': h, 'aspect': aspect_ratio, 'solidity': solidity})

print(f"Total candidates: {len(candidates)}")
for idx, cand in enumerate(candidates):
    print(f"Cand {idx}: Area={cand['area']}, w={cand['w']}, h={cand['h']}, aspect={cand['aspect']:.2f}, solidity={cand['solidity']:.2f}")

