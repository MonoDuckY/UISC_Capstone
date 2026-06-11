import cv2
import numpy as np

img = cv2.imread('data/raw data/test_0001.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Threshold
_, thresh = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY)

# Find contours
contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

img_out = img.copy()
print(f"Found {len(contours)} contours")
for i, c in enumerate(contours):
    area = cv2.contourArea(c)
    x, y, w, h = cv2.boundingRect(c)
    aspect_ratio = float(w)/h if h > 0 else 0
    # Print stats
    print(f"Contour {i}: Area={area}, w={w}, h={h}, aspect={aspect_ratio:.2f}")
    
    # Draw all contours with blue bounding boxes
    cv2.rectangle(img_out, (x, y), (x+w, y+h), (255, 0, 0), 1)
    cv2.putText(img_out, str(i), (x, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255,0,0), 1)

cv2.imwrite('data/processed data/contours_debug.jpg', img_out)
