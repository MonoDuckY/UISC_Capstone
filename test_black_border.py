import cv2
import numpy as np

img = cv2.imread('data/raw data/test_0001.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
h_img, w_img = gray.shape

_, thresh = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)
contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

calipers = []
for c in contours:
    x, y, w, h = cv2.boundingRect(c)
    
    # Location filter (center)
    if x < w_img * 0.15 or x > w_img * 0.85: continue
    if y < h_img * 0.15 or y > h_img * 0.85: continue
    
    # Size filter
    area = cv2.contourArea(c)
    if area < 5 or area > 300: continue

    # Create a mask for the contour
    mask = np.zeros_like(gray)
    cv2.drawContours(mask, [c], -1, 255, cv2.FILLED)
    
    # Dilate mask to get the border region
    kernel = np.ones((3,3), np.uint8)
    dilated = cv2.dilate(mask, kernel, iterations=2)
    border_mask = cv2.subtract(dilated, mask)
    
    # Calculate mean intensity of the border
    mean_val = cv2.mean(gray, mask=border_mask)[0]
    
    # Crosses and dotted lines are drawn with a black background/border
    if mean_val < 60: # Very dark border
        print(f"FOUND CALIPER: Area={area}, BorderMean={mean_val:.2f}, x={x}, y={y}")
        calipers.append(c)
    else:
        print(f"REJECTED TISSUE: Area={area}, BorderMean={mean_val:.2f}, x={x}, y={y}")

print(f"Found {len(calipers)} calipers based on black border.")

img_out = img.copy()
cv2.drawContours(img_out, calipers, -1, (0, 255, 255), cv2.FILLED)
cv2.imwrite('data/processed data/calipers_black_border.jpg', img_out)
