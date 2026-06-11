import cv2
import numpy as np

img = cv2.imread('data/raw data/test_0001.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

def get_templates():
    templates = []
    # Test a few reasonable sizes for the caliper
    for size in range(11, 23, 2): 
        # Create a '+' template
        plus = np.zeros((size, size), dtype=np.uint8)
        center = size // 2
        thickness = 2
        plus[center-thickness//2 : center+thickness//2+1, :] = 255
        plus[:, center-thickness//2 : center+thickness//2+1] = 255
        
        # Create a 'x' template (rotate the plus)
        M = cv2.getRotationMatrix2D((center, center), 45, 1.0)
        cross = cv2.warpAffine(plus, M, (size, size))
        
        # We need a mask because we only care about the white shape
        # The background in the template is 0, which might match dark tissue.
        # But ultrasound background is also dark, so maybe it's fine.
        templates.append(('plus', plus))
        templates.append(('cross', cross))
    return templates

templates = get_templates()
img_out = img.copy()
found_any = False

for name, tmpl in templates:
    h, w = tmpl.shape
    res = cv2.matchTemplate(gray, tmpl, cv2.TM_CCOEFF_NORMED)
    
    threshold = 0.60
    loc = np.where(res >= threshold)
    
    # Non-maximum suppression to avoid drawing multiple times
    pts = list(zip(*loc[::-1]))
    if not pts: continue
    
    # Just draw them all for debug
    for pt in pts:
        cv2.rectangle(img_out, pt, (pt[0] + w, pt[1] + h), (0, 255, 255), 1)
        found_any = True

cv2.imwrite('data/processed data/synthetic_debug.jpg', img_out)
print(f"Synthetic templates found matches: {found_any}")
