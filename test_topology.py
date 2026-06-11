import cv2
import numpy as np

# You may need cv2.ximgproc for thinning. If not available, we can implement it or use morphology.
img = cv2.imread('data/raw data/test_0001.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
h_img, w_img = gray.shape

# Threshold
_, thresh = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)

# Location filter to ignore edges
mask = np.zeros_like(thresh)
mask[int(h_img*0.15):int(h_img*0.85), int(w_img*0.15):int(w_img*0.85)] = 255
thresh = cv2.bitwise_and(thresh, mask)

# Thinning (Skeletonization)
# Since cv2.ximgproc might not be installed, we use a simple morphological skeletonization
skeleton = np.zeros(thresh.shape, np.uint8)
eroded = np.zeros(thresh.shape, np.uint8)
temp = np.zeros(thresh.shape, np.uint8)

kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3,3))
iters = 0
while True:
    cv2.erode(thresh, kernel, eroded)
    cv2.dilate(eroded, kernel, temp)
    cv2.subtract(thresh, temp, temp)
    cv2.bitwise_or(skeleton, temp, skeleton)
    thresh = eroded.copy()
    iters += 1
    if cv2.countNonZero(thresh) == 0 or iters > 100:
        break

# Now skeleton contains the 1-pixel thick skeleton.
# Let's find branch points (junctions) using a Hit-or-Miss approach or simple neighbor counting
junctions = []
for y in range(1, h_img - 1):
    for x in range(1, w_img - 1):
        if skeleton[y, x] == 255:
            # Count neighbors
            neighbors = (skeleton[y-1:y+2, x-1:x+2] == 255).sum() - 1
            if neighbors >= 3: # 3 is T-junction, 4 is X-junction or +-junction
                junctions.append((x, y, neighbors))

print(f"Found {len(junctions)} junctions in skeleton")

img_out = img.copy()
for x, y, n in junctions:
    cv2.circle(img_out, (x, y), 5, (0, 0, 255), -1)

cv2.imwrite('data/processed data/topology_debug.jpg', img_out)
