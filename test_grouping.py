import cv2
import numpy as np
import os

def process_image_by_border(input_path, output_path):
    img = cv2.imread(input_path)
    if img is None:
        print(f"Error reading image: {input_path}")
        return
        
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h_img, w_img = gray.shape
    
    # 1. Threshold to find bright white pixels
    _, thresh = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)
    
    # 2. Find contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    candidate_contours = []
    candidate_centers = []
    
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        
        # Avoid the borders where text/metadata lies
        if x < w_img * 0.10 or x > w_img * 0.90: continue
        if y < h_img * 0.10 or y > h_img * 0.90: continue
        
        # Caliper arms are small
        area = cv2.contourArea(c)
        if area < 3 or area > 150: continue
        
        # Calculate black border metric
        mask = np.zeros_like(gray)
        cv2.drawContours(mask, [c], -1, 255, cv2.FILLED)
        
        # Dilate mask to get the border
        kernel = np.ones((3, 3), np.uint8)
        dilated = cv2.dilate(mask, kernel, iterations=2)
        border_mask = cv2.subtract(dilated, mask)
        
        # Mean intensity of the border
        mean_val = cv2.mean(gray, mask=border_mask)[0]
        
        # Caliper markers have a very distinct dark/black outline/border
        if mean_val < 65:
            candidate_contours.append((c, area, mean_val))
            candidate_centers.append((x + w/2, y + h/2))
            
    print(f"Found {len(candidate_contours)} candidate caliper arms.")
    
    # 3. Group candidate arms that are close to each other (within 25 pixels)
    # This filters out any isolated random noise that happens to have a dark border
    num_candidates = len(candidate_contours)
    groups = []
    visited = [False] * num_candidates
    
    for i in range(num_candidates):
        if visited[i]: continue
        
        # Start a new group
        group = [i]
        visited[i] = True
        
        # Find all other candidates close to this one
        for j in range(num_candidates):
            if visited[j]: continue
            
            # Distance between centers
            dist = np.sqrt((candidate_centers[i][0] - candidate_centers[j][0])**2 + 
                           (candidate_centers[i][1] - candidate_centers[j][1])**2)
            if dist < 25:
                group.append(j)
                visited[j] = True
                
        groups.append(group)
        
    # 4. Draw/Color the valid calipers
    img_out = img.copy()
    valid_count = 0
    
    for group in groups:
        # A valid caliper must have at least 2 arms detected near each other
        # (since '+' has 4 arms, and 'x' has at least 2 separate diagonal segments)
        if len(group) >= 2:
            valid_count += 1
            print(f"Caliper Group {valid_count} contains {len(group)} arms. Coloring them yellow.")
            for idx in group:
                c, area, border_mean = candidate_contours[idx]
                # Color the white pixels of this contour yellow
                cv2.drawContours(img_out, [c], -1, (0, 255, 255), cv2.FILLED)
                
    cv2.imwrite(output_path, img_out)
    print(f"Saved processed image to: {output_path}. Identified {valid_count} caliper pairs/markers.")

if __name__ == "__main__":
    process_image_by_border("data/raw data/test_0001.jpg", "data/processed data/test_0001_grouped.jpg")
