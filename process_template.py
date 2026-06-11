import os
import cv2
import numpy as np

def process_with_template(input_path, output_path, template_paths):
    img = cv2.imread(input_path)
    if img is None:
        print(f"Error reading image: {input_path}")
        return
    
    # Tạo một bản sao để vẽ
    result_img = img.copy()
    
    # Ngưỡng độ chính xác khi match template (0.0 đến 1.0)
    # Giá trị 0.8 thường hoạt động tốt, có thể điều chỉnh nếu cần
    threshold = 0.8 
    
    for template_path in template_paths:
        if not os.path.exists(template_path):
            print(f"Template not found: {template_path}, skipping...")
            continue
            
        template = cv2.imread(template_path)
        h, w = template.shape[:2]
        
        # Thực hiện Template Matching
        res = cv2.matchTemplate(img, template, cv2.TM_CCOEFF_NORMED)
        loc = np.where(res >= threshold)
        
        # Đổi màu các vùng tìm được
        for pt in zip(*loc[::-1]): # pt là tọa độ (x, y) của góc trên cùng bên trái
            # Trích xuất vùng ảnh tương ứng với template
            roi = result_img[pt[1]:pt[1]+h, pt[0]:pt[0]+w]
            
            # Chỉ đổi màu những pixel màu trắng (hoặc xám sáng) trong vùng template đó thành Vàng
            # Điều này giúp giữ nguyên viền đen của caliper
            white_mask = cv2.inRange(roi, np.array([200, 200, 200]), np.array([255, 255, 255]))
            roi[white_mask > 0] = [0, 255, 255] # Chuyển thành màu vàng
            
            result_img[pt[1]:pt[1]+h, pt[0]:pt[0]+w] = roi
            
    cv2.imwrite(output_path, result_img)
    print(f"Processed image saved to: {output_path}")

def main():
    raw_dir = os.path.join("data", "raw data")
    processed_dir = os.path.join("data", "processed data")
    os.makedirs(processed_dir, exist_ok=True)
    
    test_file_name = "test_0001.jpg"
    input_path = os.path.join(raw_dir, test_file_name)
    
    base_name, ext = os.path.splitext(test_file_name)
    output_path = os.path.join(processed_dir, f"{base_name}_processed{ext}")
    
    # Bạn cần cắt 2 ảnh template của caliper và lưu vào thư mục raw data
    templates = [
        os.path.join(raw_dir, "template_plus.jpg"),
        os.path.join(raw_dir, "template_cross.jpg")
    ]
    
    if os.path.exists(input_path):
        process_with_template(input_path, output_path, templates)
    else:
        print(f"Error: Could not find input file at {input_path}")

if __name__ == "__main__":
    main()
