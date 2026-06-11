import os
import cv2
import numpy as np

def create_synthetic_templates():
    templates = []
    # Thước đo siêu âm thường có kích thước từ 11 đến 21 pixel
    for size in range(11, 23, 2): 
        # Tạo dấu '+' trắng trên nền đen
        plus = np.zeros((size, size), dtype=np.uint8)
        center = size // 2
        thickness = 2
        plus[center-thickness//2 : center+thickness//2+1, :] = 255
        plus[:, center-thickness//2 : center+thickness//2+1] = 255
        
        # Tạo dấu 'x' (xoay dấu + 45 độ)
        M = cv2.getRotationMatrix2D((center, center), 45, 1.0)
        cross = cv2.warpAffine(plus, M, (size, size))
        
        templates.append(('plus', plus))
        templates.append(('cross', cross))
    return templates

def process_image(input_path, output_path):
    img = cv2.imread(input_path)
    if img is None:
        print(f"Error reading image: {input_path}")
        return
        
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h_img, w_img = gray.shape
    
    # Giới hạn vùng tìm kiếm ở trung tâm để tránh bắt nhầm chữ ở viền
    search_roi = gray[int(h_img*0.15):int(h_img*0.85), int(w_img*0.15):int(w_img*0.85)]
    offset_y, offset_x = int(h_img*0.15), int(w_img*0.15)
    
    templates = create_synthetic_templates()
    best_matches = []
    
    # Chạy Template Matching với tất cả các kích cỡ
    for name, tmpl in templates:
        h, w = tmpl.shape
        res = cv2.matchTemplate(search_roi, tmpl, cv2.TM_CCOEFF_NORMED)
        
        # Ngưỡng 0.65 là đủ chặt chẽ để loại bỏ mô mềm, nhưng vẫn nhận diện được caliper bị mờ do nén ảnh
        threshold = 0.65
        loc = np.where(res >= threshold)
        
        for pt in zip(*loc[::-1]):
            # Lưu lại điểm có score cao (x, y, w, h, score, mask)
            score = res[pt[1], pt[0]]
            best_matches.append((pt[0] + offset_x, pt[1] + offset_y, w, h, score, tmpl))
            
    # Sắp xếp theo score giảm dần
    best_matches.sort(key=lambda x: x[4], reverse=True)
    
    # Lọc Non-Maximum Suppression (NMS) để không vẽ đè nhiều lần lên cùng 1 vị trí
    final_boxes = []
    for match in best_matches:
        x, y, w, h, score, tmpl = match
        overlap = False
        for fx, fy, fw, fh, _, _ in final_boxes:
            # Nếu khoảng cách giữa 2 tâm nhỏ hơn kích thước caliper, coi như là 1
            if abs((x + w/2) - (fx + fw/2)) < w and abs((y + h/2) - (fy + fh/2)) < h:
                overlap = True
                break
        if not overlap:
            final_boxes.append(match)
            # Dấu thước đo trên siêu âm hiếm khi xuất hiện quá 10 cái (thường là 2-4 cái)
            if len(final_boxes) >= 6: 
                break
                
    # Tiến hành tô màu vàng cho các dấu tìm được
    for x, y, w, h, score, tmpl in final_boxes:
        roi = img[y:y+h, x:x+w]
        
        # Chỉ tô màu vào phần rãnh trắng của template (giữ lại viền đen)
        # Sử dụng template như một mask
        mask = tmpl > 128
        
        # Tô màu vàng
        roi[mask] = [0, 255, 255]
        img[y:y+h, x:x+w] = roi

    cv2.imwrite(output_path, img)
    print(f"Processed image saved to: {output_path}. Found {len(final_boxes)} marks.")

def main():
    raw_dir = os.path.join("data", "raw data")
    processed_dir = os.path.join("data", "processed data")
    os.makedirs(processed_dir, exist_ok=True)
    
    test_file_name = "test_0001.jpg"
    input_path = os.path.join(raw_dir, test_file_name)
    
    base_name, ext = os.path.splitext(test_file_name)
    output_path = os.path.join(processed_dir, f"{base_name}_processed{ext}")
    
    if os.path.exists(input_path):
        process_image(input_path, output_path)
    else:
        print(f"Error: Could not find input file at {input_path}")

if __name__ == "__main__":
    main()
