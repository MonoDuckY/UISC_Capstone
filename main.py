import cv2
import easyocr
import pandas as pd
import numpy as np
import os
import glob
import re

def process_images(input_dir='input_images', output_dir='output_images'):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    print("Đang khởi động AI: Tách biệt bộ lọc Xóa và bộ lọc Xuất Excel...")
    reader = easyocr.Reader(['vi', 'en'])
    
    all_text_data = []
    image_paths = glob.glob(os.path.join(input_dir, '*.[jp][pn]*[g]')) 
    
    if not image_paths:
        print(f"Không tìm thấy ảnh nào trong '{input_dir}'.")
        return

    for img_path in image_paths:
        filename = os.path.basename(img_path)
        print(f"\nĐang xử lý: {filename}")
        
        img = cv2.imread(img_path)
        if img is None: continue
        
        # Hạ thấp ngưỡng để AI quét được cả những đốm chữ mờ nhất
        results = reader.readtext(img, text_threshold=0.05, low_text=0.05)
        
        mask = np.zeros(img.shape[:2], dtype=np.uint8)
        detected_texts = []
        img_h, img_w = img.shape[:2]
        
        for (bbox, text, prob) in results:
            tl = np.min(bbox, axis=0).astype(int)
            br = np.max(bbox, axis=0).astype(int)
            
            center_x = (tl[0] + br[0]) / 2
            center_y = (tl[1] + br[1]) / 2
            
            # 1. BẢO VỆ TỬ CUNG/THAI NHI
            if (0.25 * img_w < center_x < 0.75 * img_w) and (0.25 * img_h < center_y < 0.85 * img_h):
                continue # Nếu nằm trong tâm ảnh thì bỏ qua toàn bộ
            
            # 2. XÓA MỌI THỨ (Không cần biết là chữ xịn hay chữ rác)
            pad = 5
            x_min = max(0, tl[0] - pad)
            y_min = max(0, tl[1] - pad)
            x_max = min(img.shape[1], br[0] + pad)
            y_max = min(img.shape[0], br[1] + pad)
            
            # Lấy vùng ảnh và tìm điểm sáng (ngưỡng 110 giúp bắt cả nét chữ màu xám)
            roi = img[y_min:y_max, x_min:x_max]
            roi_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            _, roi_thresh = cv2.threshold(roi_gray, 110, 255, cv2.THRESH_BINARY)
            
            # Nhập điểm sáng vào mặt nạ xóa
            mask[y_min:y_max, x_min:x_max] = cv2.bitwise_or(mask[y_min:y_max, x_min:x_max], roi_thresh)
            
            # 3. LỌC CHỮ CHO EXCEL (Độc lập với lệnh xóa ở trên)
            clean_text = re.sub(r'[^\w\s\.,:\-\/]', '', text).strip()
            # Chỉ lưu những chuỗi dài từ 2 ký tự trở lên vào file
            if clean_text and len(clean_text) >= 2: 
                detected_texts.append(clean_text)
                
        print(" -> Đang xóa chữ và bảo toàn nền...")
        # Nở mặt nạ 1 chút để nuốt gọn các viền bóng mờ quanh nét chữ
        kernel = np.ones((3, 3), np.uint8)
        mask = cv2.dilate(mask, kernel, iterations=1)
        
        # Inpaint TELEA lấp nền trên nền đen tốt hơn
        cleaned_img = cv2.inpaint(img, mask, 3, cv2.INPAINT_TELEA)
        
        all_text_data.append({
            'Tên File': filename,
            'Nội dung text': ' | '.join(detected_texts)
        })
        
        out_path = os.path.join(output_dir, f"cleaned_{filename}")
        cv2.imwrite(out_path, cleaned_img)
        print(" -> Hoàn thành!")
        
    if all_text_data:
        pd.DataFrame(all_text_data).to_excel('KetQua_NhanDien.xlsx', index=False)
        print("\n=> Đã xuất báo cáo và làm sạch ảnh thành công!")

if __name__ == '__main__':
    process_images()