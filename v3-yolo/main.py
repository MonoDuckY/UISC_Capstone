from ultralytics import YOLO
import cv2
import os
import numpy as np
import glob
# import easyocr (Không sử dụng EasyOCR nữa)

# 1. KHỞI TẠO MÔ HÌNH
MODEL_PATH = r"E:\FPT\SET490\trainYOLO\runs\segment\train-4\weights\best.pt"
model = YOLO(MODEL_PATH)

# Gỡ bỏ Reader của EasyOCR

def process_single_image(image_path, out_folder_visual, out_folder_cleaned):
    img = cv2.imread(image_path)
    if img is None:
        print(f"❌ Không đọc được ảnh: {image_path}")
        return

    h, w, _ = img.shape

    # -------------------------------------------------------------
    # BƯỚC 1: DỰ ĐOÁN YOLO & TẠO MẶT NẠ VÙNG AN TOÀN (SAFE MASK)
    # -------------------------------------------------------------
    results = model(img, conf=0.5, verbose=False)
    
    img_visual = img.copy()
    mask_overlay = np.zeros_like(img, dtype=np.uint8)
    safe_mask_binary = np.zeros((h, w), dtype=np.uint8)

    for result in results:
        if result.masks is None:
            continue
            
        for segments, box, cls, score in zip(result.masks.xy, result.boxes.xyxy, result.boxes.cls, result.boxes.conf):
            class_id = int(cls)
            
            if class_id in [0, 1]:  # Class 0 = fan, 1 = wave
                x1, y1, x2, y2 = map(int, box)
                cv2.rectangle(img_visual, (x1, y1), (x2, y2), (0, 255, 0), 3)
                conf_value = score.item()
                label = f"{model.names[class_id]} {conf_value:.2f}"
                cv2.putText(img_visual, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                
                polygon = np.array(segments, dtype=np.int32)
                cv2.fillPoly(mask_overlay, [polygon], (0, 255, 255))
                cv2.fillPoly(safe_mask_binary, [polygon], 255)

    alpha = 0.4
    cv2.addWeighted(mask_overlay, alpha, img_visual, 1 - alpha, 0, img_visual)

    # -------------------------------------------------------------
    # BƯỚC 2: BLACKOUT NỀN NGOẠI TRỪ VÙNG AN TOÀN (SAFE ZONE)
    # -------------------------------------------------------------
    # Nhộm đen hoàn toàn các pixel nằm ngoài vùng siêu âm y khoa được phát hiện bởi YOLO
    img_cleaned = cv2.bitwise_and(img, img, mask=safe_mask_binary)

    # -------------------------------------------------------------
    # BƯỚC 5: LƯU KẾT QUẢ
    # -------------------------------------------------------------
    filename = os.path.basename(image_path)
    base, ext = os.path.splitext(filename)
    
    path_visual = os.path.join(out_folder_visual, f"{base}_visual{ext}")
    cv2.imwrite(path_visual, img_visual)
    
    path_cleaned = os.path.join(out_folder_cleaned, f"{base}_cleaned{ext}")
    cv2.imwrite(path_cleaned, img_cleaned)
    
    print(f"✅ Đã xử lý tối ưu: {filename}")


def process_batch(input_dir, output_parent_dir):
    # Định nghĩa cấu trúc 2 thư mục đầu ra bên trong thư mục output chung
    out_folder_visual = os.path.join(output_parent_dir, "visual_safe_zone")
    out_folder_cleaned = os.path.join(output_parent_dir, "cleaned_images")
    
    # Tạo thư mục nếu chưa tồn tại
    os.makedirs(out_folder_visual, exist_ok=True)
    os.makedirs(out_folder_cleaned, exist_ok=True)

    # Quét ảnh đầu vào loại bỏ trùng lặp
    valid_extensions = {'.jpg', '.jpeg', '.png', '.bmp'}
    image_paths = []
    
    if not os.path.exists(input_dir):
        print(f"❌ Không tìm thấy thư mục input: {input_dir}")
        return

    for filename in os.listdir(input_dir):
        ext = os.path.splitext(filename)[1].lower()
        if ext in valid_extensions:
            image_paths.append(os.path.join(input_dir, filename))

    total_images = len(image_paths)
    print(f"🔍 Tìm thấy {total_images} ảnh cần xử lý.")

    for idx, img_path in enumerate(image_paths, start=1):
        print(f"\n[{idx}/{total_images}] Processing...")
        process_single_image(img_path, out_folder_visual, out_folder_cleaned)
        
    print("\n🎉 HOÀN THÀNH XỬ LÝ TOÀN BỘ FOLDER!")


if __name__ == "__main__":
    INPUT_FOLDER = r"E:\FPT\SET490\useYOLOdetect\input" 
    OUTPUT_PARENT = r"E:\FPT\SET490\useYOLOdetect\output"   
    
    process_batch(INPUT_FOLDER, OUTPUT_PARENT)