import os
import glob
import cv2
import easyocr
import numpy as np

def clean_all_ultrasound_images(input_dir, output_dir):
    # 1. Kiểm tra và tạo thư mục output nếu chưa tồn tại
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Đã tạo thư mục output tại: {output_dir}")

    # 2. Khởi tạo bộ đọc EasyOCR (Chỉ dùng để bắt các text chú thích "lọt khe" ở giữa ảnh nếu có)
    print("Đang khởi tạo EasyOCR...")
    reader = easyocr.Reader(['en'])

    # 3. Lấy danh sách tất cả các file ảnh trong thư mục input
    extensions = ('*.png', '*.jpg', '*.jpeg', '*.BMP', '*.PNG', '*.JPG')
    image_paths = []
    for ext in extensions:
        image_paths.extend(glob.glob(os.path.join(input_dir, ext)))

    if not image_paths:
        print(f"Không tìm thấy ảnh nào trong thư mục: {input_dir}")
        return

    print(f"Tìm thấy {len(image_paths)} ảnh cần xử lý.\n---")

    # 4. Vòng lặp xử lý từng ảnh
    for idx, img_path in enumerate(image_paths, 1):
        filename = os.path.basename(img_path)
        print(f"[{idx}/{len(image_paths)}] Đang xử lý: {filename}...", end="")

        img = cv2.imread(img_path)
        if img is None:
            print(" -> LỖI: Không thể đọc ảnh.")
            continue

        h, w = img.shape[:2]

        # Tạo một ảnh mask để dùng riêng cho các chữ ngẫu nhiên ở giữa ảnh (nếu có)
        ocr_mask = np.zeros((h, w), dtype=np.uint8)

        # -----------------------------------------------------------------
        # BƯỚC 1: DÙNG EASYOCR ĐỂ QUÉT CHỮ TRONG VÙNG TRUNG TÂM
        # -----------------------------------------------------------------
        results = reader.readtext(img_path)
        for (bbox, text, prob) in results:
            if prob > 0.30:
                top_left = tuple(map(int, bbox[0]))
                bottom_right = tuple(map(int, bbox[2]))
                
                # Chỉ lấy các chữ nằm biệt lập ở giữa (tránh động vào các viền biên cố định)
                if top_left[0] > 130 and bottom_right[0] < (w - 50) and top_left[1] > 50 and bottom_right[1] < (h - 60):
                    cv2.rectangle(ocr_mask, 
                                  (max(0, top_left[0] - 4), max(0, top_left[1] - 4)), 
                                  (min(w, bottom_right[0] + 4), min(h, bottom_right[1] + 4)), 
                                  255, -1)

        # Inpaint TRƯỚC cho các text nhỏ lọt thỏm ở giữa ảnh (ví dụ dấu đo kích thước, text chú thích mô)
        # Bước này không làm lem viền vì nó nằm hoàn toàn trong vùng xám của siêu âm
        result_img = cv2.inpaint(img, ocr_mask, inpaintRadius=3, flags=cv2.INPAINT_NS)

        # -----------------------------------------------------------------
        # BƯỚC 2: "NHUỘM ĐEN" TUYỆT ĐỐI CÁC RÌA BIÊN (XÓA CHỮ CỐ ĐỊNH)
        # Cách này giúp cạnh biên sắc nét, không bị kéo dãn hay nhòe màu xám.
        # -----------------------------------------------------------------
        
        # 1. Nhuộm đen dải menu trái (Từ góc trái 0 đến pixel 135)
        result_img[0:h, 0:135] = [0, 0, 0]

        # 2. Nhuộm đen dải thông số đáy (Khoảng 12% kích thước từ dưới lên, che MI, TIS, D2...)
        bottom_thresshold = int(h * 0.88)
        result_img[bottom_thresshold:h, 0:w] = [0, 0, 0]

        # 3. Nhuộm đen thước đo bên phải (Từ pixel w-45 đến hết bên phải)
        result_img[0:h, (w - 45):w] = [0, 0, 0]

        # 4. Nhuộm đen logo máy siêu âm ở đỉnh góc trái (nếu có dính chữ nhỏ)
        result_img[0:40, 0:150] = [0, 0, 0]

        # 5. Lưu ảnh kết quả
        output_path = os.path.join(output_dir, filename)
        cv2.imwrite(output_path, result_img)
        print(" OK!")

    print("\n=== Hoàn thành! Toàn bộ ảnh sạch đã được lưu tại:", output_dir)

if __name__ == "__main__":
    INPUT_FOLDER = "./input"
    OUTPUT_FOLDER = "./output"

    if not os.path.exists(INPUT_FOLDER):
        os.makedirs(INPUT_FOLDER)
        print(f"Đã tạo thư mục '{INPUT_FOLDER}'. Hãy chép các ảnh cần xử lý vào đây rồi chạy lại file.")
    else:
        clean_all_ultrasound_images(INPUT_FOLDER, OUTPUT_FOLDER)