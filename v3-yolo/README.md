# Tool Định Nghĩa Vùng An Toàn & Xóa Chữ Tự Động (Version 3 - v3)

Mô hình ứng dụng trí tuệ nhân tạo tích hợp **YOLOv8-Segmentation** và **EasyOCR** nhằm tự động phát hiện, định nghĩa vùng an toàn (vùng y tế cốt lõi không được can thiệp như cấu trúc `fan` hoặc `wave` trên ảnh siêu âm) và tiến hành xóa sạch các ký tự chữ, thông số kỹ thuật nằm ngoài vùng an toàn mà không làm ảnh hưởng đến dữ liệu gốc.

## 📌 Các Tính Năng Mới Trong Bản v3
* **Định nghĩa vùng an toàn (Safe-zone Masking):** Sử dụng mô hình Custom YOLOv8-seg (`best.pt`) để bóc tách chính xác phân vùng đa giác của các đối tượng đích.
* **Nhận diện chữ nâng cao (EasyOCR GPU):** Tự động phát hiện các chuỗi text, thông số kỹ thuật mảnh, mờ hoặc chữ số rời rạc đặc thù trên ảnh y tế bằng thuật toán OCR tối ưu cấu hình nhạy.
* **Xóa chữ thông minh (Smart Inpainting):** Tự động đối chiếu và bỏ qua (không xóa) các chữ nằm đè hoặc lọt vào vùng an toàn. Chỉ tẩy xóa những text nằm ngoài vùng chỉ định thông qua thuật toán `cv2.inpaint` kết hợp bộ lọc giãn nở vùng biên (Dilate).
* **Tự động hóa luồng cài đặt:** File `setup.bat` thông minh tự nhận diện phần cứng đồ họa NVIDIA (CUDA) để cấu hình thư viện tăng tốc phần cứng tương ứng.

## 📁 Cấu Trúc Thư Mục Dự Án
Sau khi thiết lập thành công, thư mục dự án sẽ có cấu trúc như sau:
```text
├── inputs/                  # Thư mục chứa các ảnh gốc cần xử lý (Input)
├── outputs/                 # Thư mục chứa kết quả đầu ra
│   ├── visual_safe_zone/    # Ảnh trực quan hóa (chứa Bounding Box và Mask của YOLO)
│   └── cleaned_images/      # Ảnh sạch hoàn toàn (đã xóa chữ và giữ nguyên vùng an toàn)
├── models/                  # Nơi lưu trữ file trọng số YOLO (best.pt)
├── venv/                    # Môi trường ảo Python (Virtual Environment)
├── main.py                  # Mã nguồn xử lý chính của Tool
├── setup.bat                # File tự động cài đặt môi trường và thư viện
└── README.md                # Tài liệu hướng dẫn sử dụng này
```
🛠 Hướng Dẫn Cài Đặt (Setup)
Yêu cầu hệ thống
Hệ điều hành: Windows 10 / 11

Python: Phiên bản 3.8 đến 3.11 (Đã được cấu hình Path hệ thống)

Phần cứng (Tùy chọn): GPU NVIDIA hỗ trợ CUDA để chạy tăng tốc xử lý hàng loạt.

Các bước thực hiện:
Tải hoặc clone mã nguồn của project về máy tính.

Click đúp chuột vào file setup.bat để hệ thống tự động khởi tạo môi trường venv, kiểm tra card đồ họa NVIDIA và tự động cài đặt phiên bản PyTorch / EasyOCR phù hợp nhất cho máy của bạn.

Sau khi setup báo SETUP HOÀN TẤT!, hãy di chuyển file trọng số mô hình custom của bạn vào thư mục định sẵn:

Đường dẫn: models/best.pt

🚀 Hướng Dẫn Sử Dụng (Usage)
Sao chép toàn bộ các file ảnh siêu âm cần xử lý (hỗ trợ .jpg, .jpeg, .png, .bmp) vào thư mục inputs/.

Thực hiện chạy tool bằng lệnh dưới đây trong Terminal (hoặc viết vào file run.bat để kích hoạt):

Bash
# Kích hoạt môi trường ảo
call venv\Scripts\activate

# Chạy mã nguồn chính
python main.py
Kiểm tra kết quả trả về tại thư mục outputs/:

Xem file có đuôi _visual trong visual_safe_zone để thẩm định lại vùng an toàn AI quét được.

Lấy file ảnh sạch chữ phục vụ nghiên cứu trong thư mục cleaned_images.

⚙️ Cơ Chế Xử Lý Kỹ Thuật (Pipeline)
Đoạn mã
graph TD
    A[Ảnh Siêu Âm Gốc] --> B[YOLOv8-seg: Trích xuất Safe Mask]
    A --> C[EasyOCR: Phát hiện Text Boxes]
    B --> D[Hàm đối chiếu toán học giao điểm Mask]
    C --> D
    D -->|Text đè lên Safe Mask| E[Giữ lại chữ - Bỏ qua không xóa]
    D -->|Text nằm ngoài Safe Mask| F[Tạo Text Mask + Dilate mở rộng vùng biên]
    F --> G[cv2.inpaint: Tẩy chữ mịn bề mặt]
    G --> H[Output: Thư mục cleaned_images]
Tham số tối ưu hóa OCR: Bản v3 điều chỉnh hạ thấp text_threshold (0.2) và link_threshold (0.3) nhằm ngăn chặn việc bỏ sót các thông số kỹ thuật dạng text cỡ nhỏ, mờ, nét mảnh ở các góc màn hình máy siêu âm.

Xử lý răng cưa biên chữ: Vùng cần xóa được thực hiện cv2.dilate với kernel cấu trúc (7, 7) lặp lại 2 lần giúp dọn sạch hoàn toàn bóng mờ và chân chữ tổn dư sau inpaint.
