# 🛠️ Ultrasound Image Text Redaction & ROI Masking

Hệ thống tự động nhận diện chữ (OCR), xóa chữ vùng ngoài bằng thuật toán Inpainting, và bảo vệ các vùng nhận diện trọng yếu (Vùng quạt siêu âm - **Fan ROI** và Vùng phổ Doppler - **Doppler ROI**) trên ảnh siêu âm. Kết quả văn bản trích xuất được tổng hợp và xuất ra file Excel.

---

## 📌 Tính năng chính

* **Nhận diện văn bản (OCR):** Sử dụng `RapidOCR` mang lại tốc độ xử lý cao và độ chính xác tối ưu.
* **Bảo vệ Vùng cấm (ROI Protection):** Định vị tọa độ tỉ lệ để khóa vùng quạt siêu âm và vùng Doppler, đảm bảo không can thiệp xóa dữ liệu quan trọng bên trong các vùng này.
* **Xóa chữ thông minh:** Tự động xóa (Inpaint) các văn bản, thông số bệnh nhân nằm ngoài vùng cấm bằng thuật toán `Telea` từ OpenCV.
* **Tổng hợp dữ liệu:** Xuất báo cáo Excel gom nhóm các chuỗi text phát hiện được trên từng ảnh, kèm mức độ tin cậy (confidence score) và trạng thái chạm vùng bảo vệ.
* **Hệ thống xác thực:** Tự động kiểm tra tính toàn vẹn của dữ liệu đầu ra sau khi xử lý xong batch ảnh.

---

## 📂 Cấu trúc thư mục tương tác

Sau khi chạy mã nguồn, cấu trúc thư mục sẽ tự động được khởi tạo như sau:

```text
├── input/                  # Chứa các ảnh siêu âm gốc cần xử lý (*.jpg, *.png,...)
├── output/                 # Thư mục chứa kết quả đầu ra
│   ├── vung_cam/           # Ảnh xem trước (Preview) chứa khung đa giác vùng cấm
│   └── da_xu_ly/           # Ảnh đã được xóa chữ sạch và file Excel tổng hợp
│       └── ocr_texts.xlsx  # File Excel báo cáo kết quả OCR
├── main.py                 # File mã nguồn chính
└── README.md               # File hướng dẫn này
