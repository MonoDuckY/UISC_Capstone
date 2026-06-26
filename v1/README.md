# 📸 Ultrasound Clinical Data Extractor

**Ultrasound Clinical Data Extractor** là một công cụ tự động hóa việc trích xuất thông tin lâm sàng từ hình ảnh siêu âm sản khoa. Công cụ sử dụng thị giác máy tính (OpenCV) và nhận dạng ký tự quang học (Tesseract OCR) để chuyển đổi các chỉ số trên ảnh thành dữ liệu có cấu trúc (Excel/JSON).

---

## ✨ Tính năng nổi bật

*   **Trích xuất thông tin y tế chính:** Nhịp tim thai (FHR), Chiều dài đầu mông (CRL), Tuổi thai (GA), và các phép đo khoảng cách (D1, D2, BPD, FL...).
*   **Giám sát an toàn:** Tự động lấy các chỉ số an toàn sinh học (MI, TIS, TIB).
*   **Hệ thống Logic thông minh:**
    *   Tự động sửa lỗi OCR phổ biến (Ví dụ: nhầm số `5` thành chữ `S`).
    *   Kiểm soát tính hợp lý của dữ liệu (Sửa ngày tháng vô lý như `34/12`, sửa nhịp tim bị dính số).
    *   Tự động thêm dấu chấm thập phân cho các chỉ số máy (Ví dụ: `147` -> `1.47`).
*   **Lọc nhiễu:** Loại bỏ hoàn toàn thông tin kỹ thuật máy và tên phòng khám để tập trung vào dữ liệu bệnh nhân.
*   **Xuất dữ liệu đa dạng:** Hỗ trợ file `.csv` (mở bằng Excel) và `.json`.

---

## 🛠 Yêu cầu hệ thống

1.  **Python:** Phiên bản 3.6 trở lên.
2.  **Tesseract OCR Engine:** 
    *   Tải tại: [UB-Mannheim Tesseract OCR](https://github.com/UB-Mannheim/tesseract/wiki)
    *   Cài đặt vào đường dẫn mặc định: `C:\Program Files\Tesseract-OCR\tesseract.exe`

---

## 🚀 Hướng dẫn cài đặt và Sử dụng

### Bước 1: Cài đặt môi trường
Chạy file `setup.bat`. File này sẽ tự động:
*   Tạo môi trường ảo Python (`venv`).
*   Cài đặt các thư viện cần thiết (`opencv-python`, `pytesseract`, `numpy`).
*   Tạo cấu trúc thư mục `input/` và `output/`.

### Bước 2: Chuẩn bị dữ liệu
Copy toàn bộ ảnh siêu âm (định dạng `.jpg`, `.png`, `.jpeg`) vào thư mục `input/`.

### Bước 3: Chạy công cụ
Chạy file `run.bat`. Chương trình sẽ quét toàn bộ ảnh và xử lý dữ liệu.

### Bước 4: Nhận kết quả
Vào thư mục `output/` để lấy file:
*   `clinical_report.csv`: Báo cáo dạng bảng, mở bằng Excel cực kỳ tiện lợi.
*   `clinical_data.json`: Dữ liệu cấu trúc dành cho lập trình viên.

---

## 📊 Các thông số trích xuất

| Thông số | Mô tả |
| :--- | :--- |
| **Timestamp** | Ngày và giờ thực hiện siêu âm (đã qua kiểm tra logic lịch). |
| **FHR (bpm)** | Nhịp tim thai (Nhịp/phút). |
| **Gestational Age** | Tuổi thai (Ví dụ: 6w3d - 6 tuần 3 ngày). |
| **Measurements** | Chiều dài đầu mông (CRL), Đường kính (D1, D2...), v.v. |
| **Safety Indices** | Chỉ số cơ học (MI) và Chỉ số nhiệt (TIS, TIB). |

---

## 📝 Lưu ý quan trọng
*   Công cụ hoạt động tốt nhất với ảnh siêu âm có độ tương phản tốt, chữ rõ ràng.
*   Đường dẫn Tesseract OCR trong file `main.py` cần khớp với thư mục cài đặt thực tế trên máy tính của bạn.

## ⚖️ Miễn trừ trách nhiệm
*Dữ liệu trích xuất từ công cụ này chỉ mang tính chất tham khảo và hỗ trợ thống kê. Mọi kết luận lâm sàng cuối cùng phải dựa trên tờ kết quả có dấu xác nhận của bác sĩ chuyên khoa.*
