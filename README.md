=====VIE Version=====

# Tool Nhận Diện và Xóa Chữ Trong Ảnh Siêu Âm

Tool này sử dụng EasyOCR để trích xuất chữ từ ảnh (như tên phòng khám, thông số máy siêu âm), lưu vào file Excel, và sử dụng thuật toán Inpainting của OpenCV để xóa sạch chữ đó khỏi ảnh một cách tự nhiên.

## Hướng dẫn cài đặt (Setup Environment)

**Bước 1: Cài đặt Python**
Đảm bảo máy tính của bạn đã cài đặt Python (phiên bản 3.8 - 3.11).

**Bước 2: Tạo môi trường ảo (Virtual Environment)**
Mở Terminal / Command Prompt tại thư mục giải nén tool và chạy lệnh sau:
- Windows:
  `python -m venv env`
  `env\Scripts\activate`
- macOS/Linux:
  `python3 -m venv env`
  `source env/bin/activate`

**Bước 3: Cài đặt thư viện**
Chạy lệnh sau để cài đặt các thư viện cần thiết:
`pip install -r requirements.txt`

## Hướng dẫn sử dụng (Run)

1. Chạy tool lần đầu tiên để tạo thư mục cấu trúc:
   `python main.py`
   Tool sẽ tự động tạo một thư mục tên là `input_images`.

2. Copy các hình ảnh siêu âm của bạn vào thư mục `input_images`.

3. Chạy lại tool:
   `python main.py`

4. Kết quả:
   - Các ảnh đã được xóa chữ sẽ nằm trong thư mục `output_images`.
   - Nội dung chữ trích xuất sẽ nằm trong file `KetQua_NhanDien.xlsx`.


=====ENG version=====
```markdown
# Ultrasound Image Text Extractor & Clean-Inpaint Tool

An advanced Python tool engineered specifically for medical ultrasound images. It seamlessly detects and extracts textual information (metadata, scan parameters, clinical data) into an organized Excel sheet, while concurrently removing the text from the images using pixel-level inpainting—without tampering with the sensitive diagnostic areas (such as the uterus or fetus).

---

## 🛠️ Key Features

- **Dual-Pipeline Architecture**: Completely decouples the text extraction logic from the image cleaning mask. Text formatting filters never interfere with the inpainting boundaries.
- **Center Diagnostic Safe-Zone**: Intelligently constructs an inner geometric boundary (25%-75% width, 25%-85% height) over the core ultrasound wave scan. Any text or artifacts detected inside this zone are preserved to protect critical anatomical details.
- **Pixel-Level Target Masking**: Instead of crudely blanking out rectangular bounding boxes, the tool extracts precise bright pixels belonging *only* to the text strokes via binary thresholding, maintaining 100% of the surrounding grain and background textures.
- **Text Normalization for Excel**: Implements automated regular expressions (Regex) to eliminate garbage characters, scan anomalies, and symbolic artifacts, ensuring only clean, readable text strings are logged.
- **Advanced Telea Inpainting**: Employs Fast Marching Method-based inpainting (`cv2.INPAINT_TELEA`) calibrated for low-light, granular medical image environments.

---

## 📐 How the Algorithm Works

1. **Brute-Force High-Sensitivity OCR Scan**: Uses `easyocr` with ultra-low detection thresholds (`text_threshold=0.05`, `low_text=0.05`) to trap every faint, dot-matrix, or anti-aliased character node on the screen.
2. **Dynamic Safe-Zone Screening**: Computes the centroid coordinates for every boundary box. Rims/edges are flagged for deletion, while interior ultrasound wave captures are skipped.
3. **Threshold Mask Extraction**: Isolates high-luminance structures (>110 grayscale value) inside the target boxes to accurately paint a binary text stroke map.
4. **Morphological Dilation**: Slightly expands the mask strokes by 1 pixel to engulf peripheral text halos and anti-aliased borders, preventing any "ghost text" residues.
5. **Excel Structuring & Inpainting Phasing**:
   - The original pixel matrix under the dilated mask is permanently deleted and mathematically reconstructed based on the surrounding dark granular neighborhood values.
   - Concurrently, strings are scrubbed via Regex (`[^\w\s\.,:\-\/]`), filtered by length (>= 2 characters), and saved cleanly into `KetQua_NhanDien.xlsx`.

---

## 🚀 Installation & Environment Setup

### 1. Requirements
Ensure you have Python 3.8 or newer installed on your machine.

### 2. Automatic Setup (Windows)
Double-click `1_setup.bat`. This script will:
- Automatically generate the mandatory folders: `input_images/` and `output_images/`.
- Deploy a localized Python Virtual Environment (`env/`).
- Install dependencies: `easyocr`, `opencv-python`, `pandas`, `openpyxl`, and `numpy`.

---

## 📂 Project Structure

```text
Tool_XoaChuTrongAnh/
├── 1_setup.bat            # Automated workspace & environment initializer
├── 2_run.bat              # Daily one-click execution tool
├── requirements.txt       # Software dependency manifests
├── main.py                # Core Pipeline Python Script
├── input_images/          # [Drop your raw ultrasound images here]
└── output_images/         # [Target destination for textless images]
