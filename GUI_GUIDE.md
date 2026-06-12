# 🏥 MedSAM Annotation Tool — Hướng dẫn cài đặt & sử dụng

> **Đây là phần mở rộng** của repo gốc [MedSAM (bowang-lab)](https://github.com/bowang-lab/MedSAM).  
> Nhóm đã tự xây dựng thêm tính năng **gán nhãn (annotation)** vào `gui.py` để phục vụ đồ án.

---

## ✨ Tính năng bổ sung so với bản gốc

| Tính năng | Gốc MedSAM | Bản nhóm |
|---|:---:|:---:|
| Load ảnh + segment bằng SAM | ✅ | ✅ |
| Gán nhãn cho từng vùng segment | ❌ | ✅ |
| Nhãn y khoa preset (uterus, myoma, polyp...) | ❌ | ✅ |
| Nhãn tùy chỉnh | ❌ | ✅ |
| Thông tin bệnh nhân (Patient ID, tư thế tử cung...) | ❌ | ✅ |
| Danh sách annotation + Undo / Xóa | ❌ | ✅ |
| Export JSON chuẩn COCO | ❌ | ✅ |
| Lưu ảnh annotated kèm nhãn | ❌ | ✅ |

---

## 🛠️ Cài đặt

### Bước 1 — Clone repo gốc MedSAM

```bash
git clone https://github.com/bowang-lab/MedSAM.git
cd MedSAM
```

### Bước 2 — Tạo môi trường ảo & cài dependencies

```bash
# Tạo môi trường conda (Python 3.10)
conda create -n medsam python=3.10 -y
conda activate medsam

# Cài PyTorch — chọn đúng phiên bản cho máy:
# → Có GPU NVIDIA:
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
# → Chỉ có CPU:
pip install torch torchvision

# Cài các thư viện còn lại
pip install -e .
pip install PySide6 scikit-image pillow
```

### Bước 3 — Tải model đã train của nhóm

Tải file **`medsam_model_best.pth`** từ Google Drive:

> 🔗 **[Link Google Drive — medsam_model_best.pth](https://drive.google.com/your-link-here)**  
> *(Liên hệ thành viên nhóm để lấy link nếu chưa có)*

Sau khi tải về, **đặt file vào thư mục gốc** của repo:

```
MedSAM/
├── gui.py                    ← File GUI của nhóm (thay thế file gốc)
├── medsam_model_best.pth     ← ✅ Đặt file model tại đây
├── segment_anything/
└── ...
```

### Bước 4 — Thay thế `gui.py` bằng bản của nhóm

Tải file `gui.py` của nhóm và **ghi đè** lên file `gui.py` gốc trong thư mục `MedSAM/`.

---

## 🚀 Chạy ứng dụng

```bash
conda activate medsam
python gui.py
```

---

## 🖼️ Cách sử dụng

1. **Load ảnh** → Nhấn `📂 Load ảnh`, chọn file ảnh siêu âm (`.png`, `.jpg`, `.bmp`, `.tif`)
2. **Vẽ bounding box** → Kéo chuột trên ảnh để khoanh vùng cần segment
3. **MedSAM tự segment** → Mask sẽ được tô màu tự động
4. **Gán nhãn** → Chọn nhãn từ danh sách preset hoặc nhập nhãn mới
5. **Export** → Xuất file JSON (chuẩn COCO) hoặc lưu ảnh có annotation

---

## 🗂️ Lấy dataset để thử nghiệm

Nếu chưa có ảnh y khoa, bạn có thể dùng một trong các nguồn **mở, miễn phí** dưới đây:

---

### 📦 Option 1 — Ảnh demo có sẵn trong repo *(Nhanh nhất — thử ngay không cần tải thêm)*

Repo MedSAM gốc đã kèm sẵn ảnh demo trong thư mục `assets/`:

```
MedSAM/assets/img_demo.png   ← Load thẳng vào GUI là dùng được
```

Hoặc chạy inference mẫu để kiểm tra model trước:

```bash
python MedSAM_Inference.py
```
---

### 📦 Option 2 — Ultrasound Nerve Segmentation (Kaggle)

Ảnh siêu âm thần kinh cổ, định dạng PNG, dùng được ngay với GUI không cần tiền xử lý.

> 🔗 [kaggle.com/c/ultrasound-nerve-segmentation](https://www.kaggle.com/competitions/ultrasound-nerve-segmentation/data)

---

### 📦 Option 3 — FLARE22 Dataset — CT bụng *(Dataset nhóm dùng để train)*

Dataset gốc nhóm dùng để train model. Gồm 50 ảnh CT bụng với 13 nhãn organ.  
Cần tiền xử lý trước khi dùng (chạy `pre_CT_MR.py`).

> 🔗 [Tải tại Zenodo (~4 GB)](https://zenodo.org/record/7860267)

```bash
# Giải nén vào:
data/FLARE22Train/

# Chạy tiền xử lý:
pip install connected-components-3d
python pre_CT_MR.py
```

---

## ⚠️ Lưu ý

- File `medsam_model_best.pth` **phải đặt đúng thư mục gốc** (cùng cấp với `gui.py`)
- GUI hỗ trợ ảnh: `.png`, `.jpg`, `.jpeg`, `.bmp`, `.tif`, `.tiff`
- Ảnh grayscale sẽ được tự động chuyển sang RGB
- Nếu máy không có GPU, model vẫn chạy được trên CPU (chậm hơn ~5–10×)
- Kết quả JSON export theo chuẩn **COCO-like**, có thể dùng cho training tiếp theo

---

## 📌 Cấu trúc file JSON output

```json
{
  "info": {
    "tool": "MedSAM Annotation Tool",
    "date": "2026-06-12",
    "model_checkpoint": "medsam_model_best.pth"
  },
  "image": {
    "file_name": "sample.png",
    "width": 512,
    "height": 512,
    "patient_id": "PT001",
    "uterine_position": "anteverted",
    "notes": "ghi chú tùy ý"
  },
  "categories": [
    { "id": 1, "name": "myoma", "group": "pathology" }
  ],
  "annotations": [
    {
      "id": 1,
      "category_id": 1,
      "label": "myoma",
      "bbox": [120, 80, 300, 250],
      "segmentation": [[121.0, 81.5, 122.0, 83.0, "..."]],
      "area": 24500
    }
  ]
}
```

---

## Công việc tiếp theo

- Từ gui.py có sẵn, hãy build thành một cái app (execute file .exe) để có thể khởi chạy luôn mà không cần prompt cmd
- Chia ra và Export thành 3 folder: Input data, Labelled data và JSON
- Tệp json yêu cầu phải đặt tên giống như folder input data (ex. Abc.png -> Abc.json), có toạ độ gán nhãn được đánh dấu đỏ trong các ảnh được làm sạch và toạ độ khoah vùng Boundbox kèm theo nội dung gán nhãn (tổn thương mô tử cung, u xơ, u ác tính,...)