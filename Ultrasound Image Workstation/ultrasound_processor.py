import sys
import os
import datetime
import numpy as np
import cv2
from numba import jit
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QLabel, 
                             QPushButton, QSlider, QFileDialog, QVBoxLayout, 
                             QHBoxLayout, QGroupBox, QStatusBar, QComboBox, QCheckBox)
from PyQt6.QtCore import Qt, QEvent
from PyQt6.QtGui import QImage, QPixmap

# ==============================================================================
# THUẬT TOÁN SRAD ĐƯỢC TỐI ƯU HÓA BẰNG NUMBA JIT
# ==============================================================================
@jit(nopython=True, cache=True)
def srad_core(I, n_iter, delta_t, q0):
    rows, cols = I.shape
    for _ in range(n_iter):
        I_next = I.copy()
        for r in range(1, rows - 1):
            for c in range(1, cols - 1):
                dN = I[r-1, c] - I[r, c]
                dS = I[r+1, c] - I[r, c]
                dW = I[r, c-1] - I[r, c]
                dE = I[r, c+1] - I[r, c]
                
                grad_sq = (dN**2 + dS**2 + dW**2 + dE**2) / (I[r, c]**2 + 1e-5)
                laplacian = (dN + dS + dW + dE) / (I[r, c] + 1e-5)
                
                num = 0.5 * grad_sq - (1.0 / 16.0) * (laplacian**2)
                den = (1.0 + 0.25 * laplacian)**2
                q_sq = num / (den + 1e-5)
                if q_sq < 0: q_sq = 0
                q = np.sqrt(q_sq)
                
                xi_num = q**2 - q0**2
                xi_den = q0**2 * (1.0 + q0**2)
                xi = xi_num / (xi_den + 1e-5)
                
                c_c = 1.0 / (1.0 + xi)
                if c_c > 1.0: c_c = 1.0
                if c_c < 0.0: c_c = 0.0
                
                divergence = c_c*dN + c_c*dS + c_c*dW + c_c*dE
                I_next[r, c] = I[r, c] + (delta_t / 4.0) * divergence
        I = I_next
    return I

def apply_srad(img_rgb, n_iter):
    if n_iter == 0:
        return img_rgb
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)
    delta_t = 0.15
    q0 = 1.0 / np.sqrt(n_iter * delta_t + 1.0)
    gray_filtered = srad_core(gray, n_iter, delta_t, q0)
    gray_filtered = np.clip(gray_filtered, 0, 255).astype(np.uint8)
    return cv2.cvtColor(gray_filtered, cv2.COLOR_GRAY2RGB)

# ==============================================================================
# GIAO DIỆN CHÍNH PYQT6 VÀ LUỒNG XỬ LÝ TÍCH HỢP
# ==============================================================================
class UltrasoundProcessorApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Medical Ultrasound Imaging Workstation")
        self.setGeometry(50, 50, 1350, 850)
        
        # Bộ lưu trữ ma trận ảnh và các biến dữ liệu
        self.orig_image = None         
        self.left_view_image = None     
        self.processed_image = None     
        
        # Biến quản lý bộ lọc tô đỏ dấu đo tích hợp
        self.highlight_mask = None      # Mặt nạ chứa các điểm pixel cần nhuộm đỏ
        self.all_detected_boxes = []    # Danh sách dữ liệu bounding boxes phục vụ lưu XML nếu cần
        
        self.init_ui()
        
    def init_ui(self):
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        main_layout = QHBoxLayout(main_widget)
        main_layout.setContentsMargins(10, 10, 10, 10)
        main_layout.setSpacing(10)
        
        # ================= LEFT PANEL: CONTROLS =================
        control_panel = QVBoxLayout()
        control_panel.setSpacing(10)
        
        # Nhóm điều khiển Workstation và các tính năng So sánh
        file_group = QGroupBox("Workstation Control")
        file_layout = QVBoxLayout(file_group)
        file_layout.setSpacing(8)
        
        self.btn_open = QPushButton("📂 Open Ultrasound Image")
        self.btn_open.clicked.connect(self.load_image)
        self.btn_open.setStyleSheet("font-weight: bold; padding: 5px;")
        
        self.btn_save_snapshot = QPushButton("📸 Lock Current to Left View")
        self.btn_save_snapshot.clicked.connect(self.save_to_left_view)
        self.btn_save_snapshot.setEnabled(False)
        self.btn_save_snapshot.setStyleSheet("background-color: #2c3e50; color: white; padding: 5px;")
        
        self.btn_reset_left = QPushButton("🔄 Reset Left to Original")
        self.btn_reset_left.clicked.connect(self.reset_left_view)
        self.btn_reset_left.setEnabled(False)
        
        self.btn_reset_params = QPushButton("🧹 Reset All Sliders")
        self.btn_reset_params.clicked.connect(self.reset_sliders)
        self.btn_reset_params.setEnabled(False)
        
        file_layout.addWidget(self.btn_open)
        file_layout.addWidget(self.btn_save_snapshot)
        file_layout.addWidget(self.btn_reset_left)
        file_layout.addWidget(self.btn_reset_params)
        control_panel.addWidget(file_group)
        
        # Nhóm xuất file ảnh Output
        export_group = QGroupBox("Export Result")
        export_layout = QHBoxLayout(export_group)
        export_layout.setSpacing(8)
        
        self.btn_export_left = QPushButton("💾 Save Left View")
        self.btn_export_left.clicked.connect(lambda: self.export_image_to_disk("left"))
        self.btn_export_left.setEnabled(False)
        self.btn_export_left.setStyleSheet("background-color: #27ae60; color: white; font-weight: bold; padding: 5px;")
        
        self.btn_export_right = QPushButton("💾 Save Right View")
        self.btn_export_right.clicked.connect(lambda: self.export_image_to_disk("right"))
        self.btn_export_right.setEnabled(False)
        self.btn_export_right.setStyleSheet("background-color: #2980b9; color: white; font-weight: bold; padding: 5px;")
        
        export_layout.addWidget(self.btn_export_left)
        export_layout.addWidget(self.btn_export_right)
        control_panel.addWidget(export_group)
        
        # Nhóm thông số thuật toán (Pipeline xử lý ảnh nâng cao)
        param_group = QGroupBox("Advanced Pipeline")
        param_layout = QVBoxLayout(param_group)
        param_layout.setSpacing(8)
        
        # --- THÀNH PHẦN TÍCH HỢP MỚI: CHECKBOX HIGHLIGHT ---
        self.chk_highlight = QCheckBox("🎯 Highlight (Tô đỏ dấu đo y tế)")
        self.chk_highlight.setStyleSheet("font-weight: bold; color: #e74c3c; margin-bottom: 5px;")
        self.chk_highlight.stateChanged.connect(self.process_and_display)
        param_layout.addWidget(self.chk_highlight)
        
        # --- BƯỚC 1: DROP-DOWN KHỬ NHIỄU ---
        param_layout.addWidget(QLabel("<b>1. Denoise Method (Khử nhiễu):</b>"))
        self.combo_filter = QComboBox()
        self.combo_filter.addItems([
            "No Filter (Bypass)",
            "Gaussian Filter (Làm mịn)",
            "Median Filter (Trung vị)",
            "Bilateral Filter (Song phương)",
            "SRAD (Bất đẳng hướng siêu âm)"
        ])
        self.combo_filter.currentIndexChanged.connect(self.on_filter_changed)
        param_layout.addWidget(self.combo_filter)
        
        self.lbl_noise_title = QLabel("Filter Strength:")
        param_layout.addWidget(self.lbl_noise_title)
        self.slider_noise = QSlider(Qt.Orientation.Horizontal)
        self.slider_noise.setRange(0, 0)
        self.slider_noise.valueChanged.connect(self.process_and_display)
        self.lbl_noise_value = QLabel("Off")
        self.lbl_noise_value.setAlignment(Qt.AlignmentFlag.AlignRight)
        param_layout.addWidget(self.slider_noise)
        param_layout.addWidget(self.lbl_noise_value)
        
        # --- BƯỚC 2: ĐỘ SÁNG ---
        param_layout.addWidget(QLabel("<b>2. Brightness (Độ sáng):</b>"))
        self.slider_brightness = QSlider(Qt.Orientation.Horizontal)
        self.slider_brightness.setRange(-100, 100)
        self.slider_brightness.setValue(0)
        self.slider_brightness.valueChanged.connect(self.process_and_display)
        self.lbl_brightness = QLabel("0")
        self.lbl_brightness.setAlignment(Qt.AlignmentFlag.AlignRight)
        param_layout.addWidget(self.slider_brightness)
        param_layout.addWidget(self.lbl_brightness)
        
        # --- BƯỚC 3: DROP-DOWN PHƯƠNG PHÁP TƯƠNG PHẢN ---
        param_layout.addWidget(QLabel("<b>3. Contrast Method (Tương phản):</b>"))
        self.combo_contrast_method = QComboBox()
        self.combo_contrast_method.addItems([
            "Linear Contrast (Tuyến tính)",
            "CLAHE (Thích ứng y tế cao cấp)"
        ])
        self.combo_contrast_method.currentIndexChanged.connect(self.on_contrast_method_changed)
        param_layout.addWidget(self.combo_contrast_method)
        
        self.slider_contrast = QSlider(Qt.Orientation.Horizontal)
        self.slider_contrast.setRange(50, 300) 
        self.slider_contrast.setValue(100)
        self.slider_contrast.valueChanged.connect(self.process_and_display)
        self.lbl_contrast = QLabel("1.0")
        self.lbl_contrast.setAlignment(Qt.AlignmentFlag.AlignRight)
        param_layout.addWidget(self.slider_contrast)
        param_layout.addWidget(self.lbl_contrast)
        
        # --- BƯỚC 4: DROP-DOWN SẮC NÉT ---
        param_layout.addWidget(QLabel("<b>4. Sharpness Method (Sắc nét):</b>"))
        self.combo_sharp_method = QComboBox()
        self.combo_sharp_method.addItems([
            "Unsharp Masking (Tần số biên)",
            "Morphological Sharpen (Hình thái học)"
        ])
        self.combo_sharp_method.currentIndexChanged.connect(self.on_sharp_method_changed)
        param_layout.addWidget(self.combo_sharp_method)
        
        self.slider_sharpness = QSlider(Qt.Orientation.Horizontal)
        self.slider_sharpness.setRange(0, 10)
        self.slider_sharpness.setValue(0)
        self.slider_sharpness.valueChanged.connect(self.process_and_display)
        self.lbl_sharpness = QLabel("0")
        self.lbl_sharpness.setAlignment(Qt.AlignmentFlag.AlignRight)
        param_layout.addWidget(self.slider_sharpness)
        param_layout.addWidget(self.lbl_sharpness)
        
        control_panel.addWidget(param_group)
        control_panel.addStretch()
        
        control_container = QWidget()
        control_container.setLayout(control_panel)
        control_container.setFixedWidth(290)
        main_layout.addWidget(control_container)
        
        # ================= RIGHT PANEL: VISUALIZATION =================
        view_layout = QHBoxLayout()
        view_layout.setSpacing(8)
        
        self.lbl_orig_view = QLabel("No image loaded")
        self.lbl_orig_view.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.lbl_orig_view.setStyleSheet("border: 1px solid #444; background-color: #111; color: #777;")
        
        self.lbl_proc_view = QLabel("No image loaded")
        self.lbl_proc_view.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.lbl_proc_view.setStyleSheet("border: 1px solid #444; background-color: #111; color: #777;")
        
        view_layout.addWidget(self.lbl_orig_view, stretch=1)
        view_layout.addWidget(self.lbl_proc_view, stretch=1)
        
        main_layout.addLayout(view_layout, stretch=1)
        
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.status_bar.showMessage("Ready. Please open an ultrasound image.")
        
        self.toggle_controls(False)
        main_widget.installEventFilter(self)

    # ==============================================================================
    # THUẬT TOÁN TÌM KIẾM VÀ TRÍCH XUẤT MẶT NẠ DẤU ĐO (ALGORITHM INTEGRATION)
    # ==============================================================================
    def extract_highlight_mask_from_original(self, img_rgb, templates_dir, threshold=0.60):
        """
        Quét đa mẫu dựa trên thuật toán gốc của bạn trên ảnh sạch ban đầu.
        SỬA ĐỔI: Chuyển đổi đầu ra từ nhuộm trực tiếp sang tạo ma trận mặt nạ nhị phân (Mask).
        """
        h_shape, w_shape = img_rgb.shape[:2]
        # Khởi tạo mặt nạ nhị phân rỗng (đen hoàn toàn) cùng kích thước ảnh gốc
        mask = np.zeros((h_shape, w_shape), dtype=np.uint8)
        boxes = []
        
        if not os.path.exists(templates_dir):
            return mask, boxes

        valid_extensions = (".jpg", ".jpeg", ".png", ".bmp")
        template_files = [f for f in os.listdir(templates_dir) if f.lower().endswith(valid_extensions)]

        if not template_files:
            return mask, boxes

        # Chuyển đổi sang ảnh xám BGR2GRAY (Vì ảnh OpenCV đọc vào là hệ BGR/RGB)
        gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
        gray_filtered = cv2.bilateralFilter(gray, d=9, sigmaColor=75, sigmaSpace=75)

        for tpl_name in template_files:
            if "plus" in tpl_name.lower():
                class_name = "plus"
            elif "x" in tpl_name.lower():
                class_name = "x_mark"
            else:
                class_name = "caliper"

            tpl_path = os.path.join(templates_dir, tpl_name)
            tpl = cv2.imread(tpl_path, cv2.IMREAD_GRAYSCALE)
            if tpl is None: continue
            h, w = tpl.shape[:2]
            
            _, tpl_mask = cv2.threshold(tpl, 220, 255, cv2.THRESH_BINARY)
            res = cv2.matchTemplate(gray_filtered, tpl, cv2.TM_CCOEFF_NORMED)
            loc = np.where(res >= threshold)
            
            for pt in zip(*loc[::-1]):
                x_start, y_start = pt[0], pt[1]
                
                if x_start < 80 or y_start > (h_shape - 118): # Tự động tương thích độ phân giải
                    continue
                    
                if y_start + h > h_shape or x_start + w > w_shape:
                    continue

                xmin = max(0, x_start - 2)
                ymin = max(0, y_start - 2)
                xmax = min(w_shape, x_start + w + 2)
                ymax = min(h_shape, y_start + h + 2)
                
                is_duplicate = False
                for b in boxes:
                    if abs(b['xmin'] - xmin) < 8 and abs(b['ymin'] - ymin) < 8:
                        is_duplicate = True
                        break
                
                if is_duplicate:
                    continue

                gray_roi = gray[y_start:y_start+h, x_start:x_start+w]
                _, dynamic_mask = cv2.threshold(gray_roi, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                
                kernel = np.ones((3, 3), np.uint8)
                tpl_mask_dilated = cv2.dilate(tpl_mask, kernel, iterations=1)
                final_roi_mask = cv2.bitwise_and(tpl_mask_dilated, dynamic_mask)
                
                # Ghi nhận các pixel tìm được vào Mặt nạ nhị phân tổng thể
                mask[y_start:y_start+h, x_start:x_start+w] = cv2.bitwise_or(
                    mask[y_start:y_start+h, x_start:x_start+w], final_roi_mask
                )
                
                boxes.append({
                    'name': class_name,
                    'xmin': xmin,
                    'ymin': ymin,
                    'xmax': xmax,
                    'ymax': ymax
                })
                
        return mask, boxes

    # ==============================================================================
    # QUẢN LÝ GIAO DIỆN VÀ PIPELINE XỬ LÝ ĐỘNG
    # ==============================================================================
    def on_filter_changed(self, index):
        self.slider_noise.blockSignals(True)
        self.slider_noise.setValue(0) 
        if index == 0:
            self.lbl_noise_title.setText("Filter Strength:")
            self.slider_noise.setRange(0, 0)
        elif index == 1:
            self.lbl_noise_title.setText("Kernel Size (Mẫu lẻ):")
            self.slider_noise.setRange(0, 10)
        elif index == 2:
            self.lbl_noise_title.setText("Kernel Size (Mẫu lẻ):")
            self.slider_noise.setRange(0, 7)
        elif index == 3:
            self.lbl_noise_title.setText("Sigma Color/Space (Mịn biên):")
            self.slider_noise.setRange(0, 20)
        elif index == 4:
            self.lbl_noise_title.setText("SRAD Iterations (Số vòng lặp):")
            self.slider_noise.setRange(0, 40)
        self.slider_noise.blockSignals(False)
        self.process_and_display()

    def on_contrast_method_changed(self, index):
        self.slider_contrast.blockSignals(True)
        if index == 0: 
            self.slider_contrast.setRange(50, 300)
            self.slider_contrast.setValue(100) 
        else: 
            self.slider_contrast.setRange(0, 10)   
            self.slider_contrast.setValue(0)       
        self.slider_contrast.blockSignals(False)
        self.process_and_display()

    def on_sharp_method_changed(self, index):
        self.process_and_display()

    def toggle_controls(self, enabled):
        self.chk_highlight.setEnabled(enabled)
        self.combo_filter.setEnabled(enabled)
        self.slider_noise.setEnabled(enabled)
        self.slider_brightness.setEnabled(enabled)
        self.combo_contrast_method.setEnabled(enabled)
        self.slider_contrast.setEnabled(enabled)
        self.combo_sharp_method.setEnabled(enabled)
        self.slider_sharpness.setEnabled(enabled)
        self.btn_reset_params.setEnabled(enabled)
        self.btn_save_snapshot.setEnabled(enabled)
        self.btn_reset_left.setEnabled(enabled)
        self.btn_export_left.setEnabled(enabled)
        self.btn_export_right.setEnabled(enabled)

    def save_to_left_view(self):
        if self.processed_image is not None:
            self.left_view_image = self.processed_image.copy()
            self.display_on_label(self.left_view_image, self.lbl_orig_view)
            self.status_bar.showMessage("Snapshot locked to Left View. You can now modify parameters to compare.")

    def reset_left_view(self):
        if self.orig_image is not None:
            # Nếu nút Highlight đang bật, đồng bộ hiển thị ảnh gốc kèm hiệu ứng đỏ lên View Trái
            if self.chk_highlight.isChecked() and self.highlight_mask is not None:
                tmp_left = self.orig_image.copy()
                tmp_left[self.highlight_mask > 0] = [255, 0, 0] # Hệ màu RGB nhuộm đỏ
                self.left_view_image = tmp_left
            else:
                self.left_view_image = self.orig_image.copy()
            self.display_on_label(self.left_view_image, self.lbl_orig_view)
            self.status_bar.showMessage("Left View reset to Base Image.")

    def load_image(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Open Image File", "", "Images (*.png *.jpg *.jpeg *.bmp *.tiff)"
        )
        if file_path:
            img = cv2.imread(file_path)
            if img is None:
                self.status_bar.showMessage("Error: Could not decode image file.")
                return
            
            self.orig_image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            # --- LUỒNG TÍCH HỢP: TÍNH TOÁN TRƯỚC MẶT NẠ TÔ ĐỎ NGAY KHI TẢI ẢNH ---
            self.status_bar.showMessage("Scanning templates and building highlight mask...")
            QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
            
            self.highlight_mask, self.all_detected_boxes = self.extract_highlight_mask_from_original(
                self.orig_image, "templates", threshold=0.62
            )
            
            QApplication.restoreOverrideCursor()
            # -----------------------------------------------------------------
            
            self.toggle_controls(True)
            self.chk_highlight.setChecked(False) # Mặc định ban đầu chưa bật tô đỏ
            self.reset_sliders()
            self.reset_left_view()
            self.status_bar.showMessage(f"Loaded: {os.path.basename(file_path)} | Found {len(self.all_detected_boxes)} targets.")

    def process_and_display(self):
        if self.orig_image is None:
            return
            
        filter_idx = self.combo_filter.currentIndex()
        n_val = self.slider_noise.value()
        b_val = self.slider_brightness.value()
        contrast_idx = self.combo_contrast_method.currentIndex()
        c_val = self.slider_contrast.value()
        sharp_idx = self.combo_sharp_method.currentIndex()
        s_val = self.slider_sharpness.value()
        
        # Cập nhật thông số nhãn chữ
        if filter_idx == 0 or n_val == 0: self.lbl_noise_value.setText("Off")
        elif filter_idx == 1: self.lbl_noise_value.setText(f"Kernel: {n_val*2+1}x{n_val*2+1}")
        elif filter_idx == 2: self.lbl_noise_value.setText(f"Kernel Size: {n_val*2+1}")
        elif filter_idx == 3: self.lbl_noise_value.setText(f"Sigma: {n_val*4}")
        elif filter_idx == 4: self.lbl_noise_value.setText(f"{n_val} Iters")
            
        self.lbl_brightness.setText(str(b_val))
        self.lbl_sharpness.setText(str(s_val))
        
        if contrast_idx == 0:
            self.lbl_contrast.setText(f"Alpha: {c_val / 100.0:.2f}")
        else:
            if c_val == 0: self.lbl_contrast.setText("Off (Bypass)")
            else: self.lbl_contrast.setText(f"Clip Limit: {c_val}.0")
        
        # Chạy bản sao xử lý ảnh qua Pipeline chỉnh sửa thông số
        img = self.orig_image.copy()
        
        # --- PIPELINE BƯỚC 1: KHỬ NHIỄU ---
        if n_val > 0:
            if filter_idx == 1:
                k = n_val * 2 + 1
                img = cv2.GaussianBlur(img, (k, k), 0)
            elif filter_idx == 2:
                img = cv2.medianBlur(img, n_val * 2 + 1)
            elif filter_idx == 3:
                img = cv2.bilateralFilter(img, d=9, sigmaColor=n_val*4, sigmaSpace=n_val*4)
            elif filter_idx == 4:
                img = apply_srad(self.orig_image, n_iter=n_val)
                
        # --- PIPELINE BƯỚC 2: ĐỘ SÁNG ---
        if b_val != 0:
            img = np.clip(img.astype(np.float32) + b_val, 0, 255).astype(np.uint8)
            
        # --- PIPELINE BƯỚC 3: ĐỘ TƯƠNG PHẢN ---
        if contrast_idx == 0: 
            alpha = c_val / 100.0
            img = np.clip(img.astype(np.float32) * alpha, 0, 255).astype(np.uint8)
        else: 
            if c_val > 0: 
                gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
                clahe = cv2.createCLAHE(clipLimit=float(c_val), tileGridSize=(8, 8))
                gray_clahe = clahe.apply(gray)
                img = cv2.cvtColor(gray_clahe, cv2.COLOR_GRAY2RGB)
            
        # --- PIPELINE BƯỚC 4: ĐỘ SẮC NÉT ---
        if s_val > 0:
            if sharp_idx == 0: 
                blurred = cv2.GaussianBlur(img, (5, 5), 0)
                weight = s_val * 0.3
                sharpened = cv2.addWeighted(img, 1.0 + weight, blurred, -weight, 0)
                img = np.clip(sharpened, 0, 255).astype(np.uint8)
            else: 
                kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
                tophat = cv2.morphologyEx(img, cv2.MORPH_TOPHAT, kernel)
                blackhat = cv2.morphologyEx(img, cv2.MORPH_BLACKHAT, kernel)
                img = cv2.add(img, tophat)
                img = cv2.subtract(img, blackhat)
                
        # --- LUỒNG TÍCH HỢP MỚI: CHỒNG LỚP MẶT NẠ TÔ ĐỎ (HIGHLIGHT OVERLAY) ---
        # Áp dụng cuối Pipeline giúp màu đỏ nguyên bản tuyệt đối, không bị biến đổi sắc độ
        if self.chk_highlight.isChecked() and self.highlight_mask is not None:
            img[self.highlight_mask > 0] = [255, 0, 0] # Hệ màu RGB trong PyQt nhuộm màu Red
            
        self.processed_image = img
        
        self.display_on_label(self.left_view_image, self.lbl_orig_view)
        self.display_on_label(self.processed_image, self.lbl_proc_view)

    def export_image_to_disk(self, target_view):
        if target_view == "left":
            img_to_save = self.left_view_image
            view_label = "Left_View"
        else:
            img_to_save = self.processed_image
            view_label = "Right_View"
            
        if img_to_save is None:
            self.status_bar.showMessage("Error: No image content to save.")
            return
            
        output_dir = "./output"
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"ultrasound_{view_label}_{timestamp}.png"
        full_path = os.path.join(output_dir, filename)
        
        # Chuyển đổi hệ màu RGB về lại BGR trước khi ghi file bằng OpenCV
        bgr_img = cv2.cvtColor(img_to_save, cv2.COLOR_RGB2BGR)
        success = cv2.imwrite(full_path, bgr_img)
        
        if success:
            self.status_bar.showMessage(f"Saved successfully to: {full_path}")
        else:
            self.status_bar.showMessage("Error: Failed to write image to disk.")

    def reset_sliders(self):
        self.combo_filter.blockSignals(True)
        self.slider_noise.blockSignals(True)
        self.slider_brightness.blockSignals(True)
        self.combo_contrast_method.blockSignals(True)
        self.slider_contrast.blockSignals(True)
        self.combo_sharp_method.blockSignals(True)
        self.slider_sharpness.blockSignals(True)
        
        self.combo_filter.setCurrentIndex(0)
        self.slider_noise.setRange(0, 0)
        self.slider_noise.setValue(0)
        self.slider_brightness.setValue(0)
        self.combo_contrast_method.setCurrentIndex(0)
        self.slider_contrast.setRange(50, 300)
        self.slider_contrast.setValue(100)
        self.combo_sharp_method.setCurrentIndex(0)
        self.slider_sharpness.setValue(0)
        self.lbl_noise_title.setText("Filter Strength:")
        
        self.combo_filter.blockSignals(False)
        self.slider_noise.blockSignals(False)
        self.slider_brightness.blockSignals(False)
        self.combo_contrast_method.blockSignals(False)
        self.slider_contrast.blockSignals(False)
        self.combo_sharp_method.blockSignals(False)
        self.slider_sharpness.blockSignals(False)
        
        self.process_and_display()

    def eventFilter(self, source, event):
        if event.type() == QEvent.Type.Resize and self.orig_image is not None:
            self.display_on_label(self.left_view_image, self.lbl_orig_view)
            self.display_on_label(self.processed_image, self.lbl_proc_view)
        return super().eventFilter(source, event)

    def display_on_label(self, rgb_array, label_element):
        if rgb_array is None: return
        height, width, channel = rgb_array.shape
        bytes_per_line = channel * width
        q_img = QImage(rgb_array.data, width, height, bytes_per_line, QImage.Format.Format_RGB888)
        pixmap = QPixmap.fromImage(q_img)
        scaled_pixmap = pixmap.scaled(
            label_element.width() - 4, label_element.height() - 4, 
            Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation
        )
        label_element.setPixmap(scaled_pixmap)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = UltrasoundProcessorApp()
    window.show()
    sys.exit(app.exec())