import sys
import os
import datetime
import numpy as np
import cv2
from numba import jit
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QLabel, 
                             QPushButton, QSlider, QFileDialog, QVBoxLayout, 
                             QHBoxLayout, QGroupBox, QStatusBar, QComboBox, QCheckBox,
                             QScrollArea)
from PyQt6.QtCore import Qt, QEvent, QRect, QPoint
from PyQt6.QtGui import QImage, QPixmap, QPainter, QPen, QColor

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
# HỆ THỐNG THUẬT TOÁN ĐÁNH GIÁ TIÊU CHÍ CHẤT LƯỢNG THEO NHÓM NGHIÊN CỨU
# ==============================================================================
def analyze_medical_criteria(img_current, img_reference=None):
    """
    Tính toán chi tiết các tiêu chí y khoa dựa trên các công thức khoa học:
    Trường nhìn (R), Độ sáng (SNR), Độ tương phản (CNR), Độ sắc nét (ENL, PSNR), Nhiễu hạt (VoL, Tenengrad)
    """
    if img_current is None:
        return None
        
    gray = cv2.cvtColor(img_current, cv2.COLOR_RGB2GRAY)
    total_pixels = gray.size
    h, w = gray.shape
    
    # 1. TIÊU CHÍ TRƯỜNG NHÌN (Field of View - R)
    _, thresh_ut = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    ut_pixels = np.sum(thresh_ut == 255)
    r_fov = ut_pixels / total_pixels
    
    if 0.5 <= r_fov <= 0.85:
        fov_status = "PASS"
        fov_color = "#2ecc71"
    elif r_fov < 0.5:
        fov_status = "CROP IMAGE"
        fov_color = "#f1c40f"
    else:
        fov_status = "REJECT"
        fov_color = "#e74c3c"
        
    # Lọc tay: Kiểm tra cấu trúc ảnh ghép đôi (Dual-frame layout) qua đối xứng biên dọc
    mid_col = w // 2
    left_side = gray[:, :mid_col-10]
    right_side = gray[:, mid_col+10:]
    min_w = min(left_side.shape[1], right_side.shape[1])
    diff_score = np.mean(np.abs(left_side[:, :min_w].astype(float) - right_side[:, :min_w].astype(float)))
    if diff_score < 35.0:  # Ngưỡng phát hiện cấu trúc hai khung hình tương đồng
        fov_status += " (DUAL FRAME DETECTED - REJECT)"
        fov_color = "#e74c3c"

    # 2. TIÊU CHÍ ĐỘ SÁNG (SNR)
    mean_signal = np.mean(gray)
    std_noise = np.std(gray) + 1e-5
    snr_val = mean_signal / std_noise
    
    if snr_val >= 5.0:
        snr_status = "Ideal"
        snr_color = "#2ecc71"
    elif 2.0 <= snr_val < 5.0:
        snr_status = "Acceptable (CLAHE Fixable)"
        snr_color = "#f1c40f"
    else:
        snr_status = "Discard"
        snr_color = "#e74c3c"

    # 3. TIÊU CHÍ ĐỘ TƯƠNG PHẢN (CNR)
    # Mô phỏng vùng nội mạc tử cung (A) và cơ tử cung (B) thông qua phân ngưỡng Otsu nhị phân nâng cao
    mask_a = (gray > mean_signal)
    mask_b = (gray <= mean_signal) & (gray > 15)
    sa = np.mean(gray[mask_a]) if np.any(mask_a) else 255.0
    sb = np.mean(gray[mask_b]) if np.any(mask_b) else 0.0
    cnr_val = np.abs(sa - sb) / std_noise
    
    if cnr_val >= 1.5:
        cnr_status = "Ideal (Good/Excellent)"
        cnr_color = "#2ecc71"
    elif 0.75 <= cnr_val < 1.5:
        cnr_status = "Acceptable (Fair - Adjust clipLimit)"
        cnr_color = "#f1c40f"
    else:
        cnr_status = "Discard (Poor)"
        cnr_color = "#e74c3c"

    # 4. TIÊU CHÍ ĐỘ SẮC NÉT (ENL & PSNR)
    # Tính toán ENL (Equivalent Number of Looks) trên vùng đồng nhất giả định
    local_std = cv2.blur(gray.astype(float), (15, 15))
    mu_local = cv2.blur(gray.astype(float), (15, 15))
    enl_map = (mu_local / (local_std + 1e-5)) ** 2
    enl_val = np.median(enl_map) / 10.0  # Chuẩn hóa về dải đo mô phỏng siêu âm
    
    if enl_val > 20.0:
        enl_status = "Ideal"
        enl_color = "#2ecc71"
    elif 10.0 <= enl_val <= 20.0:
        enl_status = "Acceptable"
        enl_color = "#f1c40f"
    else:
        enl_status = "Discard (Over-processed/Burnt)"
        enl_color = "#e74c3c"
        
    # Tính toán PSNR nếu có ảnh đối chứng gốc
    psnr_text = "PSNR: N/A"
    psnr_color = "#7f8c8d"
    if img_reference is not None:
        gray_ref = cv2.cvtColor(img_reference, cv2.COLOR_RGB2GRAY)
        mse = np.mean((gray.astype(float) - gray_ref.astype(float)) ** 2)
        if mse == 0:
            psnr_text = "PSNR: Inf (Ideal)"
            psnr_color = "#2ecc71"
        else:
            psnr_val = 20 * np.log10(255.0 / np.sqrt(mse))
            if psnr_val > 25.0:
                psnr_text = f"PSNR: {psnr_val:.2f} dB (Ideal)"
                psnr_color = "#2ecc71"
            elif 19.0 <= psnr_val <= 25.0:
                psnr_text = f"PSNR: {psnr_val:.2f} dB (Acceptable)"
                psnr_color = "#f1c40f"
            else:
                psnr_text = f"PSNR: {psnr_val:.2f} dB (Discard)"
                psnr_color = "#e74c3c"

    # 5. TIÊU CHÍ NHIỄU HẠT & KHẢ NĂNG BẮT NÉT (VoL & Tenengrad)
    # Variance of Laplacian (VoL) để phát hiện ảnh mất nét
    vol_val = cv2.Laplacian(gray, cv2.CV_64F).var()
    if vol_val >= 150.0:
        vol_status = "Ideal (Sharp)"
        vol_color = "#2ecc71"
    elif 50.0 <= vol_val < 150.0:
        vol_status = "Acceptable (Soft edges - Need Sharpening)"
        vol_color = "#f1c40f"
    else:
        vol_status = "Discard (Out of focus)"
        vol_color = "#e74c3c"
        
    # Tenengrad Gradient để đánh giá cấu trúc biên sau chỉnh sửa
    gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    gz = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    tenengrad_val = np.mean(gx**2 + gz**2) / 100.0  # Chuẩn hóa giá trị hiển thị

    return {
        "fov_r": r_fov, "fov_status": fov_status, "fov_color": fov_color,
        "snr": snr_val, "snr_status": snr_status, "snr_color": snr_color,
        "cnr": cnr_val, "cnr_status": cnr_status, "cnr_color": cnr_color,
        "enl": enl_val, "enl_status": enl_status, "enl_color": enl_color,
        "psnr_text": psnr_text, "psnr_color": psnr_color,
        "vol": vol_val, "vol_status": vol_status, "vol_color": vol_color,
        "tenengrad": tenengrad_val
    }


# ==============================================================================
# THÀNH PHẦN HIỂN THỊ ẢNH GỐC HỖ TRỢ KÉO THẢ CHUỘT ĐỂ KHOANH VÙNG (ROI)
# ==============================================================================
class CustomImageLabel(QLabel):
    def __init__(self, text, main_app):
        super().__init__(text)
        self.main_app = main_app
        self.roi_enabled = False       
        self.is_drawing = False        
        self.start_point = QPoint()
        self.end_point = QPoint()
        self.current_rect = QRect()
        self.actual_pixmap_rect = QRect() 

    def set_roi_enabled(self, enabled):
        self.roi_enabled = enabled
        if not enabled:
            self.current_rect = QRect()
            self.update()

    def mousePressEvent(self, event):
        if self.roi_enabled and event.button() == Qt.MouseButton.LeftButton and self.pixmap():
            if self.actual_pixmap_rect.contains(event.position().toPoint()):
                self.is_drawing = True
                self.start_point = event.position().toPoint()
                self.end_point = self.start_point
                self.current_rect = QRect(self.start_point, self.end_point)
                self.update()

    def mouseMoveEvent(self, event):
        if self.roi_enabled and self.is_drawing:
            p = event.position().toPoint()
            x = max(self.actual_pixmap_rect.left(), min(p.x(), self.actual_pixmap_rect.right()))
            y = max(self.actual_pixmap_rect.top(), min(p.y(), self.actual_pixmap_rect.bottom()))
            
            self.end_point = QPoint(x, y)
            self.current_rect = QRect(self.start_point, self.end_point).normalized()
            self.update()

    def mouseReleaseEvent(self, event):
        if self.roi_enabled and event.button() == Qt.MouseButton.LeftButton and self.is_drawing:
            self.is_drawing = False
            if self.current_rect.width() > 5 and self.current_rect.height() > 5:
                self.calculate_orig_coordinates()
            else:
                self.current_rect = QRect()
                self.main_app.clear_roi_selection()
            self.update()

    def paintEvent(self, event):
        super().paintEvent(event)
        if self.roi_enabled and not self.current_rect.isNull():
            painter = QPainter(self)
            pen = QPen(QColor(255, 215, 0), 2, Qt.PenStyle.DashLine)
            painter.setPen(pen)
            painter.drawRect(self.current_rect)
            painter.end()

    def calculate_orig_coordinates(self):
        if self.main_app.orig_image is None or not self.pixmap():
            return
            
        orig_h, orig_w = self.main_app.orig_image.shape[:2]
        
        scale_x = orig_w / self.actual_pixmap_rect.width()
        scale_y = orig_h / self.actual_pixmap_rect.height()
        
        offset_x = self.current_rect.left() - self.actual_pixmap_rect.left()
        offset_y = self.current_rect.top() - self.actual_pixmap_rect.top()
        
        orig_xmin = int(offset_x * scale_x)
        orig_ymin = int(offset_y * scale_y)
        orig_xmax = int((offset_x + self.current_rect.width()) * scale_x)
        orig_ymax = int((offset_y + self.current_rect.height()) * scale_y)
        
        orig_xmin = max(0, min(orig_xmin, orig_w))
        orig_ymin = max(0, min(orig_ymin, orig_h))
        orig_xmax = max(0, min(orig_xmax, orig_w))
        orig_ymax = max(0, min(orig_ymax, orig_h))
        
        self.main_app.update_roi_area(orig_xmin, orig_ymin, orig_xmax, orig_ymax)


# ==============================================================================
# GIAO DIỆN CHÍNH PYQT6 VÀ LUỒNG XỬ LÝ TÍCH HỢP
# ==============================================================================
class UltrasoundProcessorApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Medical Ultrasound Imaging Workstation")
        self.setGeometry(50, 50, 1440, 880)
        
        self.orig_image = None         
        self.left_view_image = None     
        self.processed_image = None     
        
        self.highlight_mask = None      
        self.all_detected_boxes = []    
        self.roi_coordinates = None     

        self.init_ui()
        
    def init_ui(self):
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        main_layout = QHBoxLayout(main_widget)
        main_layout.setContentsMargins(10, 10, 10, 10)
        main_layout.setSpacing(10)
        
        # ================= LEFT PANEL: LAYOUT & COMPONENTS =================
        control_panel = QVBoxLayout()
        control_panel.setContentsMargins(0, 0, 5, 0)
        control_panel.setSpacing(10)
        
        file_group = QGroupBox("Workstation Control")
        file_layout = QVBoxLayout(file_group)
        file_layout.setSpacing(8)
        
        self.btn_open = QPushButton("📂 Open Ultrasound Image")
        self.btn_open.clicked.connect(self.load_image)
        self.btn_open.setStyleSheet("font-weight: bold; padding: 5px;")
        
        self.btn_toggle_roi = QPushButton("🎯 Turn ON ROI Selection")
        self.btn_toggle_roi.setCheckable(True)
        self.btn_toggle_roi.clicked.connect(self.toggle_roi_mode)
        self.btn_toggle_roi.setStyleSheet("background-color: #8e44ad; color: white; padding: 5px;")
        self.btn_toggle_roi.setEnabled(False)
        
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
        file_layout.addWidget(self.btn_toggle_roi)
        file_layout.addWidget(self.btn_save_snapshot)
        file_layout.addWidget(self.btn_reset_left)
        file_layout.addWidget(self.btn_reset_params)
        control_panel.addWidget(file_group)
        
        export_group = QGroupBox("Export Result")
        export_layout = QHBoxLayout(export_group)
        export_layout.setSpacing(8)
        
        self.btn_export_left = QPushButton("💾 Save Left")
        self.btn_export_left.clicked.connect(lambda: self.export_image_to_disk("left"))
        self.btn_export_left.setEnabled(False)
        self.btn_export_left.setStyleSheet("background-color: #27ae60; color: white; font-weight: bold; padding: 5px;")
        
        self.btn_export_right = QPushButton("💾 Save Right")
        self.btn_export_right.clicked.connect(lambda: self.export_image_to_disk("right"))
        self.btn_export_right.setEnabled(False)
        self.btn_export_right.setStyleSheet("background-color: #2980b9; color: white; font-weight: bold; padding: 5px;")
        
        export_layout.addWidget(self.btn_export_left)
        export_layout.addWidget(self.btn_export_right)
        control_panel.addWidget(export_group)
        
        param_group = QGroupBox("Advanced Pipeline")
        param_layout = QVBoxLayout(param_group)
        param_layout.setSpacing(8)
        
        self.chk_highlight = QCheckBox("🎯 Highlight (Tô đỏ dấu đo)")
        self.chk_highlight.setStyleSheet("font-weight: bold; color: #e74c3c; margin-bottom: 2px;")
        self.chk_highlight.stateChanged.connect(self.process_and_display)
        param_layout.addWidget(self.chk_highlight)
        
        self.lbl_thresh_title = QLabel("Caliper Match Threshold:")
        param_layout.addWidget(self.lbl_thresh_title)
        self.slider_detect_thresh = QSlider(Qt.Orientation.Horizontal)
        self.slider_detect_thresh.setRange(40, 90)  
        self.slider_detect_thresh.setValue(62)      
        self.slider_detect_thresh.valueChanged.connect(self.process_and_display)
        self.lbl_thresh_value = QLabel("0.62")
        self.lbl_thresh_value.setAlignment(Qt.AlignmentFlag.AlignRight)
        param_layout.addWidget(self.slider_detect_thresh)
        param_layout.addWidget(self.lbl_thresh_value)
        
        param_layout.addWidget(QLabel("<b>1. Denoise Method:</b>"))
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
        
        param_layout.addWidget(QLabel("<b>2. Brightness (Độ sáng):</b>"))
        self.slider_brightness = QSlider(Qt.Orientation.Horizontal)
        self.slider_brightness.setRange(-100, 100)
        self.slider_brightness.setValue(0)
        self.slider_brightness.valueChanged.connect(self.process_and_display)
        self.lbl_brightness = QLabel("0")
        self.lbl_brightness.setAlignment(Qt.AlignmentFlag.AlignRight)
        param_layout.addWidget(self.slider_brightness)
        param_layout.addWidget(self.lbl_brightness)
        
        param_layout.addWidget(QLabel("<b>3. Contrast Method:</b>"))
        self.combo_contrast_method = QComboBox()
        self.combo_contrast_method.addItems([
            "Linear Contrast (Tuyến tính)",
            "CLAHE (Thích ứng y tế)"
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
        
        param_layout.addWidget(QLabel("<b>4. Sharpness Method:</b>"))
        self.combo_sharp_method = QComboBox()
        self.combo_sharp_method.addItems([
            "Unsharp Masking (Tần số biên)",
            "Morphological Sharpen"
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
        
        scroll_area = QScrollArea()
        scroll_area.setWidget(control_container)
        scroll_area.setWidgetResizable(True) 
        scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff) 
        scroll_area.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)     
        scroll_area.setFixedWidth(310) 
        scroll_area.setStyleSheet("QScrollArea { border: none; background: transparent; }")
        
        main_layout.addWidget(scroll_area) 
        
        # ================= RIGHT PANEL: VISUALIZATION & CRITERIA GROUPS =================
        view_layout = QHBoxLayout()
        view_layout.setSpacing(12)
        
        # --- CẤU TRÚC PHẦN HIỂN THỊ TRÁI (ẢNH GỐC HOẶC SNAPSHOT) ---
        left_container = QWidget()
        left_vbox = QVBoxLayout(left_container)
        left_vbox.setContentsMargins(0, 0, 0, 0)
        left_vbox.setSpacing(6)
        
        self.lbl_orig_view = CustomImageLabel("No image loaded", self)
        self.lbl_orig_view.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.lbl_orig_view.setStyleSheet("border: 1px solid #444; background-color: #111; color: #777;")
        
        # Cắt 1 phần khung dưới làm panel điền tiêu chí ảnh Trái
        left_crit_group = QGroupBox("Left Image Criteria Evaluation Panel")
        left_crit_group.setStyleSheet("QGroupBox { font-weight: bold; color: #e67e22; }")
        left_crit_layout = QVBoxLayout(left_crit_group)
        left_crit_layout.setContentsMargins(10, 8, 10, 8)
        left_crit_layout.setSpacing(4)
        
        self.lbl_l_fov = QLabel("Field of View (R): N/A")
        self.lbl_l_brightness = QLabel("Brightness (SNR): N/A")
        self.lbl_l_contrast = QLabel("Contrast (CNR): N/A")
        self.lbl_l_sharpness = QLabel("Sharpness (ENL): N/A")
        self.lbl_l_speckle = QLabel("Speckle Noise (VoL): N/A")
        
        left_crit_layout.addWidget(self.lbl_l_fov)
        left_crit_layout.addWidget(self.lbl_l_brightness)
        left_crit_layout.addWidget(self.lbl_l_contrast)
        left_crit_layout.addWidget(self.lbl_l_sharpness)
        left_crit_layout.addWidget(self.lbl_l_speckle)
        
        left_vbox.addWidget(self.lbl_orig_view, stretch=1)
        left_vbox.addWidget(left_crit_group)
        
        # --- CẤU TRÚC PHẦN HIỂN THỊ PHẢI (ẢNH ĐÃ TIỀN XỬ LÝ) ---
        right_container = QWidget()
        right_vbox = QVBoxLayout(right_container)
        right_vbox.setContentsMargins(0, 0, 0, 0)
        right_vbox.setSpacing(6)
        
        self.lbl_proc_view = QLabel("No image loaded")
        self.lbl_proc_view.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.lbl_proc_view.setStyleSheet("border: 1px solid #444; background-color: #111; color: #777;")
        
        # Cắt 1 phần khung dưới làm panel điền tiêu chí ảnh Phải
        right_crit_group = QGroupBox("Processed Pipeline Criteria Evaluation Panel")
        right_crit_group.setStyleSheet("QGroupBox { font-weight: bold; color: #3498db; }")
        right_crit_layout = QVBoxLayout(right_crit_group)
        right_crit_layout.setContentsMargins(10, 8, 10, 8)
        right_crit_layout.setSpacing(4)
        
        self.lbl_r_fov = QLabel("Field of View (R): N/A")
        self.lbl_r_brightness = QLabel("Brightness (SNR): N/A")
        self.lbl_r_contrast = QLabel("Contrast (CNR): N/A")
        self.lbl_r_sharpness = QLabel("Sharpness (ENL & PSNR): N/A")
        self.lbl_r_speckle = QLabel("Speckle Noise (VoL & Tenengrad): N/A")
        
        right_crit_layout.addWidget(self.lbl_r_fov)
        right_crit_layout.addWidget(self.lbl_r_brightness)
        right_crit_layout.addWidget(self.lbl_r_contrast)
        right_crit_layout.addWidget(self.lbl_r_sharpness)
        right_crit_layout.addWidget(self.lbl_r_speckle)
        
        right_vbox.addWidget(self.lbl_proc_view, stretch=1)
        right_vbox.addWidget(right_crit_group)
        
        view_layout.addWidget(left_container, stretch=1)
        view_layout.addWidget(right_container, stretch=1)
        main_layout.addLayout(view_layout, stretch=1)
        
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.status_bar.showMessage("Ready. Please open an ultrasound image.")
        
        self.toggle_controls(False)
        main_widget.installEventFilter(self)

    # ==============================================================================
    # QUẢN LÝ CHẾ ĐỘ KHOANH VÙNG ROI ĐỘNG
    # ==============================================================================
    def toggle_roi_mode(self):
        is_checked = self.btn_toggle_roi.isChecked()
        if is_checked:
            self.btn_toggle_roi.setText("🛑 Turn OFF ROI Selection")
            self.btn_toggle_roi.setStyleSheet("background-color: #d35400; color: white; padding: 5px;")
            self.lbl_orig_view.set_roi_enabled(True)
            self.status_bar.showMessage("ROI Mode: ON. Click and drag left mouse button on left image.")
        else:
            self.btn_toggle_roi.setText("🎯 Turn ON ROI Selection")
            self.btn_toggle_roi.setStyleSheet("background-color: #8e44ad; color: white; padding: 5px;")
            self.lbl_orig_view.set_roi_enabled(False)
            self.clear_roi_selection()

    def update_roi_area(self, xmin, ymin, xmax, ymax):
        self.roi_coordinates = [xmin, ymin, xmax, ymax]
        self.status_bar.showMessage(f"ROI area locked: X[{xmin}->{xmax}], Y[{ymin}->{ymax}].")
        self.process_and_display()

    def clear_roi_selection(self):
        self.roi_coordinates = None
        self.status_bar.showMessage("ROI cleared. Scanning the full image area.")
        self.process_and_display()

    # ==============================================================================
    # THUẬT TOÁN TÌM KIẾM THEO TEMPLATE DẤU ĐO
    # ==============================================================================
    def extract_highlight_mask_from_original(self, img_rgb, templates_dir, threshold=0.62, roi_box=None):
        h_shape, w_shape = img_rgb.shape[:2]
        mask = np.zeros((h_shape, w_shape), dtype=np.uint8)
        boxes = []
        
        if not os.path.exists(templates_dir):
            return mask, boxes

        valid_extensions = (".jpg", ".jpeg", ".png", ".bmp")
        template_files = [f for f in os.listdir(templates_dir) if f.lower().endswith(valid_extensions)]

        if not template_files:
            return mask, boxes

        gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
        gray_filtered = cv2.bilateralFilter(gray, d=9, sigmaColor=75, sigmaSpace=75)

        if roi_box is not None:
            roi_xmin, roi_ymin, roi_xmax, roi_ymax = roi_box
        else:
            roi_xmin, roi_ymin, roi_xmax, roi_ymax = 0, 0, w_shape, h_shape

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
                if x_start < 80 or y_start > (h_shape - 118): continue
                if y_start + h > h_shape or x_start + w > w_shape: continue
                if x_start < roi_xmin or (x_start + w) > roi_xmax or y_start < roi_ymin or (y_start + h) > roi_ymax: continue

                xmin = max(0, x_start - 2)
                ymin = max(0, y_start - 2)
                xmax = min(w_shape, x_start + w + 2)
                ymax = min(h_shape, y_start + h + 2)
                
                is_duplicate = False
                for b in boxes:
                    if abs(b['xmin'] - xmin) < 8 and abs(b['ymin'] - ymin) < 8:
                        is_duplicate = True
                        break
                if is_duplicate: continue

                gray_roi = gray[y_start:y_start+h, x_start:x_start+w]
                _, dynamic_mask = cv2.threshold(gray_roi, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                
                kernel = np.ones((3, 3), np.uint8)
                tpl_mask_dilated = cv2.dilate(tpl_mask, kernel, iterations=1)
                final_roi_mask = cv2.bitwise_and(tpl_mask_dilated, dynamic_mask)
                
                mask[y_start:y_start+h, x_start:x_start+w] = cv2.bitwise_or(
                    mask[y_start:y_start+h, x_start:x_start+w], final_roi_mask
                )
                boxes.append({'name': class_name, 'xmin': xmin, 'ymin': ymin, 'xmax': xmax, 'ymax': ymax})
                
        return mask, boxes

    def on_filter_changed(self, index):
        self.slider_noise.blockSignals(True)
        self.slider_noise.setValue(0) 
        if index == 0:
            self.lbl_noise_title.setText("Filter Strength:")
            self.slider_noise.setRange(0, 0)
        elif index == 1 or index == 2:
            self.lbl_noise_title.setText("Kernel Size (Mẫu lẻ):")
            self.slider_noise.setRange(0, 8)
        elif index == 3:
            self.lbl_noise_title.setText("Sigma Color/Space:")
            self.slider_noise.setRange(0, 20)
        elif index == 4:
            self.lbl_noise_title.setText("SRAD Iterations:")
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
        self.btn_toggle_roi.setEnabled(enabled)
        self.chk_highlight.setEnabled(enabled)
        self.slider_detect_thresh.setEnabled(enabled)
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

    def update_criteria_ui_labels(self, metrics, side):
        """Cập nhật văn bản và màu sắc cảnh báo lên vùng giao diện đã cắt bên dưới"""
        if metrics is None:
            return
            
        if side == "left":
            self.lbl_l_fov.setText(f"Field of View (R): {metrics['fov_r']:.3f} ➔ <b style='color:{metrics['fov_color']};'>{metrics['fov_status']}</b>")
            self.lbl_l_brightness.setText(f"Brightness (SNR): {metrics['snr']:.2f} ➔ <b style='color:{metrics['snr_color']};'>{metrics['snr_status']}</b>")
            self.lbl_l_contrast.setText(f"Contrast (CNR): {metrics['cnr']:.2f} ➔ <b style='color:{metrics['cnr_color']};'>{metrics['cnr_status']}</b>")
            self.lbl_l_sharpness.setText(f"Sharpness (ENL): {metrics['enl']:.2f} ➔ <b style='color:{metrics['enl_color']};'>{metrics['enl_status']}</b>")
            self.lbl_l_speckle.setText(f"Speckle (VoL): {metrics['vol']:.1f} ➔ <b style='color:{metrics['vol_color']};'>{metrics['vol_status']}</b>")
        else:
            self.lbl_r_fov.setText(f"Field of View (R): {metrics['fov_r']:.3f} ➔ <b style='color:{metrics['fov_color']};'>{metrics['fov_status']}</b>")
            self.lbl_r_brightness.setText(f"Brightness (SNR): {metrics['snr']:.2f} ➔ <b style='color:{metrics['snr_color']};'>{metrics['snr_status']}</b>")
            self.lbl_r_contrast.setText(f"Contrast (CNR): {metrics['cnr']:.2f} ➔ <b style='color:{metrics['cnr_color']};'>{metrics['cnr_status']}</b>")
            self.lbl_r_sharpness.setText(f"Sharpness (ENL): {metrics['enl']:.2f} | <span style='color:{metrics['psnr_color']};'>{metrics['psnr_text']}</span>")
            self.lbl_r_speckle.setText(f"Speckle (VoL): {metrics['vol']:.1f} | Tenengrad Post: {metrics['tenengrad']:.1f} ➔ <b style='color:{metrics['vol_color']};'>{metrics['vol_status']}</b>")

    def save_to_left_view(self):
        if self.processed_image is not None:
            self.left_view_image = self.processed_image.copy()
            self.display_on_label(self.left_view_image, self.lbl_orig_view)
            
            metrics = analyze_medical_criteria(self.left_view_image, self.orig_image)
            self.update_criteria_ui_labels(metrics, "left")
            self.status_bar.showMessage("Snapshot locked to Left View and recalculating criteria.")

    def reset_left_view(self):
        if self.orig_image is not None:
            if self.chk_highlight.isChecked() and self.highlight_mask is not None:
                tmp_left = self.orig_image.copy()
                if tmp_left.shape[:2] == self.highlight_mask.shape:
                    tmp_left[self.highlight_mask > 0] = [255, 0, 0]
                self.left_view_image = tmp_left
            else:
                self.left_view_image = self.orig_image.copy()
                
            self.display_on_label(self.left_view_image, self.lbl_orig_view)
            
            metrics = analyze_medical_criteria(self.orig_image, None)
            self.update_criteria_ui_labels(metrics, "left")
            self.status_bar.showMessage("Left View reset to Base Image.")

    def load_image(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Open Image File", "", "Images (*.png *.jpg *.jpeg *.bmp *.tiff)")
        if file_path:
            img = cv2.imread(file_path)
            if img is None:
                self.status_bar.showMessage("Error: Could not decode image file.")
                return
            
            self.orig_image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            self.left_view_image = self.orig_image.copy()
            
            self.roi_coordinates = None
            self.btn_toggle_roi.setChecked(False)
            self.btn_toggle_roi.setText("🎯 Turn ON ROI Selection")
            self.btn_toggle_roi.setStyleSheet("background-color: #8e44ad; color: white; padding: 5px;")
            self.lbl_orig_view.set_roi_enabled(False)

            self.toggle_controls(True)
            self.reset_sliders()
            
            metrics_left = analyze_medical_criteria(self.orig_image, None)
            self.update_criteria_ui_labels(metrics_left, "left")
            
            self.status_bar.showMessage(f"Loaded: {os.path.basename(file_path)}")

    def process_and_display(self):
        if self.orig_image is None:
            return
            
        thresh_slider_val = self.slider_detect_thresh.value()
        current_threshold = thresh_slider_val / 100.0
        self.lbl_thresh_value.setText(f"{current_threshold:.2f}")
        
        filter_idx = self.combo_filter.currentIndex()
        n_val = self.slider_noise.value()
        b_val = self.slider_brightness.value()
        contrast_idx = self.combo_contrast_method.currentIndex()
        c_val = self.slider_contrast.value()
        sharp_idx = self.combo_sharp_method.currentIndex()
        s_val = self.slider_sharpness.value()
        
        if filter_idx == 0 or n_val == 0: self.lbl_noise_value.setText("Off")
        elif filter_idx == 1 or filter_idx == 2: self.lbl_noise_value.setText(f"Kernel: {n_val*2+1}x{n_val*2+1}")
        elif filter_idx == 3: self.lbl_noise_value.setText(f"Sigma: {n_val*4}")
        elif filter_idx == 4: self.lbl_noise_value.setText(f"{n_val} Iters")
            
        self.lbl_brightness.setText(str(b_val))
        self.lbl_sharpness.setText(str(s_val))
        if contrast_idx == 0:
            self.lbl_contrast.setText(f"Alpha: {c_val / 100.0:.2f}")
        else:
            if c_val == 0: self.lbl_contrast.setText("Off")
            else: self.lbl_contrast.setText(f"Clip: {c_val}.0")
            
        if self.chk_highlight.isChecked():
            self.highlight_mask, self.all_detected_boxes = self.extract_highlight_mask_from_original(
                self.orig_image, "templates", threshold=current_threshold, roi_box=self.roi_coordinates
            )
            if self.left_view_image is not None and self.left_view_image.shape == self.orig_image.shape:
                tmp_left = self.orig_image.copy()
                tmp_left[self.highlight_mask > 0] = [255, 0, 0]
                self.left_view_image = tmp_left
        
        # --- THỰC THI PIPELINE XỬ LÝ ẢNH ---
        img = self.orig_image.copy()
        
        # 1. Khử nhiễu
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
                
        # 2. Độ sáng
        if b_val != 0:
            img = np.clip(img.astype(np.float32) + b_val, 0, 255).astype(np.uint8)
            
        # 3. Độ tương phản
        if contrast_idx == 0: 
            alpha = c_val / 100.0
            img = np.clip(img.astype(np.float32) * alpha, 0, 255).astype(np.uint8)
        else: 
            if c_val > 0: 
                gray_c = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
                clahe = cv2.createCLAHE(clipLimit=float(c_val), tileGridSize=(8, 8))
                img = cv2.cvtColor(clahe.apply(gray_c), cv2.COLOR_GRAY2RGB)
            
        # 4. Độ sắc nét
        if s_val > 0:
            if sharp_idx == 0: 
                blurred = cv2.GaussianBlur(img, (5, 5), 0)
                weight = s_val * 0.3
                img = np.clip(cv2.addWeighted(img, 1.0 + weight, blurred, -weight, 0), 0, 255).astype(np.uint8)
            else: 
                kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
                img = cv2.subtract(cv2.add(img, cv2.morphologyEx(img, cv2.MORPH_TOPHAT, kernel)), cv2.morphologyEx(img, cv2.MORPH_BLACKHAT, kernel))
                
        if self.chk_highlight.isChecked() and self.highlight_mask is not None:
            if img.shape[:2] == self.highlight_mask.shape:
                img[self.highlight_mask > 0] = [255, 0, 0]
            
        self.processed_image = img
        
        # Tính toán tiêu chí cho ảnh kết quả thời gian thực
        metrics_right = analyze_medical_criteria(self.processed_image, self.orig_image)
        self.update_criteria_ui_labels(metrics_right, "right")
        
        self.display_on_label(self.left_view_image, self.lbl_orig_view)
        self.display_on_label(self.processed_image, self.lbl_proc_view)

    def export_image_to_disk(self, target_view):
        img_to_save = self.left_view_image if target_view == "left" else self.processed_image
        view_label = "Left_View" if target_view == "left" else "Right_View"
        if img_to_save is None: return
            
        output_dir = "./output"
        if not os.path.exists(output_dir): os.makedirs(output_dir)
        filename = f"ultrasound_{view_label}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        full_path = os.path.join(output_dir, filename)
        
        if cv2.imwrite(full_path, cv2.cvtColor(img_to_save, cv2.COLOR_RGB2BGR)):
            self.status_bar.showMessage(f"Saved to: {full_path}")

    def reset_sliders(self):
        sliders = [self.slider_detect_thresh, self.slider_noise, self.slider_brightness, self.slider_contrast, self.slider_sharpness]
        for s in sliders: s.blockSignals(True)
        self.combo_filter.blockSignals(True)
        self.combo_contrast_method.blockSignals(True)
        self.combo_sharp_method.blockSignals(True)
        
        self.slider_detect_thresh.setValue(62)
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
        
        for s in sliders: s.blockSignals(False)
        self.combo_filter.blockSignals(False)
        self.combo_contrast_method.blockSignals(False)
        self.combo_sharp_method.blockSignals(False)
        self.process_and_display()

    def eventFilter(self, source, event):
        if event.type() == QEvent.Type.Resize and self.orig_image is not None:
            self.display_on_label(self.left_view_image, self.lbl_orig_view)
            self.display_on_label(self.processed_image, self.lbl_proc_view)
        return super().eventFilter(source, event)

    def display_on_label(self, rgb_array, label_element):
        if rgb_array is None: return
        height, width, channel = rgb_array.shape
        q_img = QImage(rgb_array.data, width, height, channel * width, QImage.Format.Format_RGB888)
        scaled_pixmap = QPixmap.fromImage(q_img).scaled(
            label_element.width() - 4, label_element.height() - 4, 
            Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation
        )
        label_element.setPixmap(scaled_pixmap)
        if isinstance(label_element, CustomImageLabel):
            lx = (label_element.width() - scaled_pixmap.width()) // 2
            ly = (label_element.height() - scaled_pixmap.height()) // 2
            label_element.actual_pixmap_rect = QRect(lx, ly, scaled_pixmap.width(), scaled_pixmap.height())

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = UltrasoundProcessorApp()
    window.show()
    sys.exit(app.exec())