import sys
import os
import datetime
import shutil
import json
import xml.etree.ElementTree as ET
from xml.dom import minidom
import numpy as np
import cv2
from numba import jit
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QLabel, 
                             QPushButton, QSlider, QFileDialog, QVBoxLayout, 
                             QHBoxLayout, QGroupBox, QStatusBar, QComboBox, QCheckBox,
                             QScrollArea, QDialog, QDialogButtonBox, QMessageBox,
                             QProgressDialog, QSizePolicy)
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

def compute_enl_hras(gray, block_size=32, top_k=8, return_boxes=False, search_roi=None):
    """
    Automatic ENL estimation using an HRAS-like strategy.

    If search_roi is provided as (xmin, ymin, xmax, ymax), HRAS searches only
    inside that manually selected ROI. Returned boxes are still in full-image
    coordinates so they can be drawn correctly on the displayed image.

    ENL = mean^2 / std^2 is computed on the selected homogeneous blocks.
    """
    gray_f = gray.astype(np.float32)
    img_h, img_w = gray_f.shape

    if search_roi is not None:
        rx1, ry1, rx2, ry2 = [int(v) for v in search_roi]
        rx1 = max(0, min(rx1, img_w - 1))
        ry1 = max(0, min(ry1, img_h - 1))
        rx2 = max(rx1 + 1, min(rx2, img_w))
        ry2 = max(ry1 + 1, min(ry2, img_h))
    else:
        rx1, ry1, rx2, ry2 = 0, 0, img_w, img_h

    roi_area = gray_f[ry1:ry2, rx1:rx2]
    h, w = roi_area.shape

    candidates = []

    # If the manual ROI is smaller than the default block, shrink the block.
    effective_block = min(block_size, h, w)
    if effective_block < 8:
        m = float(np.mean(roi_area))
        s = float(np.std(roi_area))
        fallback_enl = (m * m) / (s * s + 1e-8)
        if return_boxes:
            return fallback_enl, [(rx1, ry1, rx2, ry2)]
        return fallback_enl

    for y in range(0, h - effective_block + 1, effective_block):
        for x in range(0, w - effective_block + 1, effective_block):
            block = roi_area[y:y + effective_block, x:x + effective_block]
            mean = float(np.mean(block))
            std = float(np.std(block))

            # Reject very dark/bright blocks and nearly constant invalid areas.
            if mean < 15.0 or mean > 240.0 or std < 1e-6:
                continue

            gx = cv2.Sobel(block, cv2.CV_32F, 1, 0, ksize=3)
            gy = cv2.Sobel(block, cv2.CV_32F, 0, 1, ksize=3)
            grad = float(np.mean(np.sqrt(gx * gx + gy * gy)))

            enl = (mean * mean) / (std * std + 1e-8)
            score = 1.0 / (std + grad + 1e-6)

            full_x1 = rx1 + x
            full_y1 = ry1 + y
            full_x2 = full_x1 + effective_block
            full_y2 = full_y1 + effective_block

            candidates.append({
                "score": score,
                "enl": enl,
                "box": (full_x1, full_y1, full_x2, full_y2),
                "mean": mean,
                "std": std,
                "grad": grad,
            })

    if not candidates:
        m = float(np.mean(roi_area))
        s = float(np.std(roi_area))
        fallback_enl = (m * m) / (s * s + 1e-8)
        if return_boxes:
            return fallback_enl, [(rx1, ry1, rx2, ry2)]
        return fallback_enl

    candidates.sort(key=lambda item: item["score"], reverse=True)
    selected = candidates[:min(top_k, len(candidates))]
    vals = [item["enl"] for item in selected]
    boxes = [item["box"] for item in selected]
    enl_value = float(np.median(vals))

    if return_boxes:
        return enl_value, boxes
    return enl_value


def compute_cnr_from_two_rois(gray, roi1_box=None, roi2_box=None):
    """
    Compute CNR using two user-selected ROIs:
    ROI 1: target/tissue region of interest
    ROI 2: background or reference tissue region

    CNR = |mu1 - mu2| / sqrt(std1^2 + std2^2)
    """
    if roi1_box is None or roi2_box is None:
        return None

    h, w = gray.shape

    def crop_roi(box):
        x1, y1, x2, y2 = [int(v) for v in box]
        x1 = max(0, min(x1, w - 1))
        y1 = max(0, min(y1, h - 1))
        x2 = max(x1 + 1, min(x2, w))
        y2 = max(y1 + 1, min(y2, h))
        return gray[y1:y2, x1:x2].astype(np.float32)

    roi1 = crop_roi(roi1_box)
    roi2 = crop_roi(roi2_box)

    if roi1.size == 0 or roi2.size == 0:
        return None

    mu1 = float(np.mean(roi1))
    mu2 = float(np.mean(roi2))
    std1 = float(np.std(roi1))
    std2 = float(np.std(roi2))

    return abs(mu1 - mu2) / (np.sqrt(std1 * std1 + std2 * std2) + 1e-8)


def prettify_xml(elem):
    rough_string = ET.tostring(elem, "utf-8")
    reparsed = minidom.parseString(rough_string)
    return reparsed.toprettyxml(indent="\t")


def save_to_combined_xml(output_path, filename, img_shape, boxes, folder_name="output"):
    """Save Pascal VOC XML for detected caliper boxes."""
    annotation = ET.Element("annotation")
    ET.SubElement(annotation, "folder").text = folder_name
    ET.SubElement(annotation, "filename").text = filename
    ET.SubElement(annotation, "path").text = os.path.abspath(output_path)

    source = ET.SubElement(annotation, "source")
    ET.SubElement(source, "database").text = "Unknown"

    size = ET.SubElement(annotation, "size")
    ET.SubElement(size, "width").text = str(img_shape[1])
    ET.SubElement(size, "height").text = str(img_shape[0])
    ET.SubElement(size, "depth").text = str(img_shape[2])

    ET.SubElement(annotation, "segmented").text = "0"

    for box in boxes:
        obj = ET.SubElement(annotation, "object")
        ET.SubElement(obj, "name").text = box["name"]
        ET.SubElement(obj, "pose").text = "Unspecified"
        ET.SubElement(obj, "truncated").text = "0"
        ET.SubElement(obj, "difficult").text = "0"
        bndbox = ET.SubElement(obj, "bndbox")
        ET.SubElement(bndbox, "xmin").text = str(box["xmin"])
        ET.SubElement(bndbox, "ymin").text = str(box["ymin"])
        ET.SubElement(bndbox, "xmax").text = str(box["xmax"])
        ET.SubElement(bndbox, "ymax").text = str(box["ymax"])

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(prettify_xml(annotation))





# ==============================================================================
# HỆ THỐNG THUẬT TOÁN ĐÁNH GIÁ TIÊU CHÍ CHẤT LƯỢNG THEO NHÓM NGHIÊN CỨU
# ==============================================================================
def analyze_medical_criteria(img_current, img_reference=None, roi_box=None, roi2_box=None):
    """
    Tính toán chi tiết các tiêu chí y khoa dựa trên các công thức khoa học:
    Trường nhìn (R), Độ sáng (SNR), Độ tương phản (CNR), Nhiễu hạt (ENL), độ sắc nét (VoL/Tenengrad), PSNR, Nhiễu hạt (VoL, Tenengrad)
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
    # Nếu người dùng đã khoanh đủ ROI 1 và ROI 2:
    # ROI 1 = vùng mô cần quan sát, ROI 2 = vùng nền hoặc mô đối chứng.
    # CNR = |mu1 - mu2| / sqrt(std1^2 + std2^2)
    roi_cnr = compute_cnr_from_two_rois(gray, roi_box, roi2_box)
    if roi_cnr is not None:
        cnr_val = roi_cnr
    else:
        # Fallback tự động cũ khi chưa có đủ 2 ROI.
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

    # 4. TIÊU CHÍ NHIỄU HẠT / ĐỘ ĐỒNG NHẤT (ENL) & PSNR
    enl_val = compute_enl_hras(gray, search_roi=roi_box)

    # ENL thresholds for ultrasound preprocessing evaluation
    # Ideal: ENL > 20
    # Acceptable: ENL between 10 and 20
    # Discard: ENL < 10
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

    # 5. TIÊU CHÍ ĐỘ SẮC NÉT & KHẢ NĂNG BẮT NÉT (VoL & Tenengrad)
    # Nếu có ROI 1, VoL và Tenengrad chỉ được tính trong ROI 1.
    # Nếu chưa có ROI 1, tính trên toàn ảnh như trước.
    metric_gray = gray
    if roi_box is not None:
        x1, y1, x2, y2 = [int(v) for v in roi_box]
        x1 = max(0, min(x1, w - 1))
        y1 = max(0, min(y1, h - 1))
        x2 = max(x1 + 1, min(x2, w))
        y2 = max(y1 + 1, min(y2, h))
        metric_gray = gray[y1:y2, x1:x2]

    # Variance of Laplacian (VoL) để phát hiện ảnh mất nét / mất chi tiết biên
    vol_val = cv2.Laplacian(metric_gray, cv2.CV_64F).var()
    if vol_val >= 150.0:
        vol_status = "Ideal (Sharp)"
        vol_color = "#2ecc71"
    elif 50.0 <= vol_val < 150.0:
        vol_status = "Acceptable (Soft edges - Need Sharpening)"
        vol_color = "#f1c40f"
    else:
        vol_status = "Discard (Out of focus)"
        vol_color = "#e74c3c"
        
    # Tenengrad Gradient để đánh giá cấu trúc biên / độ sắc nét
    gx = cv2.Sobel(metric_gray, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(metric_gray, cv2.CV_64F, 0, 1, ksize=3)
    tenengrad_val = np.mean(gx**2 + gy**2) / 100.0  # Chuẩn hóa giá trị hiển thị

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
    def __init__(self, text, main_app, show_roi_overlay=True, view_id="left"):
        super().__init__(text)
        self.main_app = main_app
        self.show_roi_overlay = show_roi_overlay
        self.view_id = view_id
        self.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Ignored)
        self.setMinimumSize(120, 120)

        self.roi_enabled = False
        self.is_drawing = False
        self.start_point = QPoint()
        self.end_point = QPoint()
        self.current_rect = QRect()
        self.actual_pixmap_rect = QRect()

        # Zoom/pan state. We draw the pixmap ourselves so the QWidget size never
        # grows with the image. This prevents QWindowsWindow::setGeometry crashes.
        self.rendered_pixmap = None
        self.pan_offset = QPoint(0, 0)
        self.is_panning = False
        self.pan_start_mouse = QPoint()
        self.pan_start_offset = QPoint()
        self.setMouseTracking(True)

    def pixmap(self):
        # Keep compatibility with existing code that checks self.pixmap().
        return self.rendered_pixmap

    def set_custom_pixmap(self, pixmap):
        self.rendered_pixmap = pixmap
        self.setText("")
        self.update_pixmap_rect()
        self.update()

    def update_pixmap_rect(self):
        if self.rendered_pixmap is None:
            self.actual_pixmap_rect = QRect()
            return

        label_w = max(1, self.width())
        label_h = max(1, self.height())
        pix_w = self.rendered_pixmap.width()
        pix_h = self.rendered_pixmap.height()

        # If image is smaller than viewport, keep it centered and disable offset
        # on that axis. If larger, clamp panning so the image cannot disappear.
        if pix_w <= label_w:
            self.pan_offset.setX(0)
            x = (label_w - pix_w) // 2
        else:
            min_x = (label_w - pix_w) // 2
            max_x = (pix_w - label_w) // 2
            self.pan_offset.setX(max(min_x, min(max_x, self.pan_offset.x())))
            x = (label_w - pix_w) // 2 + self.pan_offset.x()

        if pix_h <= label_h:
            self.pan_offset.setY(0)
            y = (label_h - pix_h) // 2
        else:
            min_y = (label_h - pix_h) // 2
            max_y = (pix_h - label_h) // 2
            self.pan_offset.setY(max(min_y, min(max_y, self.pan_offset.y())))
            y = (label_h - pix_h) // 2 + self.pan_offset.y()

        self.actual_pixmap_rect = QRect(x, y, pix_w, pix_h)

    def set_roi_enabled(self, enabled):
        self.roi_enabled = enabled
        if not enabled:
            self.current_rect = QRect()
            self.update()

    def mousePressEvent(self, event):
        if self.rendered_pixmap is None:
            return super().mousePressEvent(event)

        if self.roi_enabled and event.button() == Qt.MouseButton.LeftButton:
            if self.actual_pixmap_rect.contains(event.position().toPoint()):
                self.is_drawing = True
                self.start_point = event.position().toPoint()
                self.end_point = self.start_point
                self.current_rect = QRect(self.start_point, self.end_point)
                self.update()
                event.accept()
                return

        # Pan image: hold left mouse and drag when ROI selection is OFF.
        if (not self.roi_enabled) and event.button() == Qt.MouseButton.LeftButton:
            if self.actual_pixmap_rect.contains(event.position().toPoint()):
                self.is_panning = True
                self.pan_start_mouse = event.position().toPoint()
                self.pan_start_offset = QPoint(self.pan_offset)
                self.setCursor(Qt.CursorShape.ClosedHandCursor)
                event.accept()
                return

        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if self.roi_enabled and self.is_drawing:
            p = event.position().toPoint()
            x = max(self.actual_pixmap_rect.left(), min(p.x(), self.actual_pixmap_rect.right()))
            y = max(self.actual_pixmap_rect.top(), min(p.y(), self.actual_pixmap_rect.bottom()))
            self.end_point = QPoint(x, y)
            self.current_rect = QRect(self.start_point, self.end_point).normalized()
            self.update()
            event.accept()
            return

        if self.is_panning:
            p = event.position().toPoint()
            delta = p - self.pan_start_mouse
            self.pan_offset = self.pan_start_offset + delta
            self.update_pixmap_rect()
            self.update()
            event.accept()
            return

        if (not self.roi_enabled) and self.actual_pixmap_rect.contains(event.position().toPoint()):
            self.setCursor(Qt.CursorShape.OpenHandCursor)
        else:
            self.unsetCursor()
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        if self.roi_enabled and event.button() == Qt.MouseButton.LeftButton and self.is_drawing:
            self.is_drawing = False
            if self.current_rect.width() > 5 and self.current_rect.height() > 5:
                self.calculate_orig_coordinates()
            else:
                self.current_rect = QRect()
                self.main_app.clear_roi_selection(self.main_app.active_roi_id)
            self.update()
            event.accept()
            return

        if self.is_panning and event.button() == Qt.MouseButton.LeftButton:
            self.is_panning = False
            self.setCursor(Qt.CursorShape.OpenHandCursor)
            event.accept()
            return

        super().mouseReleaseEvent(event)

    def wheelEvent(self, event):
        if self.rendered_pixmap is None or self.main_app.orig_image is None:
            return super().wheelEvent(event)
        delta = event.angleDelta().y()
        if delta == 0:
            return
        factor = 1.08 if delta > 0 else 1 / 1.08
        self.main_app.set_image_zoom(self.view_id, factor)
        event.accept()

    def resizeEvent(self, event):
        self.update_pixmap_rect()
        super().resizeEvent(event)

    def paintEvent(self, event):
        super().paintEvent(event)

        if self.rendered_pixmap is None:
            return

        self.update_pixmap_rect()
        painter = QPainter(self)
        painter.drawPixmap(self.actual_pixmap_rect.topLeft(), self.rendered_pixmap)

        if self.main_app.orig_image is None or not self.show_roi_overlay:
            painter.end()
            return

        def draw_stored_roi(roi_box, color, label):
            if roi_box is None or self.actual_pixmap_rect.width() <= 0 or self.actual_pixmap_rect.height() <= 0:
                return

            orig_h, orig_w = self.main_app.orig_image.shape[:2]
            sx = self.actual_pixmap_rect.width() / orig_w
            sy = self.actual_pixmap_rect.height() / orig_h

            x1, y1, x2, y2 = roi_box
            rx = self.actual_pixmap_rect.left() + int(x1 * sx)
            ry = self.actual_pixmap_rect.top() + int(y1 * sy)
            rw = int((x2 - x1) * sx)
            rh = int((y2 - y1) * sy)

            pen = QPen(color, 2, Qt.PenStyle.SolidLine)
            painter.setPen(pen)
            painter.drawRect(QRect(rx, ry, rw, rh))
            painter.drawText(rx + 4, max(ry - 4, 12), label)

        draw_stored_roi(self.main_app.roi_coordinates, QColor(255, 215, 0), "ROI 1")
        draw_stored_roi(self.main_app.roi2_coordinates, QColor(52, 152, 219), "ROI 2")

        if self.roi_enabled and not self.current_rect.isNull():
            if self.main_app.active_roi_id == 1:
                pen = QPen(QColor(255, 215, 0), 2, Qt.PenStyle.DashLine)
            else:
                pen = QPen(QColor(52, 152, 219), 2, Qt.PenStyle.DashLine)
            painter.setPen(pen)
            painter.drawRect(self.current_rect)

        painter.end()

    def calculate_orig_coordinates(self):
        if self.main_app.orig_image is None or self.rendered_pixmap is None:
            return

        orig_h, orig_w = self.main_app.orig_image.shape[:2]
        if self.actual_pixmap_rect.width() <= 0 or self.actual_pixmap_rect.height() <= 0:
            return

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
        self.roi_coordinates = None      # ROI 1: target/tissue region
        self.roi2_coordinates = None     # ROI 2: background/reference region
        self.active_roi_id = 1
        self.auto_roi_boxes = []
        self.current_folder = None
        self.image_paths = []
        self.current_image_index = -1
        self.valid_image_extensions = (".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif")
        self.left_zoom_factor = 1.0
        self.right_zoom_factor = 1.0
        self.zoom_min = 0.25
        self.zoom_max = 4.0

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

        self.btn_open_folder = QPushButton("📁 Select Image Folder")
        self.btn_open_folder.clicked.connect(self.load_folder)
        self.btn_open_folder.setStyleSheet("font-weight: bold; padding: 5px;")

        nav_layout = QHBoxLayout()
        self.btn_prev = QPushButton("⬅ Previous")
        self.btn_prev.clicked.connect(self.show_previous_image)
        self.btn_prev.setEnabled(False)
        self.btn_next = QPushButton("Next ➡")
        self.btn_next.clicked.connect(self.show_next_image)
        self.btn_next.setEnabled(False)
        nav_layout.addWidget(self.btn_prev)
        nav_layout.addWidget(self.btn_next)

        self.lbl_folder_info = QLabel("No folder selected")
        self.lbl_folder_info.setWordWrap(True)
        self.lbl_folder_info.setStyleSheet("color: #7f8c8d;")
        
        self.btn_toggle_roi = QPushButton("🎯 Turn ON ROI 1 Selection")
        self.btn_toggle_roi.setCheckable(True)
        self.btn_toggle_roi.clicked.connect(lambda: self.toggle_roi_mode(1))
        self.btn_toggle_roi.setStyleSheet("background-color: #8e44ad; color: white; padding: 5px;")
        self.btn_toggle_roi.setEnabled(False)

        self.btn_toggle_roi2 = QPushButton("🟦 Turn ON ROI 2 Selection")
        self.btn_toggle_roi2.setCheckable(True)
        self.btn_toggle_roi2.clicked.connect(lambda: self.toggle_roi_mode(2))
        self.btn_toggle_roi2.setStyleSheet("background-color: #2980b9; color: white; padding: 5px;")
        self.btn_toggle_roi2.setEnabled(False)
        
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

        config_layout = QHBoxLayout()
        config_layout.setSpacing(6)
        self.btn_save_config = QPushButton("💾 Save Config")
        self.btn_save_config.clicked.connect(self.save_processing_config)
        self.btn_save_config.setToolTip("Lưu toàn bộ bảng điều chỉnh bên trái thành file .json")
        self.btn_load_config = QPushButton("📂 Load Config")
        self.btn_load_config.clicked.connect(self.load_processing_config)
        self.btn_load_config.setToolTip("Load lại bảng điều chỉnh bên trái từ file .json")
        config_layout.addWidget(self.btn_save_config)
        config_layout.addWidget(self.btn_load_config)
        
        file_layout.addWidget(self.btn_open)
        file_layout.addWidget(self.btn_open_folder)
        file_layout.addLayout(nav_layout)
        file_layout.addWidget(self.lbl_folder_info)
        file_layout.addWidget(self.btn_toggle_roi)
        file_layout.addWidget(self.btn_toggle_roi2)
        file_layout.addWidget(self.btn_save_snapshot)
        file_layout.addWidget(self.btn_reset_left)
        file_layout.addWidget(self.btn_reset_params)
        file_layout.addLayout(config_layout)
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

        self.btn_output_options = QPushButton("📦 Output...")
        self.btn_output_options.clicked.connect(self.open_output_options_dialog)
        self.btn_output_options.setEnabled(False)
        self.btn_output_options.setStyleSheet("background-color: #8e44ad; color: white; font-weight: bold; padding: 5px;")
        
        export_layout.addWidget(self.btn_export_left)
        export_layout.addWidget(self.btn_export_right)
        export_layout.addWidget(self.btn_output_options)
        control_panel.addWidget(export_group)
        
        param_group = QGroupBox("Advanced Pipeline")
        param_layout = QVBoxLayout(param_group)
        param_layout.setSpacing(8)
        
        self.chk_highlight = QCheckBox("🎯 Enable Caliper Mark Processing")
        self.chk_highlight.setStyleSheet("font-weight: bold; color: #e74c3c; margin-bottom: 2px;")
        self.chk_highlight.stateChanged.connect(self.process_and_display)
        param_layout.addWidget(self.chk_highlight)

        param_layout.addWidget(QLabel("Highlight / Edit Option:"))
        self.combo_highlight_mode = QComboBox()
        self.combo_highlight_mode.addItems([
            "1. Highlight Red (Tô đỏ dấu đo)",
            "2. Fast Marching / Telea Inpainting",
            "3. Navier-Stokes Inpainting",
            "4. Local Median Fill",
            "5. Adaptive Local Median Clean (Recommended)",
            "6. Large Kernel Median + Soft Edge",
            "7. Ring Median Color Fill",
            "8. Directional Median Fill",
            "9. Median Core + Seam Repair",
            "10. Two-pass Median Natural Blend"
        ])
        self.combo_highlight_mode.currentIndexChanged.connect(self.process_and_display)
        param_layout.addWidget(self.combo_highlight_mode)

        param_layout.addWidget(QLabel("Mask Shape:"))
        self.combo_mask_shape = QComboBox()
        self.combo_mask_shape.addItems([
            "Shape mask (theo hình dấu đo)",
            "Box mask (hình chữ nhật quanh dấu đo)"
        ])
        self.combo_mask_shape.currentIndexChanged.connect(self.process_and_display)
        param_layout.addWidget(self.combo_mask_shape)

        self.lbl_extra_mask_title = QLabel("Mask Size Expand:")
        param_layout.addWidget(self.lbl_extra_mask_title)
        self.slider_extra_mask = QSlider(Qt.Orientation.Horizontal)
        self.slider_extra_mask.setRange(0, 20)
        self.slider_extra_mask.setValue(0)
        self.slider_extra_mask.valueChanged.connect(self.process_and_display)
        self.lbl_extra_mask_value = QLabel("0 px")
        self.lbl_extra_mask_value.setAlignment(Qt.AlignmentFlag.AlignRight)
        param_layout.addWidget(self.slider_extra_mask)
        param_layout.addWidget(self.lbl_extra_mask_value)
        
        self.lbl_inpaint_radius_title = QLabel("Fill Kernel / Radius:")
        param_layout.addWidget(self.lbl_inpaint_radius_title)
        self.slider_inpaint_radius = QSlider(Qt.Orientation.Horizontal)
        self.slider_inpaint_radius.setRange(1, 12)
        self.slider_inpaint_radius.setValue(3)
        self.slider_inpaint_radius.valueChanged.connect(self.process_and_display)
        self.lbl_inpaint_radius_value = QLabel("3 px")
        self.lbl_inpaint_radius_value.setAlignment(Qt.AlignmentFlag.AlignRight)
        param_layout.addWidget(self.slider_inpaint_radius)
        param_layout.addWidget(self.lbl_inpaint_radius_value)

        self.lbl_edge_feather_title = QLabel("Edge Feather:")
        param_layout.addWidget(self.lbl_edge_feather_title)
        self.slider_edge_feather = QSlider(Qt.Orientation.Horizontal)
        self.slider_edge_feather.setRange(1, 31)
        self.slider_edge_feather.setValue(11)
        self.slider_edge_feather.valueChanged.connect(self.process_and_display)
        self.lbl_edge_feather_value = QLabel("11 px")
        self.lbl_edge_feather_value.setAlignment(Qt.AlignmentFlag.AlignRight)
        param_layout.addWidget(self.slider_edge_feather)
        param_layout.addWidget(self.lbl_edge_feather_value)

        self.lbl_speckle_strength_title = QLabel("Median Replace Strength:")
        param_layout.addWidget(self.lbl_speckle_strength_title)
        self.slider_speckle_strength = QSlider(Qt.Orientation.Horizontal)
        self.slider_speckle_strength.setRange(0, 100)
        self.slider_speckle_strength.setValue(100)
        if hasattr(self, 'slider_context_expand'):
            self.slider_context_expand.setValue(8)
        self.slider_speckle_strength.valueChanged.connect(self.process_and_display)
        self.lbl_speckle_strength_value = QLabel("100 %")
        self.lbl_speckle_strength_value.setAlignment(Qt.AlignmentFlag.AlignRight)
        param_layout.addWidget(self.slider_speckle_strength)
        param_layout.addWidget(self.lbl_speckle_strength_value)

        self.lbl_context_expand_title = QLabel("Seam Repair Width:")
        param_layout.addWidget(self.lbl_context_expand_title)
        self.slider_context_expand = QSlider(Qt.Orientation.Horizontal)
        self.slider_context_expand.setRange(0, 30)
        self.slider_context_expand.setValue(8)
        self.slider_context_expand.valueChanged.connect(self.process_and_display)
        self.lbl_context_expand_value = QLabel("8 px")
        self.lbl_context_expand_value.setAlignment(Qt.AlignmentFlag.AlignRight)
        param_layout.addWidget(self.slider_context_expand)
        param_layout.addWidget(self.lbl_context_expand_value)

        self.chk_auto_roi_border = QCheckBox("🟩 Auto ROI Border (HRAS ENL)")
        self.chk_auto_roi_border.setStyleSheet("font-weight: bold; color: #2ecc71; margin-bottom: 2px;")
        self.chk_auto_roi_border.stateChanged.connect(self.process_and_display)
        param_layout.addWidget(self.chk_auto_roi_border)
        
        self.lbl_thresh_title = QLabel("Caliper Match Threshold:")
        param_layout.addWidget(self.lbl_thresh_title)
        self.slider_detect_thresh = QSlider(Qt.Orientation.Horizontal)
        self.slider_detect_thresh.setRange(40, 90)  
        self.slider_detect_thresh.setValue(62)
        self.combo_mask_shape.setCurrentIndex(0)
        self.slider_extra_mask.setValue(0)      
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
        control_container.setMinimumWidth(0)
        
        scroll_area = QScrollArea()
        scroll_area.setWidget(control_container)
        scroll_area.setWidgetResizable(True) 
        scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff) 
        scroll_area.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)     
        scroll_area.setFixedWidth(390) 
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
        
        self.lbl_orig_view = CustomImageLabel("No image loaded", self, show_roi_overlay=True, view_id="left")
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
        self.lbl_l_sharpness = QLabel("Noise (ENL): N/A")
        self.lbl_l_speckle = QLabel("Sharpness (VoL & Tenengrad): N/A")
        
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
        
        self.lbl_proc_view = CustomImageLabel("No image loaded", self, show_roi_overlay=False, view_id="right")
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
        self.lbl_r_sharpness = QLabel("Noise (ENL & PSNR): N/A")
        self.lbl_r_speckle = QLabel("Sharpness (VoL & Tenengrad): N/A")
        
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
    def toggle_roi_mode(self, roi_id=1):
        self.active_roi_id = roi_id

        if roi_id == 1:
            is_checked = self.btn_toggle_roi.isChecked()
            if is_checked:
                self.btn_toggle_roi2.blockSignals(True)
                self.btn_toggle_roi2.setChecked(False)
                self.btn_toggle_roi2.setText("🟦 Turn ON ROI 2 Selection")
                self.btn_toggle_roi2.setStyleSheet("background-color: #2980b9; color: white; padding: 5px;")
                self.btn_toggle_roi2.blockSignals(False)

                self.btn_toggle_roi.setText("🛑 Turn OFF ROI 1 Selection")
                self.btn_toggle_roi.setStyleSheet("background-color: #d35400; color: white; padding: 5px;")
                self.lbl_orig_view.set_roi_enabled(True)
                self.status_bar.showMessage("ROI 1 Mode: ON. Drag on the left image to select target/tissue region.")
            else:
                self.btn_toggle_roi.setText("🎯 Turn ON ROI 1 Selection")
                self.btn_toggle_roi.setStyleSheet("background-color: #8e44ad; color: white; padding: 5px;")
                self.lbl_orig_view.set_roi_enabled(False)
                self.clear_roi_selection(1)
        else:
            is_checked = self.btn_toggle_roi2.isChecked()
            if is_checked:
                self.btn_toggle_roi.blockSignals(True)
                self.btn_toggle_roi.setChecked(False)
                self.btn_toggle_roi.setText("🎯 Turn ON ROI 1 Selection")
                self.btn_toggle_roi.setStyleSheet("background-color: #8e44ad; color: white; padding: 5px;")
                self.btn_toggle_roi.blockSignals(False)

                self.btn_toggle_roi2.setText("🛑 Turn OFF ROI 2 Selection")
                self.btn_toggle_roi2.setStyleSheet("background-color: #d35400; color: white; padding: 5px;")
                self.lbl_orig_view.set_roi_enabled(True)
                self.status_bar.showMessage("ROI 2 Mode: ON. Drag on the left image to select background/reference region.")
            else:
                self.btn_toggle_roi2.setText("🟦 Turn ON ROI 2 Selection")
                self.btn_toggle_roi2.setStyleSheet("background-color: #2980b9; color: white; padding: 5px;")
                self.lbl_orig_view.set_roi_enabled(False)
                self.clear_roi_selection(2)

    def update_roi_area(self, xmin, ymin, xmax, ymax):
        if self.active_roi_id == 1:
            self.roi_coordinates = [xmin, ymin, xmax, ymax]
            self.status_bar.showMessage(
                f"ROI 1 locked: X[{xmin}->{xmax}], Y[{ymin}->{ymax}]. "
                "HRAS Auto ROI will search only inside ROI 1."
            )
        else:
            self.roi2_coordinates = [xmin, ymin, xmax, ymax]
            self.status_bar.showMessage(
                f"ROI 2 locked: X[{xmin}->{xmax}], Y[{ymin}->{ymax}]. "
                "CNR will use ROI 1 vs ROI 2 when both are available."
            )

        self.lbl_orig_view.update()
        self.process_and_display()

    def clear_roi_selection(self, roi_id=None):
        if roi_id == 1:
            self.roi_coordinates = None
            self.status_bar.showMessage("ROI 1 cleared. HRAS Auto ROI will scan the full image area.")
        elif roi_id == 2:
            self.roi2_coordinates = None
            self.status_bar.showMessage("ROI 2 cleared. CNR will use automatic fallback until ROI 1 and ROI 2 are selected.")
        else:
            self.roi_coordinates = None
            self.roi2_coordinates = None
            self.status_bar.showMessage("ROI 1 and ROI 2 cleared.")

        self.lbl_orig_view.update()
        self.process_and_display()

    # ==============================================================================
    # THUẬT TOÁN TÌM KIẾM THEO TEMPLATE DẤU ĐO
    # ==============================================================================
    def extract_highlight_mask_from_original(self, img_rgb, templates_dir, threshold=0.62, roi_box=None, use_box_mask=False, extra_mask_px=0):
        h_shape, w_shape = img_rgb.shape[:2]
        mask = np.zeros((h_shape, w_shape), dtype=np.uint8)
        boxes = []
        extra_mask_px = max(0, int(extra_mask_px))
        
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
                if use_box_mask:
                    bx1 = max(0, xmin - extra_mask_px)
                    by1 = max(0, ymin - extra_mask_px)
                    bx2 = min(w_shape, xmax + extra_mask_px)
                    by2 = min(h_shape, ymax + extra_mask_px)
                    mask[by1:by2, bx1:bx2] = 255
                else:
                    # Khôi phục logic shape mask giống process_images, nhưng cho phép mask nở
                    # vượt ra ngoài ô template ban đầu bằng cách đưa ROI mask lên full-image
                    # trước rồi mới dilate thêm.
                    tpl_mask_dilated = cv2.dilate(tpl_mask, kernel, iterations=2)
                    final_roi_mask = cv2.bitwise_and(tpl_mask_dilated, dynamic_mask)
                    final_roi_mask = cv2.dilate(final_roi_mask, kernel, iterations=1)

                    tmp_full_mask = np.zeros_like(mask)
                    tmp_full_mask[y_start:y_start+h, x_start:x_start+w] = final_roi_mask
                    if extra_mask_px > 0:
                        extra_kernel_size = 2 * extra_mask_px + 1
                        extra_kernel = np.ones((extra_kernel_size, extra_kernel_size), np.uint8)
                        tmp_full_mask = cv2.dilate(tmp_full_mask, extra_kernel, iterations=1)
                    mask = cv2.bitwise_or(mask, tmp_full_mask)
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
        self.btn_toggle_roi2.setEnabled(enabled)
        self.chk_highlight.setEnabled(enabled)
        self.combo_highlight_mode.setEnabled(enabled)
        self.combo_mask_shape.setEnabled(enabled)
        self.slider_extra_mask.setEnabled(enabled)
        self.slider_inpaint_radius.setEnabled(enabled)
        self.slider_edge_feather.setEnabled(enabled)
        self.slider_speckle_strength.setEnabled(enabled)
        self.chk_auto_roi_border.setEnabled(enabled)
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
        self.btn_output_options.setEnabled(enabled)

    def update_criteria_ui_labels(self, metrics, side):
        """Cập nhật văn bản và màu sắc cảnh báo lên vùng giao diện đã cắt bên dưới"""
        if metrics is None:
            return
            
        if side == "left":
            self.lbl_l_fov.setText(f"Field of View (R): {metrics['fov_r']:.3f} ➔ <b style='color:{metrics['fov_color']};'>{metrics['fov_status']}</b>")
            self.lbl_l_brightness.setText(f"Brightness (SNR): {metrics['snr']:.2f} ➔ <b style='color:{metrics['snr_color']};'>{metrics['snr_status']}</b>")
            self.lbl_l_contrast.setText(f"Contrast (CNR): {metrics['cnr']:.2f} ➔ <b style='color:{metrics['cnr_color']};'>{metrics['cnr_status']}</b>")
            self.lbl_l_sharpness.setText(f"Noise (ENL): {metrics['enl']:.2f} ➔ <b style='color:{metrics['enl_color']};'>{metrics['enl_status']}</b>")
            self.lbl_l_speckle.setText(
                f"Sharpness (VoL): {metrics['vol']:.1f} ➔ "
                f"<b style='color:{metrics['vol_color']};'>{metrics['vol_status']}</b> "
                f"| Tenengrad: {metrics['tenengrad']:.1f}"
            )
        else:
            self.lbl_r_fov.setText(f"Field of View (R): {metrics['fov_r']:.3f} ➔ <b style='color:{metrics['fov_color']};'>{metrics['fov_status']}</b>")
            self.lbl_r_brightness.setText(f"Brightness (SNR): {metrics['snr']:.2f} ➔ <b style='color:{metrics['snr_color']};'>{metrics['snr_status']}</b>")
            self.lbl_r_contrast.setText(f"Contrast (CNR): {metrics['cnr']:.2f} ➔ <b style='color:{metrics['cnr_color']};'>{metrics['cnr_status']}</b>")
            self.lbl_r_sharpness.setText(
                f"Noise (ENL): {metrics['enl']:.2f} ➔ "
                f"<b style='color:{metrics['enl_color']};'>{metrics['enl_status']}</b> "
                f"| <span style='color:{metrics['psnr_color']};'>{metrics['psnr_text']}</span>"
            )
            self.lbl_r_speckle.setText(
                f"Sharpness (VoL): {metrics['vol']:.1f} ➔ "
                f"<b style='color:{metrics['vol_color']};'>{metrics['vol_status']}</b> "
                f"| Tenengrad Post: {metrics['tenengrad']:.1f}"
            )

    def save_to_left_view(self):
        if self.processed_image is not None:
            self.left_view_image = self.processed_image.copy()
            self.display_on_label(self.left_view_image, self.lbl_orig_view)
            
            metrics = analyze_medical_criteria(self.left_view_image, self.orig_image, self.roi_coordinates, self.roi2_coordinates)
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
            
            metrics = analyze_medical_criteria(self.orig_image, None, self.roi_coordinates, self.roi2_coordinates)
            self.update_criteria_ui_labels(metrics, "left")
            self.status_bar.showMessage("Left View reset to Base Image.")

    def refresh_navigation_buttons(self):
        count = len(self.image_paths)
        self.btn_prev.setEnabled(count > 1 and self.current_image_index > 0)
        self.btn_next.setEnabled(count > 1 and self.current_image_index < count - 1)
        if count > 0 and 0 <= self.current_image_index < count:
            self.lbl_folder_info.setText(f"{self.current_image_index + 1}/{count}: {os.path.basename(self.image_paths[self.current_image_index])}")
        else:
            self.lbl_folder_info.setText("No folder selected")

    def find_images_in_folder(self, folder_path):
        found = []
        for root, _, files in os.walk(folder_path):
            for name in files:
                if name.lower().endswith(self.valid_image_extensions):
                    found.append(os.path.join(root, name))
        return sorted(found, key=lambda x: os.path.relpath(x, folder_path).lower())

    def load_folder(self):
        folder_path = QFileDialog.getExistingDirectory(self, "Select Folder Containing Ultrasound Images", "")
        if not folder_path:
            return
        images = self.find_images_in_folder(folder_path)
        if not images:
            QMessageBox.warning(self, "No images", "Không tìm thấy ảnh hợp lệ trong folder đã chọn.")
            return
        self.current_folder = folder_path
        self.image_paths = images
        self.current_image_index = 0
        self.load_image_by_index(reset_params=False, reset_roi=True)

    def load_image_by_index(self, index=None, reset_params=False, reset_roi=False):
        if index is not None:
            self.current_image_index = index
        if not (0 <= self.current_image_index < len(self.image_paths)):
            return
        file_path = self.image_paths[self.current_image_index]
        img = cv2.imread(file_path)
        if img is None:
            self.status_bar.showMessage(f"Error: Could not decode {os.path.basename(file_path)}")
            return

        self.orig_image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.left_view_image = self.orig_image.copy()
        self.auto_roi_boxes = []
        self.highlight_mask = None
        self.all_detected_boxes = []

        if reset_roi:
            self.roi_coordinates = None
            self.roi2_coordinates = None
            self.active_roi_id = 1
            self.btn_toggle_roi.setChecked(False)
            self.btn_toggle_roi.setText("🎯 Turn ON ROI 1 Selection")
            self.btn_toggle_roi.setStyleSheet("background-color: #8e44ad; color: white; padding: 5px;")
            self.btn_toggle_roi2.setChecked(False)
            self.btn_toggle_roi2.setText("🟦 Turn ON ROI 2 Selection")
            self.btn_toggle_roi2.setStyleSheet("background-color: #2980b9; color: white; padding: 5px;")
            self.lbl_orig_view.set_roi_enabled(False)

        self.toggle_controls(True)
        if reset_params:
            self.left_zoom_factor = 1.0
            self.right_zoom_factor = 1.0
            self.reset_sliders()
        else:
            self.process_and_display()

        self.refresh_navigation_buttons()
        self.status_bar.showMessage(f"Loaded: {os.path.basename(file_path)}")

    def show_previous_image(self):
        if self.current_image_index > 0:
            self.load_image_by_index(self.current_image_index - 1, reset_params=False, reset_roi=False)

    def show_next_image(self):
        if self.current_image_index < len(self.image_paths) - 1:
            self.load_image_by_index(self.current_image_index + 1, reset_params=False, reset_roi=False)

    def load_image(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Open Image File", "", "Images (*.png *.jpg *.jpeg *.bmp *.tiff *.tif)")
        if not file_path:
            return
        self.current_folder = os.path.dirname(file_path)
        self.image_paths = [file_path]
        self.current_image_index = 0
        # Giữ nguyên toàn bộ thông số chỉnh sửa bên trái khi mở ảnh mới.
        # Chỉ reset ROI vì tọa độ ROI thường không còn đúng với ảnh khác.
        self.load_image_by_index(reset_params=False, reset_roi=True)

    def draw_auto_roi_borders(self, img_rgb, boxes):
        """
        Draw HRAS-selected automatic ROI borders on an RGB image.
        These boxes show which homogeneous regions are used for automatic ENL.
        """
        if img_rgb is None or not boxes:
            return img_rgb

        out_img = img_rgb.copy()
        for idx, (xmin, ymin, xmax, ymax) in enumerate(boxes, start=1):
            cv2.rectangle(out_img, (xmin, ymin), (xmax, ymax), (46, 204, 113), 2)
            cv2.putText(
                out_img,
                f"ROI {idx}",
                (xmin + 3, max(ymin - 5, 12)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (46, 204, 113),
                1,
                cv2.LINE_AA
            )
        return out_img

    def _odd(self, value, minimum=3):
        value = max(minimum, int(value))
        return value if value % 2 == 1 else value + 1

    def _soft_replace_by_mask(self, img_bgr, replacement_bgr, mask_u8, blur_ksize=11, dilate_px=0):
        """Feather replacement only around the mask edge, avoiding hard square borders."""
        mask_u8 = (mask_u8 > 0).astype(np.uint8) * 255
        if dilate_px > 0:
            kernel = np.ones((3, 3), np.uint8)
            alpha_mask = cv2.dilate(mask_u8, kernel, iterations=int(dilate_px))
        else:
            alpha_mask = mask_u8.copy()

        blur_ksize = self._odd(blur_ksize, 3)
        alpha = cv2.GaussianBlur(alpha_mask, (blur_ksize, blur_ksize), 0).astype(np.float32) / 255.0
        alpha = np.clip(alpha, 0.0, 1.0)[:, :, None]
        edited = img_bgr.astype(np.float32) * (1.0 - alpha) + replacement_bgr.astype(np.float32) * alpha
        return np.clip(edited, 0, 255).astype(np.uint8)

    def _hard_core_soft_edge_replace(self, img_bgr, replacement_bgr, mask_u8, feather_px=9, strength=1.0):
        """Replace the actual mask at full strength, feather only the outside edge.

        This avoids the old problem where Gaussian alpha made thin plus/x strokes
        only partially replaced, leaving a visible ghost of the mark.
        """
        mask_u8 = (mask_u8 > 0).astype(np.uint8) * 255
        strength = max(0.0, min(1.0, float(strength)))
        if strength <= 0.0 or np.count_nonzero(mask_u8) == 0:
            return img_bgr.copy()

        feather_px = max(1, int(feather_px))
        k = self._odd(feather_px, 3)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
        dilated = cv2.dilate(mask_u8, kernel, iterations=1)

        alpha = cv2.GaussianBlur(dilated, (k, k), 0).astype(np.float32) / 255.0
        alpha = np.clip(alpha * strength, 0.0, 1.0)
        # Critical: inside the detected caliper mask, always replace fully enough
        # to remove the +/x footprint instead of blurring it.
        alpha[mask_u8 > 0] = strength
        alpha = alpha[:, :, None]

        edited = img_bgr.astype(np.float32) * (1.0 - alpha) + replacement_bgr.astype(np.float32) * alpha
        return np.clip(edited, 0, 255).astype(np.uint8)

    def _connected_component_fill(self, img_bgr, mask_u8, fill_func):
        """Apply a fill function per connected mark so plus/x arms do not split the region into quadrants."""
        mask_u8 = (mask_u8 > 0).astype(np.uint8) * 255
        num, labels, stats, _ = cv2.connectedComponentsWithStats(mask_u8, connectivity=8)
        result = img_bgr.copy()
        for i in range(1, num):
            x, y, w, h, area = stats[i]
            if area <= 0:
                continue
            comp_mask = (labels == i).astype(np.uint8) * 255
            replacement = fill_func(result, comp_mask, (x, y, w, h))
            result[comp_mask > 0] = replacement[comp_mask > 0]
        return result

    def _ring_median_color_fill(self, img_bgr, mask_u8, ring_px=9, feather=9, strength=1.0):
        """Fill each detected mark with one robust median color from its surrounding ring."""
        def fill_one(base, comp_mask, box):
            x, y, w, h = box
            h_img, w_img = comp_mask.shape
            pad = max(4, int(ring_px))
            x1, y1 = max(0, x - pad), max(0, y - pad)
            x2, y2 = min(w_img, x + w + pad), min(h_img, y + h + pad)
            local_mask = comp_mask[y1:y2, x1:x2]
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * pad + 1, 2 * pad + 1))
            dil = cv2.dilate(local_mask, kernel, iterations=1)
            ring = cv2.subtract(dil, local_mask)
            patch = img_bgr[y1:y2, x1:x2]
            vals = patch[ring > 0]
            repl = img_bgr.copy()
            if vals.size > 0:
                med = np.median(vals.reshape(-1, 3), axis=0).astype(np.uint8)
                repl[comp_mask > 0] = med
            else:
                repl = cv2.medianBlur(img_bgr, self._odd(max(9, ring_px * 2 + 1), 9))
            return repl

        filled = self._connected_component_fill(img_bgr, mask_u8, fill_one)
        return self._hard_core_soft_edge_replace(img_bgr, filled, mask_u8, feather_px=feather, strength=strength)

    def _directional_median_fill(self, img_bgr, mask_u8, kernel_len=15, feather=9, strength=1.0):
        """Use horizontal/vertical/large median and pick the least edge-breaking candidate."""
        k = self._odd(max(7, int(kernel_len)), 7)
        horiz = cv2.medianBlur(cv2.blur(img_bgr, (k, 1)), 3)
        vert = cv2.medianBlur(cv2.blur(img_bgr, (1, k)), 3)
        large = cv2.medianBlur(img_bgr, self._odd(k + 4, 9))
        # Blend directional candidates so the x/plus mark does not divide the fill into 4 visible cells.
        candidate = cv2.addWeighted(horiz, 0.35, vert, 0.35, 0)
        candidate = cv2.addWeighted(candidate, 0.70, large, 0.30, 0)
        return self._hard_core_soft_edge_replace(img_bgr, candidate, mask_u8, feather_px=feather, strength=strength)

    def _adaptive_local_median_clean(self, img_bgr, mask_u8, radius=3, feather=9, strength=1.0, strong=False):
        """Median-based remover tuned for ultrasound caliper marks.

        It uses full-strength replacement inside the shape mask, larger median kernels
        to erase black/white caliper strokes, and only softens the outside seam.
        """
        base_k = self._odd(max(9, radius * 4 + 5), 9)
        large_k = self._odd(max(base_k + 6, 17), 17)
        if strong:
            large_k = self._odd(max(large_k + 8, 25), 25)

        med1 = cv2.medianBlur(img_bgr, base_k)
        med2 = cv2.medianBlur(img_bgr, large_k)
        candidate = cv2.addWeighted(med1, 0.55, med2, 0.45, 0)
        # A tiny bilateral pass removes median blockiness without reintroducing the mark.
        candidate = cv2.bilateralFilter(candidate, d=5, sigmaColor=25, sigmaSpace=25)
        return self._hard_core_soft_edge_replace(img_bgr, candidate, mask_u8, feather_px=feather, strength=strength)

    def _boundary_ring_mask(self, mask_u8, ring_px=9):
        ring_px = max(3, int(ring_px))
        kernel = np.ones((2 * ring_px + 1, 2 * ring_px + 1), np.uint8)
        dilated = cv2.dilate(mask_u8, kernel, iterations=1)
        ring = cv2.subtract(dilated, mask_u8)
        return ring

    def _match_speckle_to_ring(self, original_bgr, edited_bgr, mask_u8, strength=0.35, ring_px=9):
        """Add local ultrasound-like residual texture from the surrounding ring into the edited mask."""
        strength = max(0.0, min(1.0, float(strength)))
        if strength <= 0:
            return edited_bgr

        ring = self._boundary_ring_mask(mask_u8, ring_px=ring_px)
        if np.count_nonzero(ring) < 10:
            return edited_bgr

        gray = cv2.cvtColor(original_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
        smooth = cv2.GaussianBlur(gray, (0, 0), 3.0)
        residual = gray - smooth
        ring_vals = residual[ring > 0]
        if ring_vals.size < 10:
            return edited_bgr

        target_std = float(np.std(ring_vals))
        if target_std < 0.1:
            return edited_bgr

        noise = cv2.GaussianBlur(residual, (0, 0), 1.2)
        noise_std = float(np.std(noise[mask_u8 > 0])) + 1e-6
        noise = noise * (target_std / noise_std) * strength

        feather = self._odd(self.slider_edge_feather.value(), 3) if hasattr(self, 'slider_edge_feather') else 11
        alpha = cv2.GaussianBlur(mask_u8, (feather, feather), 0).astype(np.float32) / 255.0
        alpha = np.clip(alpha, 0.0, 1.0)
        out = edited_bgr.astype(np.float32)
        for ch in range(3):
            out[:, :, ch] += noise * alpha
        return np.clip(out, 0, 255).astype(np.uint8)

    def _texture_clone_from_ring(self, img_bgr, mask_u8, ring_px=12, feather=11, speckle_strength=0.25):
        """Fill masked pixels by sampling visually similar pixels from the surrounding ring."""
        result = img_bgr.copy()
        ring = self._boundary_ring_mask(mask_u8, ring_px=ring_px)
        ring_pixels = img_bgr[ring > 0]
        if ring_pixels.size == 0:
            return cv2.inpaint(img_bgr, mask_u8, 3, cv2.INPAINT_TELEA)

        # Use distance labels to clone nearest valid boundary texture into the mask.
        valid = (mask_u8 == 0).astype(np.uint8)
        _, labels = cv2.distanceTransformWithLabels(valid, cv2.DIST_L2, 5, labelType=cv2.DIST_LABEL_PIXEL)
        h, w = mask_u8.shape
        ys, xs = np.where(valid > 0)
        flat_index_to_coord = {}
        # Build a compact label lookup for nearest non-mask pixels.
        for idx, (yy, xx) in enumerate(zip(ys, xs), start=1):
            flat_index_to_coord[idx] = (yy, xx)

        my, mx = np.where(mask_u8 > 0)
        for yy, xx in zip(my, mx):
            lab = int(labels[yy, xx])
            if lab in flat_index_to_coord:
                sy, sx = flat_index_to_coord[lab]
                result[yy, xx] = img_bgr[sy, sx]

        # Slightly smooth cloned fill, then restore surrounding speckle.
        result = cv2.bilateralFilter(result, d=7, sigmaColor=35, sigmaSpace=35)
        result = self._soft_replace_by_mask(img_bgr, result, mask_u8, blur_ksize=feather, dilate_px=0)
        return self._match_speckle_to_ring(img_bgr, result, mask_u8, strength=speckle_strength, ring_px=ring_px)

    def _frequency_texture_restore(self, img_bgr, base_bgr, mask_u8, feather=11, speckle_strength=0.4):
        """Restore high-frequency ultrasound texture after structure inpainting."""
        low_original = cv2.GaussianBlur(img_bgr, (0, 0), 2.0)
        high = img_bgr.astype(np.float32) - low_original.astype(np.float32)
        # Use surrounding high-frequency pattern, softened to avoid copying obvious shapes.
        high = cv2.GaussianBlur(high, (0, 0), 0.8)
        restored = base_bgr.astype(np.float32) + high * float(speckle_strength)
        restored = np.clip(restored, 0, 255).astype(np.uint8)
        restored = self._soft_replace_by_mask(img_bgr, restored, mask_u8, blur_ksize=feather, dilate_px=0)
        return self._match_speckle_to_ring(img_bgr, restored, mask_u8, strength=speckle_strength, ring_px=10)

    def _edge_aware_bilateral_inpaint(self, img_bgr, mask_u8, radius=3, feather=11, speckle_strength=0.3):
        telea = cv2.inpaint(img_bgr, mask_u8, radius, cv2.INPAINT_TELEA)
        bilateral = cv2.bilateralFilter(telea, d=9, sigmaColor=45, sigmaSpace=45)
        # Preserve Telea structure, use bilateral mainly inside edited mask.
        hybrid = cv2.addWeighted(telea, 0.72, bilateral, 0.28, 0)
        hybrid = self._soft_replace_by_mask(img_bgr, hybrid, mask_u8, blur_ksize=feather, dilate_px=0)
        return self._match_speckle_to_ring(img_bgr, hybrid, mask_u8, strength=speckle_strength, ring_px=9)

    def _expand_mask_px(self, mask_u8, px):
        """Expand a mask for context analysis without changing the final edit region."""
        px = max(0, int(px))
        mask_u8 = (mask_u8 > 0).astype(np.uint8) * 255
        if px <= 0:
            return mask_u8.copy()
        k = 2 * px + 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
        return cv2.dilate(mask_u8, kernel, iterations=1)

    def _color_match_candidate_to_context(self, original_bgr, candidate_bgr, edit_mask_u8, context_px=8, strength=1.0):
        """Match candidate brightness/color to a larger surrounding context but edit only the original mask.

        This targets the visible square/frame problem: the median-filled pixels may be
        clean, but their mean intensity is slightly different from the nearby tissue.
        We estimate the target mean/std from an expanded ring around the mark, then
        remap only the final edit mask. Pixels outside edit_mask_u8 remain unchanged.
        """
        edit_mask_u8 = (edit_mask_u8 > 0).astype(np.uint8) * 255
        if np.count_nonzero(edit_mask_u8) == 0:
            return candidate_bgr.copy()

        context_px = max(1, int(context_px))
        context_mask = self._expand_mask_px(edit_mask_u8, context_px)
        ring = cv2.subtract(context_mask, edit_mask_u8)
        if np.count_nonzero(ring) < 20:
            return candidate_bgr.copy()

        strength = max(0.0, min(1.0, float(strength)))
        out = candidate_bgr.astype(np.float32).copy()
        src = candidate_bgr.astype(np.float32)
        ref = original_bgr.astype(np.float32)

        m = edit_mask_u8 > 0
        r = ring > 0
        for ch in range(3):
            target_mean = float(np.mean(ref[:, :, ch][r]))
            target_std = float(np.std(ref[:, :, ch][r]))
            source_mean = float(np.mean(src[:, :, ch][m]))
            source_std = float(np.std(src[:, :, ch][m]))
            if source_std < 1e-3:
                adjusted = src[:, :, ch][m] - source_mean + target_mean
            else:
                scale = target_std / (source_std + 1e-6)
                # Keep this conservative; too much std matching can create noisy patches.
                scale = max(0.60, min(1.65, scale))
                adjusted = (src[:, :, ch][m] - source_mean) * scale + target_mean
            out[:, :, ch][m] = src[:, :, ch][m] * (1.0 - strength) + adjusted * strength

        return np.clip(out, 0, 255).astype(np.uint8)

    def _context_aware_median_candidate(self, img_bgr, edit_mask_u8, radius=3, context_px=8, strong=False):
        """Build a median candidate using a larger analysis region, but not a larger edit region."""
        context_px = max(0, int(context_px))
        base_k = self._odd(max(9, radius * 4 + 5 + context_px // 2), 9)
        large_k = self._odd(max(base_k + 8 + context_px, 19), 19)
        if strong:
            large_k = self._odd(max(large_k + 8, 27), 27)
        med1 = cv2.medianBlur(img_bgr, base_k)
        med2 = cv2.medianBlur(img_bgr, large_k)
        candidate = cv2.addWeighted(med1, 0.45, med2, 0.55, 0)
        return self._color_match_candidate_to_context(img_bgr, candidate, edit_mask_u8, context_px=max(4, context_px), strength=0.85)

    def apply_caliper_editing_mode(self, img_rgb, mask, mode):
        if mask is None or np.count_nonzero(mask) == 0:
            return img_rgb.copy()
        if mode == 1:
            result = img_rgb.copy()
            result[mask > 0] = [255, 0, 0]
            return result

        mask_u8 = mask.astype(np.uint8)
        mask_u8[mask_u8 > 0] = 255
        img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
        radius = self.slider_inpaint_radius.value() if hasattr(self, 'slider_inpaint_radius') else 3
        feather = self._odd(self.slider_edge_feather.value(), 3) if hasattr(self, 'slider_edge_feather') else 11
        speckle = (self.slider_speckle_strength.value() / 100.0) if hasattr(self, 'slider_speckle_strength') else 0.35
        context_px = self.slider_context_expand.value() if hasattr(self, 'slider_context_expand') else 8
        kernel3 = np.ones((3, 3), np.uint8)

        replace_strength = speckle  # UI label: Median Replace Strength

        if mode == 2:
            edited = cv2.inpaint(img_bgr, mask_u8, radius, cv2.INPAINT_TELEA)
        elif mode == 3:
            edited = cv2.inpaint(img_bgr, mask_u8, radius, cv2.INPAINT_NS)
        elif mode == 4:
            # Original Local Median Fill, but with full-strength core replacement so the mark does not ghost through.
            median_img = cv2.medianBlur(img_bgr, self._odd(max(9, radius * 4 + 3), 9))
            edited = self._hard_core_soft_edge_replace(img_bgr, median_img, mask_u8, feather_px=feather, strength=replace_strength)
        elif mode == 5:
            # Recommended: strongest Local Median style without the 4-quadrant artifact from Telea/speckle methods.
            edited = self._adaptive_local_median_clean(img_bgr, mask_u8, radius=radius, feather=feather, strength=replace_strength, strong=False)
        elif mode == 6:
            # Larger median kernels for very visible black/white caliper strokes.
            edited = self._adaptive_local_median_clean(img_bgr, mask_u8, radius=radius + 2, feather=max(feather, 9), strength=replace_strength, strong=True)
        elif mode == 7:
            # Flat but clean fill from surrounding ring; useful when median still leaves a line footprint.
            edited = self._ring_median_color_fill(img_bgr, mask_u8, ring_px=max(7, radius * 3), feather=feather, strength=replace_strength)
        elif mode == 8:
            # Directional median reduces square/block feeling on long thin marks.
            edited = self._directional_median_fill(img_bgr, mask_u8, kernel_len=max(11, radius * 5), feather=feather, strength=replace_strength)
        elif mode == 9:
            # Reverted away from v8 color/statistics matching because it made the square frame more visible.
            # This keeps the strong median core from v7, then repairs only a very narrow seam around the mask.
            # The actual symbol area is still replaced by median, while Telea touches only the outside ring.
            seam_expand = max(1, min(int(context_px), 4))
            seam_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * seam_expand + 1, 2 * seam_expand + 1))
            strong_mask = cv2.dilate(mask_u8, seam_kernel, iterations=1)
            seam_ring = cv2.subtract(strong_mask, mask_u8)

            median_img = cv2.medianBlur(img_bgr, self._odd(max(13, radius * 5 + 5), 13))
            core = self._hard_core_soft_edge_replace(img_bgr, median_img, mask_u8, feather_px=3, strength=replace_strength)

            # Repair only the transition seam, not the whole filled block.
            telea_edge = cv2.inpaint(core, strong_mask, max(2, radius), cv2.INPAINT_TELEA)
            edited = self._hard_core_soft_edge_replace(core, telea_edge, seam_ring, feather_px=max(3, min(feather, 9)), strength=0.35)
        elif mode == 10:
            # Two-pass clean: first erase with large median, then soften only the seam.
            strong_mask = cv2.dilate(mask_u8, kernel3, iterations=1)
            pass1 = self._adaptive_local_median_clean(img_bgr, strong_mask, radius=radius + 2, feather=3, strength=replace_strength, strong=True)
            smooth = cv2.bilateralFilter(pass1, d=5, sigmaColor=18, sigmaSpace=18)
            edited = self._hard_core_soft_edge_replace(pass1, smooth, strong_mask, feather_px=feather, strength=0.35)
        else:
            edited = img_bgr
        return cv2.cvtColor(edited, cv2.COLOR_BGR2RGB)

    def run_adjustment_pipeline(self, source_rgb, apply_caliper=True):
        """Apply current UI parameters to an RGB image without resetting sliders."""
        thresh_slider_val = self.slider_detect_thresh.value()
        current_threshold = thresh_slider_val / 100.0
        filter_idx = self.combo_filter.currentIndex()
        n_val = self.slider_noise.value()
        b_val = self.slider_brightness.value()
        contrast_idx = self.combo_contrast_method.currentIndex()
        c_val = self.slider_contrast.value()
        sharp_idx = self.combo_sharp_method.currentIndex()
        s_val = self.slider_sharpness.value()
        use_box_mask = self.combo_mask_shape.currentIndex() == 1
        extra_mask_px = self.slider_extra_mask.value()
        mode = self.combo_highlight_mode.currentIndex() + 1
        # Highlight Red should look like the original tool: tight shape mask,
        # no Box mask and no user expansion. The stronger mask controls are
        # reserved for removal/fill modes.
        effective_box_mask = False if mode == 1 else use_box_mask
        effective_extra_mask_px = 0 if mode == 1 else extra_mask_px

        img = source_rgb.copy()
        mask = None
        boxes = []

        if apply_caliper and self.chk_highlight.isChecked():
            mask, boxes = self.extract_highlight_mask_from_original(
                source_rgb, "templates", threshold=current_threshold, roi_box=None,
                use_box_mask=effective_box_mask, extra_mask_px=effective_extra_mask_px
            )
            img = self.apply_caliper_editing_mode(img, mask, mode)

        if n_val > 0:
            if filter_idx == 1:
                k = n_val * 2 + 1
                img = cv2.GaussianBlur(img, (k, k), 0)
            elif filter_idx == 2:
                img = cv2.medianBlur(img, n_val * 2 + 1)
            elif filter_idx == 3:
                img = cv2.bilateralFilter(img, d=9, sigmaColor=n_val * 4, sigmaSpace=n_val * 4)
            elif filter_idx == 4:
                img = apply_srad(img, n_iter=n_val)

        if b_val != 0:
            img = np.clip(img.astype(np.float32) + b_val, 0, 255).astype(np.uint8)

        if contrast_idx == 0:
            alpha = c_val / 100.0
            img = np.clip(img.astype(np.float32) * alpha, 0, 255).astype(np.uint8)
        elif c_val > 0:
            gray_c = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            clahe = cv2.createCLAHE(clipLimit=float(c_val), tileGridSize=(8, 8))
            img = cv2.cvtColor(clahe.apply(gray_c), cv2.COLOR_GRAY2RGB)

        if s_val > 0:
            if sharp_idx == 0:
                blurred = cv2.GaussianBlur(img, (5, 5), 0)
                weight = s_val * 0.3
                img = np.clip(cv2.addWeighted(img, 1.0 + weight, blurred, -weight, 0), 0, 255).astype(np.uint8)
            else:
                kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
                img = cv2.subtract(
                    cv2.add(img, cv2.morphologyEx(img, cv2.MORPH_TOPHAT, kernel)),
                    cv2.morphologyEx(img, cv2.MORPH_BLACKHAT, kernel)
                )
        return img, mask, boxes

    def process_and_display(self):
        if self.orig_image is None:
            return

        current_threshold = self.slider_detect_thresh.value() / 100.0
        self.lbl_thresh_value.setText(f"{current_threshold:.2f}")

        filter_idx = self.combo_filter.currentIndex()
        n_val = self.slider_noise.value()
        b_val = self.slider_brightness.value()
        contrast_idx = self.combo_contrast_method.currentIndex()
        c_val = self.slider_contrast.value()
        s_val = self.slider_sharpness.value()
        use_box_mask = self.combo_mask_shape.currentIndex() == 1
        extra_mask_px = self.slider_extra_mask.value()
        mode = self.combo_highlight_mode.currentIndex() + 1
        effective_box_mask = False if mode == 1 else use_box_mask
        effective_extra_mask_px = 0 if mode == 1 else extra_mask_px

        if filter_idx == 0 or n_val == 0:
            self.lbl_noise_value.setText("Off")
        elif filter_idx == 1 or filter_idx == 2:
            self.lbl_noise_value.setText(f"Kernel: {n_val * 2 + 1}x{n_val * 2 + 1}")
        elif filter_idx == 3:
            self.lbl_noise_value.setText(f"Sigma: {n_val * 4}")
        elif filter_idx == 4:
            self.lbl_noise_value.setText(f"{n_val} Iters")

        self.lbl_brightness.setText(str(b_val))
        self.lbl_extra_mask_value.setText(f"{extra_mask_px} px")
        self.lbl_inpaint_radius_value.setText(f"{self.slider_inpaint_radius.value()} px")
        self.lbl_edge_feather_value.setText(f"{self.slider_edge_feather.value()} px")
        self.lbl_speckle_strength_value.setText(f"{self.slider_speckle_strength.value()} %")
        if hasattr(self, 'lbl_context_expand_value'):
            self.lbl_context_expand_value.setText(f"{self.slider_context_expand.value()} px")
        self.lbl_sharpness.setText(str(s_val))
        if contrast_idx == 0:
            self.lbl_contrast.setText(f"Alpha: {c_val / 100.0:.2f}")
        else:
            self.lbl_contrast.setText("Off" if c_val == 0 else f"Clip: {c_val}.0")

        base_left = self.orig_image.copy()
        if self.chk_highlight.isChecked():
            self.highlight_mask, self.all_detected_boxes = self.extract_highlight_mask_from_original(
                self.orig_image, "templates", threshold=current_threshold, roi_box=self.roi_coordinates,
                use_box_mask=effective_box_mask, extra_mask_px=effective_extra_mask_px
            )
            base_left = self.apply_caliper_editing_mode(base_left, self.highlight_mask, mode)
        else:
            self.highlight_mask = None
            self.all_detected_boxes = []

        self.left_view_image = base_left
        self.processed_image, _, _ = self.run_adjustment_pipeline(self.orig_image, apply_caliper=False)

        if self.chk_highlight.isChecked() and self.highlight_mask is not None:
            self.processed_image = self.apply_caliper_editing_mode(self.processed_image, self.highlight_mask, mode)

        gray_for_hras = cv2.cvtColor(self.processed_image, cv2.COLOR_RGB2GRAY)
        _, self.auto_roi_boxes = compute_enl_hras(gray_for_hras, return_boxes=True, search_roi=self.roi_coordinates)

        metrics_left = analyze_medical_criteria(self.left_view_image, None, self.roi_coordinates, self.roi2_coordinates)
        self.update_criteria_ui_labels(metrics_left, "left")

        metrics_right = analyze_medical_criteria(self.processed_image, self.orig_image, self.roi_coordinates, self.roi2_coordinates)
        self.update_criteria_ui_labels(metrics_right, "right")

        left_display = self.left_view_image
        right_display = self.processed_image
        if self.chk_auto_roi_border.isChecked():
            right_display = self.draw_auto_roi_borders(self.processed_image, self.auto_roi_boxes)

        self.display_on_label(left_display, self.lbl_orig_view)
        self.display_on_label(right_display, self.lbl_proc_view)

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

    def open_output_options_dialog(self):
        if self.orig_image is None:
            return
        dialog = QDialog(self)
        dialog.setWindowTitle("Output Options")
        layout = QVBoxLayout(dialog)

        chk_edit_all = QCheckBox("Edit all img")
        chk_edit_all.setChecked(len(self.image_paths) > 1)
        layout.addWidget(chk_edit_all)

        layout.addWidget(QLabel("Output structure:"))
        combo_structure = QComboBox()
        combo_structure.addItems([
            "Default (processed images only)",
            "Keep folder structure (copy all files, replace images)",
            "New structure (output_images / output_xmls / output_combined)"
        ])
        layout.addWidget(combo_structure)

        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(dialog.accept)
        buttons.rejected.connect(dialog.reject)
        layout.addWidget(buttons)

        if dialog.exec() == QDialog.DialogCode.Accepted:
            self.output_with_options(chk_edit_all.isChecked(), combo_structure.currentIndex())

    def get_output_source_paths(self, edit_all):
        if edit_all and self.image_paths:
            return list(self.image_paths)
        if self.image_paths and 0 <= self.current_image_index < len(self.image_paths):
            return [self.image_paths[self.current_image_index]]
        return []

    def output_with_options(self, edit_all, structure_index):
        source_paths = self.get_output_source_paths(edit_all)
        if not source_paths:
            QMessageBox.warning(self, "No image", "Không có ảnh để output.")
            return

        output_dir = QFileDialog.getExistingDirectory(self, "Select Output Folder", "")
        if not output_dir:
            return

        progress = QProgressDialog("Đang output ảnh...", "Cancel", 0, len(source_paths), self)
        progress.setWindowModality(Qt.WindowModality.WindowModal)
        progress.setMinimumDuration(0)

        if structure_index == 1 and self.current_folder:
            self.copy_full_folder_tree(self.current_folder, output_dir)

        for i, input_path in enumerate(source_paths, start=1):
            progress.setValue(i - 1)
            if progress.wasCanceled():
                break
            img_bgr = cv2.imread(input_path)
            if img_bgr is None:
                continue
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            processed_rgb, _, boxes = self.run_adjustment_pipeline(img_rgb, apply_caliper=True)
            processed_bgr = cv2.cvtColor(processed_rgb, cv2.COLOR_RGB2BGR)
            rel_path = os.path.relpath(input_path, self.current_folder) if self.current_folder else os.path.basename(input_path)
            base_name = os.path.splitext(os.path.basename(input_path))[0]

            if structure_index == 0:
                dst_img = os.path.join(output_dir, os.path.basename(input_path))
                cv2.imwrite(dst_img, processed_bgr)
            elif structure_index == 1:
                dst_img = os.path.join(output_dir, rel_path)
                os.makedirs(os.path.dirname(dst_img), exist_ok=True)
                cv2.imwrite(dst_img, processed_bgr)
            else:
                out_images_dir = os.path.join(output_dir, "output_images")
                out_xmls_dir = os.path.join(output_dir, "output_xmls")
                out_combined_dir = os.path.join(output_dir, "output_combined")
                for d in [out_images_dir, out_xmls_dir, out_combined_dir]:
                    os.makedirs(d, exist_ok=True)

                dst_img = os.path.join(out_images_dir, os.path.basename(input_path))
                cv2.imwrite(dst_img, processed_bgr)
                save_to_combined_xml(
                    os.path.join(out_xmls_dir, f"{base_name}.xml"),
                    os.path.basename(input_path),
                    processed_bgr.shape,
                    boxes,
                    folder_name="output_xmls"
                )

                combined_img = os.path.join(out_combined_dir, os.path.basename(input_path))
                combined_xml = os.path.join(out_combined_dir, f"{base_name}.xml")
                cv2.imwrite(combined_img, processed_bgr)
                save_to_combined_xml(
                    combined_xml,
                    os.path.basename(input_path),
                    processed_bgr.shape,
                    boxes,
                    folder_name="output_combined"
                )

        progress.setValue(len(source_paths))
        self.status_bar.showMessage(f"Output completed: {output_dir}")
        QMessageBox.information(self, "Done", f"Đã output xong vào:\n{output_dir}")

    def copy_full_folder_tree(self, src_folder, dst_folder):
        for root, dirs, files in os.walk(src_folder):
            rel_root = os.path.relpath(root, src_folder)
            target_root = dst_folder if rel_root == "." else os.path.join(dst_folder, rel_root)
            os.makedirs(target_root, exist_ok=True)
            for file_name in files:
                src = os.path.join(root, file_name)
                dst = os.path.join(target_root, file_name)
                try:
                    shutil.copy2(src, dst)
                except shutil.SameFileError:
                    pass

    def get_processing_config(self):
        """Thu thập toàn bộ thông số chỉnh sửa ở panel trái để lưu ra JSON."""
        return {
            "version": 1,
            "caliper_processing_enabled": self.chk_highlight.isChecked(),
            "highlight_mode_index": self.combo_highlight_mode.currentIndex(),
            "mask_shape_index": self.combo_mask_shape.currentIndex(),
            "mask_size_expand": self.slider_extra_mask.value(),
            "fill_kernel_radius": self.slider_inpaint_radius.value(),
            "edge_feather": self.slider_edge_feather.value(),
            "median_replace_strength": self.slider_speckle_strength.value(),
            "seam_repair_width": self.slider_context_expand.value() if hasattr(self, "slider_context_expand") else 8,
            "auto_roi_border": self.chk_auto_roi_border.isChecked(),
            "caliper_match_threshold": self.slider_detect_thresh.value(),
            "filter_method_index": self.combo_filter.currentIndex(),
            "filter_strength": self.slider_noise.value(),
            "brightness": self.slider_brightness.value(),
            "contrast_method_index": self.combo_contrast_method.currentIndex(),
            "contrast_value": self.slider_contrast.value(),
            "sharpness_method_index": self.combo_sharp_method.currentIndex(),
            "sharpness_value": self.slider_sharpness.value(),
        }

    def save_processing_config(self):
        default_name = f"ultrasound_config_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        path, _ = QFileDialog.getSaveFileName(
            self,
            "Save Processing Config",
            default_name,
            "JSON Config (*.json);;All Files (*)"
        )
        if not path:
            return
        if not path.lower().endswith(".json"):
            path += ".json"
        try:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(self.get_processing_config(), f, ensure_ascii=False, indent=4)
            self.status_bar.showMessage(f"Config saved: {path}")
            QMessageBox.information(self, "Saved", f"Đã lưu config vào:\n{path}")
        except Exception as e:
            QMessageBox.critical(self, "Save Config Error", f"Không thể lưu config:\n{e}")

    def load_processing_config(self):
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Load Processing Config",
            "",
            "JSON Config (*.json);;All Files (*)"
        )
        if not path:
            return
        try:
            with open(path, "r", encoding="utf-8") as f:
                config = json.load(f)
            self.apply_processing_config(config)
            self.status_bar.showMessage(f"Config loaded: {path}")
            QMessageBox.information(self, "Loaded", f"Đã load config từ:\n{path}")
        except Exception as e:
            QMessageBox.critical(self, "Load Config Error", f"Không thể load config:\n{e}")

    def apply_processing_config(self, config):
        """Áp dụng config đã lưu, không đụng tới ảnh/folder/ROI hiện tại."""
        widgets = [
            self.chk_highlight, self.combo_highlight_mode, self.combo_mask_shape,
            self.slider_extra_mask, self.slider_inpaint_radius, self.slider_edge_feather,
            self.slider_speckle_strength, self.chk_auto_roi_border, self.slider_detect_thresh,
            self.combo_filter, self.slider_noise, self.slider_brightness,
            self.combo_contrast_method, self.slider_contrast, self.combo_sharp_method,
            self.slider_sharpness
        ]
        if hasattr(self, "slider_context_expand"):
            widgets.append(self.slider_context_expand)

        for w in widgets:
            w.blockSignals(True)

        def clamp(v, lo, hi):
            return max(lo, min(hi, int(v)))

        self.chk_highlight.setChecked(bool(config.get("caliper_processing_enabled", self.chk_highlight.isChecked())))
        self.combo_highlight_mode.setCurrentIndex(clamp(config.get("highlight_mode_index", self.combo_highlight_mode.currentIndex()), 0, self.combo_highlight_mode.count() - 1))
        self.combo_mask_shape.setCurrentIndex(clamp(config.get("mask_shape_index", self.combo_mask_shape.currentIndex()), 0, self.combo_mask_shape.count() - 1))
        self.slider_extra_mask.setValue(clamp(config.get("mask_size_expand", self.slider_extra_mask.value()), self.slider_extra_mask.minimum(), self.slider_extra_mask.maximum()))
        self.slider_inpaint_radius.setValue(clamp(config.get("fill_kernel_radius", self.slider_inpaint_radius.value()), self.slider_inpaint_radius.minimum(), self.slider_inpaint_radius.maximum()))
        self.slider_edge_feather.setValue(clamp(config.get("edge_feather", self.slider_edge_feather.value()), self.slider_edge_feather.minimum(), self.slider_edge_feather.maximum()))
        self.slider_speckle_strength.setValue(clamp(config.get("median_replace_strength", self.slider_speckle_strength.value()), self.slider_speckle_strength.minimum(), self.slider_speckle_strength.maximum()))
        if hasattr(self, "slider_context_expand"):
            self.slider_context_expand.setValue(clamp(config.get("seam_repair_width", self.slider_context_expand.value()), self.slider_context_expand.minimum(), self.slider_context_expand.maximum()))
        self.chk_auto_roi_border.setChecked(bool(config.get("auto_roi_border", self.chk_auto_roi_border.isChecked())))
        self.slider_detect_thresh.setValue(clamp(config.get("caliper_match_threshold", self.slider_detect_thresh.value()), self.slider_detect_thresh.minimum(), self.slider_detect_thresh.maximum()))

        filter_index = clamp(config.get("filter_method_index", self.combo_filter.currentIndex()), 0, self.combo_filter.count() - 1)
        self.combo_filter.setCurrentIndex(filter_index)
        # Cập nhật range của slider noise theo filter đã load, nhưng không reset giá trị.
        if filter_index == 0:
            self.lbl_noise_title.setText("Filter Strength:")
            self.slider_noise.setRange(0, 0)
        elif filter_index in (1, 2):
            self.lbl_noise_title.setText("Kernel Size (Mẫu lẻ):")
            self.slider_noise.setRange(0, 8)
        elif filter_index == 3:
            self.lbl_noise_title.setText("Sigma Color/Space:")
            self.slider_noise.setRange(0, 20)
        else:
            self.lbl_noise_title.setText("SRAD Iterations:")
            self.slider_noise.setRange(0, 40)
        self.slider_noise.setValue(clamp(config.get("filter_strength", self.slider_noise.value()), self.slider_noise.minimum(), self.slider_noise.maximum()))

        self.slider_brightness.setValue(clamp(config.get("brightness", self.slider_brightness.value()), self.slider_brightness.minimum(), self.slider_brightness.maximum()))

        contrast_index = clamp(config.get("contrast_method_index", self.combo_contrast_method.currentIndex()), 0, self.combo_contrast_method.count() - 1)
        self.combo_contrast_method.setCurrentIndex(contrast_index)
        if contrast_index == 0:
            self.slider_contrast.setRange(50, 300)
        else:
            self.slider_contrast.setRange(0, 10)
        self.slider_contrast.setValue(clamp(config.get("contrast_value", self.slider_contrast.value()), self.slider_contrast.minimum(), self.slider_contrast.maximum()))

        self.combo_sharp_method.setCurrentIndex(clamp(config.get("sharpness_method_index", self.combo_sharp_method.currentIndex()), 0, self.combo_sharp_method.count() - 1))
        self.slider_sharpness.setValue(clamp(config.get("sharpness_value", self.slider_sharpness.value()), self.slider_sharpness.minimum(), self.slider_sharpness.maximum()))

        for w in widgets:
            w.blockSignals(False)

        self.process_and_display()

    def reset_sliders(self):
        sliders = [self.slider_detect_thresh, self.slider_extra_mask, self.slider_inpaint_radius, self.slider_edge_feather, self.slider_speckle_strength, self.slider_noise, self.slider_brightness, self.slider_contrast, self.slider_sharpness]
        if hasattr(self, "slider_context_expand"):
            sliders.append(self.slider_context_expand)
        for s in sliders: s.blockSignals(True)
        self.combo_filter.blockSignals(True)
        self.combo_highlight_mode.blockSignals(True)
        self.combo_mask_shape.blockSignals(True)
        self.combo_contrast_method.blockSignals(True)
        self.combo_sharp_method.blockSignals(True)
        
        self.slider_detect_thresh.setValue(62)
        self.combo_mask_shape.setCurrentIndex(0)
        self.slider_extra_mask.setValue(0)
        self.slider_inpaint_radius.setValue(3)
        self.slider_edge_feather.setValue(11)
        self.slider_speckle_strength.setValue(100)
        if hasattr(self, 'slider_context_expand'):
            self.slider_context_expand.setValue(8)
        self.combo_filter.setCurrentIndex(0)
        self.combo_highlight_mode.setCurrentIndex(0)
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
        self.combo_highlight_mode.blockSignals(False)
        self.combo_mask_shape.blockSignals(False)
        self.combo_contrast_method.blockSignals(False)
        self.combo_sharp_method.blockSignals(False)
        self.process_and_display()

    def set_image_zoom(self, view_id, factor):
        """Zoom riêng từng khung ảnh, không làm QLabel/window phình theo pixmap."""
        if view_id == "left":
            self.left_zoom_factor = max(self.zoom_min, min(self.zoom_max, self.left_zoom_factor * float(factor)))
            if self.left_view_image is not None:
                self.display_on_label(self.left_view_image, self.lbl_orig_view)
            self.status_bar.showMessage(f"Left zoom: {self.left_zoom_factor * 100:.0f}%")
        else:
            self.right_zoom_factor = max(self.zoom_min, min(self.zoom_max, self.right_zoom_factor * float(factor)))
            if self.processed_image is not None:
                right_display = self.processed_image
                if self.chk_auto_roi_border.isChecked():
                    right_display = self.draw_auto_roi_borders(self.processed_image, self.auto_roi_boxes)
                self.display_on_label(right_display, self.lbl_proc_view)
            self.status_bar.showMessage(f"Right zoom: {self.right_zoom_factor * 100:.0f}%")

    def eventFilter(self, source, event):
        if event.type() == QEvent.Type.Resize and self.orig_image is not None:
            self.display_on_label(self.left_view_image, self.lbl_orig_view)
            right_display = self.processed_image
            if self.chk_auto_roi_border.isChecked():
                right_display = self.draw_auto_roi_borders(self.processed_image, self.auto_roi_boxes)
            self.display_on_label(right_display, self.lbl_proc_view)
        return super().eventFilter(source, event)

    def display_on_label(self, rgb_array, label_element):
        if rgb_array is None:
            return

        height, width, channel = rgb_array.shape
        bytes_per_line = channel * width
        q_img = QImage(rgb_array.data, width, height, bytes_per_line, QImage.Format.Format_RGB888)

        zoom = 1.0
        if isinstance(label_element, CustomImageLabel):
            zoom = self.left_zoom_factor if label_element.view_id == "left" else self.right_zoom_factor

        # Base size luôn bám theo viewport hiện tại. Zoom chỉ đổi kích thước pixmap
        # bên trong label, không đổi minimum size của cửa sổ.
        viewport_w = max(1, label_element.width() - 4)
        viewport_h = max(1, label_element.height() - 4)
        target_w = max(1, int(viewport_w * zoom))
        target_h = max(1, int(viewport_h * zoom))

        scaled_pixmap = QPixmap.fromImage(q_img).scaled(
            target_w,
            target_h,
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation
        )

        if isinstance(label_element, CustomImageLabel):
            label_element.set_custom_pixmap(scaled_pixmap)
        else:
            label_element.setPixmap(scaled_pixmap)
            label_element.setMinimumSize(120, 120)
            label_element.updateGeometry()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = UltrasoundProcessorApp()
    window.show()
    sys.exit(app.exec())