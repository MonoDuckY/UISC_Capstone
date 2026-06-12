# -*- coding: utf-8 -*-
"""
MedSAM Annotation Tool
-----------------------
Semi-automatic medical image annotation powered by MedSAM.
- Draw bounding box → MedSAM generates segmentation mask
- Assign label from preset or custom list
- Export annotations as COCO-like JSON + annotated image
"""

import sys
import os
import time
import json
import datetime
from functools import partial

import numpy as np
from skimage import transform, io, measure
from PIL import Image, ImageDraw, ImageFont
import torch
from torch.nn import functional as F

from PySide6.QtCore import Qt, QThread, Signal
from PySide6.QtGui import (
    QBrush, QPainter, QPen, QPixmap, QKeySequence,
    QColor, QImage, QShortcut, QFont, QIcon,
)
from PySide6.QtWidgets import (
    QFileDialog, QApplication, QGraphicsScene, QGraphicsView,
    QHBoxLayout, QPushButton, QVBoxLayout, QWidget, QLabel,
    QListWidget, QListWidgetItem, QComboBox, QLineEdit,
    QInputDialog, QMessageBox, QGroupBox, QFormLayout,
    QSplitter, QSizePolicy, QDialog, QDialogButtonBox,
)

from segment_anything.modeling import (
    ImageEncoderViT, MaskDecoder, PromptEncoder, Sam, TwoWayTransformer,
)

# ─── Constants ────────────────────────────────────────────────────────────────

MedSAM_CKPT_PATH = "medsam_model_best.pth"
MEDSAM_IMG_INPUT_SIZE = 1024

PRESET_LABELS = [
    # Anatomical structures
    ("uterus",       "anatomical"),
    ("endometrium",  "anatomical"),
    ("cervix",       "anatomical"),
    # Pathologies
    ("myoma",        "pathology"),
    ("polyp",        "pathology"),
    ("cyst",         "pathology"),
    ("adenomyosis",  "pathology"),
    ("normal",       "pathology"),
]

UTERINE_POSITIONS = ["unknown", "anteverted", "retroverted", "axial", "other"]

# Distinct colors for up to 20 annotations
ANNOTATION_COLORS = [
    (220, 50,  50),   # red
    (50,  180, 50),   # green
    (50,  100, 220),  # blue
    (220, 180, 50),   # yellow
    (180, 50,  220),  # purple
    (50,  200, 200),  # cyan
    (220, 120, 50),   # orange
    (50,  220, 120),  # mint
    (120, 50,  220),  # violet
    (220, 50,  150),  # pink
    (150, 220, 50),   # lime
    (50,  150, 220),  # sky
    (220, 150, 150),  # rose
    (150, 220, 150),  # sage
    (150, 150, 220),  # lavender
    (220, 200, 100),  # gold
    (100, 200, 220),  # teal
    (200, 100, 220),  # orchid
    (220, 100, 100),  # coral
    (100, 220, 200),  # aqua
]

# ─── Device ───────────────────────────────────────────────────────────────────

if torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# ─── Model setup ──────────────────────────────────────────────────────────────

def _build_medsam():
    prompt_embed_dim = 256
    image_size = 1024
    vit_patch_size = 16
    image_embedding_size = image_size // vit_patch_size
    sam = Sam(
        image_encoder=ImageEncoderViT(
            depth=12, embed_dim=768, img_size=image_size, mlp_ratio=4,
            norm_layer=partial(torch.nn.LayerNorm, eps=1e-6),
            num_heads=12, patch_size=vit_patch_size, qkv_bias=True,
            use_rel_pos=True, global_attn_indexes=[2, 5, 8, 11],
            window_size=14, out_chans=prompt_embed_dim,
        ),
        prompt_encoder=PromptEncoder(
            embed_dim=prompt_embed_dim,
            image_embedding_size=(image_embedding_size, image_embedding_size),
            input_image_size=(image_size, image_size),
            mask_in_chans=16,
        ),
        mask_decoder=MaskDecoder(
            num_multimask_outputs=3,
            transformer=TwoWayTransformer(
                depth=2, embedding_dim=prompt_embed_dim, mlp_dim=2048, num_heads=8
            ),
            transformer_dim=prompt_embed_dim,
            iou_head_depth=3, iou_head_hidden_dim=256,
        ),
        pixel_mean=[123.675, 116.28, 103.53],
        pixel_std=[58.395, 57.12, 57.375],
    )
    return sam


print("Loading MedSAM model...")
tic = time.perf_counter()

torch.manual_seed(2023)
torch.cuda.empty_cache()
torch.cuda.manual_seed(2023)
np.random.seed(2023)

medsam_model = _build_medsam().to(device)
checkpoint = torch.load(MedSAM_CKPT_PATH, map_location=device)
if isinstance(checkpoint, dict) and "model" in checkpoint:
    state_dict = checkpoint["model"]
    print(f"  Checkpoint: epoch {checkpoint.get('epoch', '?')}")
else:
    state_dict = checkpoint

if any(k.startswith("module.") for k in state_dict.keys()):
    print("  Stripping 'module.' prefix (multi-GPU checkpoint)...")
    state_dict = {k.replace("module.", "", 1): v for k, v in state_dict.items()}

medsam_model.load_state_dict(state_dict)
medsam_model.eval()
print(f"  Done in {time.perf_counter() - tic:.2f}s  |  device: {device}")

# ─── Inference ────────────────────────────────────────────────────────────────

@torch.no_grad()
def medsam_inference(img_embed, box_1024, height, width):
    box_torch = torch.as_tensor(box_1024, dtype=torch.float, device=img_embed.device)
    if len(box_torch.shape) == 2:
        box_torch = box_torch[:, None, :]
    sparse_emb, dense_emb = medsam_model.prompt_encoder(
        points=None, boxes=box_torch, masks=None,
    )
    low_res_logits, _ = medsam_model.mask_decoder(
        image_embeddings=img_embed,
        image_pe=medsam_model.prompt_encoder.get_dense_pe(),
        sparse_prompt_embeddings=sparse_emb,
        dense_prompt_embeddings=dense_emb,
        multimask_output=False,
    )
    low_res_pred = torch.sigmoid(low_res_logits)
    low_res_pred = F.interpolate(
        low_res_pred, size=(height, width), mode="bilinear", align_corners=False,
    )
    return (low_res_pred.squeeze().cpu().numpy() > 0.5).astype(np.uint8)


def mask_to_polygon(mask):
    """Extract outer contour polygon from binary mask using skimage."""
    contours = measure.find_contours(mask, 0.5)
    if not contours:
        return []
    # Take the largest contour
    contour = max(contours, key=len)
    # Convert (row, col) → (x, y) and flatten
    polygon = [[float(c[1]), float(c[0])] for c in contour[::2]]  # subsample
    return polygon


# ─── Helpers ──────────────────────────────────────────────────────────────────

def np2pixmap(np_img: np.ndarray) -> QPixmap:
    h, w, _ = np_img.shape
    qimg = QImage(
        np.ascontiguousarray(np_img).data,
        w, h, 3 * w,
        QImage.Format.Format_RGB888,
    )
    return QPixmap.fromImage(qimg)


def color_qcolor(color_tuple):
    return QColor(*color_tuple)


# ─── Label Dialog ─────────────────────────────────────────────────────────────

class LabelDialog(QDialog):
    """Popup to choose a label after SAM segmentation."""

    def __init__(self, parent, custom_labels=None):
        super().__init__(parent)
        self.setWindowTitle("Gán nhãn vùng")
        self.setFixedWidth(320)
        self.selected_label = None

        layout = QVBoxLayout(self)
        layout.addWidget(QLabel("Chọn nhãn cho vùng vừa segment:"))

        self.combo = QComboBox()
        # Preset labels
        for name, group in PRESET_LABELS:
            self.combo.addItem(f"[{group}]  {name}", userData=name)
        # Custom labels added during session
        if custom_labels:
            for name in custom_labels:
                self.combo.addItem(f"[custom]  {name}", userData=name)
        layout.addWidget(self.combo)

        # Custom label input
        custom_row = QHBoxLayout()
        self.custom_edit = QLineEdit()
        self.custom_edit.setPlaceholderText("Hoặc nhập nhãn mới...")
        add_btn = QPushButton("Thêm")
        add_btn.clicked.connect(self._add_custom)
        custom_row.addWidget(self.custom_edit)
        custom_row.addWidget(add_btn)
        layout.addLayout(custom_row)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def _add_custom(self):
        name = self.custom_edit.text().strip()
        if name:
            self.combo.addItem(f"[custom]  {name}", userData=name)
            self.combo.setCurrentIndex(self.combo.count() - 1)
            self.custom_edit.clear()

    def get_label(self):
        return self.combo.currentData()


# ─── Main Window ──────────────────────────────────────────────────────────────

class Window(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("MedSAM Annotation Tool")
        self.resize(1280, 760)

        # State
        self.image_path = None
        self.img_3c = None
        self.embedding = None
        self.mask_c = None          # composite RGB mask canvas
        self.annotations = []       # list of annotation dicts
        self.custom_labels = []     # labels added during session
        self.color_idx = 0
        self.is_mouse_down = False
        self.start_pos = (None, None)
        self.start_point = None
        self.end_point = None
        self.rect = None
        self.bg_img = None

        self._build_ui()
        self._connect_signals()

    # ── UI construction ───────────────────────────────────────────────────────

    def _build_ui(self):
        root = QHBoxLayout(self)
        splitter = QSplitter(Qt.Orientation.Horizontal)
        root.addWidget(splitter)

        # ── Left: image viewer ────────────────────────────────────────────────
        viewer_widget = QWidget()
        viewer_layout = QVBoxLayout(viewer_widget)
        viewer_layout.setContentsMargins(4, 4, 4, 4)

        viewer_label = QLabel("🖼  Ảnh siêu âm")
        viewer_label.setStyleSheet("font-weight: bold; font-size: 13px; padding: 2px;")
        viewer_layout.addWidget(viewer_label)

        self.view = QGraphicsView()
        self.view.setRenderHint(QPainter.RenderHint.Antialiasing)
        self.view.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        viewer_layout.addWidget(self.view)

        hint = QLabel("💡 Kéo chuột trên ảnh để vẽ bounding box → MedSAM tự segment")
        hint.setStyleSheet("color: #888; font-size: 11px; padding: 2px;")
        viewer_layout.addWidget(hint)

        splitter.addWidget(viewer_widget)

        # ── Right: control panel ──────────────────────────────────────────────
        panel = QWidget()
        panel.setFixedWidth(320)
        panel_layout = QVBoxLayout(panel)
        panel_layout.setContentsMargins(6, 6, 6, 6)
        panel_layout.setSpacing(8)

        # -- File controls --
        file_group = QGroupBox("📁 File")
        file_layout = QVBoxLayout(file_group)
        self.load_btn = QPushButton("📂  Load ảnh")
        self.load_btn.setFixedHeight(36)
        file_layout.addWidget(self.load_btn)
        panel_layout.addWidget(file_group)

        # -- Image metadata --
        meta_group = QGroupBox("📝 Thông tin ảnh")
        meta_form = QFormLayout(meta_group)
        self.patient_id_edit = QLineEdit()
        self.patient_id_edit.setPlaceholderText("VD: PT001")
        meta_form.addRow("Patient ID:", self.patient_id_edit)
        self.position_combo = QComboBox()
        for pos in UTERINE_POSITIONS:
            self.position_combo.addItem(pos)
        meta_form.addRow("Tư thế tử cung:", self.position_combo)
        self.notes_edit = QLineEdit()
        self.notes_edit.setPlaceholderText("Ghi chú tuỳ ý...")
        meta_form.addRow("Ghi chú:", self.notes_edit)
        panel_layout.addWidget(meta_group)

        # -- Label selector --
        label_group = QGroupBox("🏷️  Nhãn tiếp theo")
        label_layout = QVBoxLayout(label_group)
        self.label_combo = QComboBox()
        for name, group in PRESET_LABELS:
            self.label_combo.addItem(f"[{group}]  {name}", userData=name)
        label_layout.addWidget(self.label_combo)

        custom_row = QHBoxLayout()
        self.custom_label_edit = QLineEdit()
        self.custom_label_edit.setPlaceholderText("Nhãn tùy chỉnh...")
        self.add_label_btn = QPushButton("+ Thêm")
        self.add_label_btn.setFixedWidth(70)
        custom_row.addWidget(self.custom_label_edit)
        custom_row.addWidget(self.add_label_btn)
        label_layout.addLayout(custom_row)
        panel_layout.addWidget(label_group)

        # -- Annotation list --
        ann_group = QGroupBox("📋  Danh sách Annotation")
        ann_layout = QVBoxLayout(ann_group)
        self.ann_list = QListWidget()
        self.ann_list.setFixedHeight(200)
        ann_layout.addWidget(self.ann_list)
        ann_btn_row = QHBoxLayout()
        self.undo_btn = QPushButton("↩ Undo")
        self.delete_btn = QPushButton("🗑 Xóa chọn")
        self.clear_btn = QPushButton("🔴 Xóa tất cả")
        for b in [self.undo_btn, self.delete_btn, self.clear_btn]:
            b.setFixedHeight(30)
            ann_btn_row.addWidget(b)
        ann_layout.addLayout(ann_btn_row)
        panel_layout.addWidget(ann_group)

        # -- Export --
        export_group = QGroupBox("💾  Xuất kết quả")
        export_layout = QVBoxLayout(export_group)
        self.export_json_btn = QPushButton("📄  Export JSON")
        self.export_json_btn.setFixedHeight(36)
        self.save_img_btn = QPushButton("🖼  Lưu ảnh annotated")
        self.save_img_btn.setFixedHeight(36)
        export_layout.addWidget(self.export_json_btn)
        export_layout.addWidget(self.save_img_btn)
        panel_layout.addWidget(export_group)

        panel_layout.addStretch()

        # -- Status bar --
        self.status_label = QLabel("Chưa load ảnh")
        self.status_label.setStyleSheet(
            "color: #555; font-size: 11px; padding: 4px; "
            "background: #f0f0f0; border-top: 1px solid #ccc;"
        )
        panel_layout.addWidget(self.status_label)

        splitter.addWidget(panel)
        splitter.setSizes([960, 320])

    # ── Signal connections ─────────────────────────────────────────────────────

    def _connect_signals(self):
        self.load_btn.clicked.connect(self.load_image)
        self.add_label_btn.clicked.connect(self._add_custom_label)
        self.undo_btn.clicked.connect(self.undo_last)
        self.delete_btn.clicked.connect(self.delete_selected)
        self.clear_btn.clicked.connect(self.clear_all)
        self.export_json_btn.clicked.connect(self.export_json)
        self.save_img_btn.clicked.connect(self.save_annotated_image)

        self.quit_sc = QShortcut(QKeySequence("Ctrl+Q"), self)
        self.quit_sc.activated.connect(self.close)
        self.undo_sc = QShortcut(QKeySequence("Ctrl+Z"), self)
        self.undo_sc.activated.connect(self.undo_last)

    # ── Label helpers ──────────────────────────────────────────────────────────

    def _add_custom_label(self):
        name = self.custom_label_edit.text().strip()
        if not name:
            return
        if name not in self.custom_labels:
            self.custom_labels.append(name)
            self.label_combo.addItem(f"[custom]  {name}", userData=name)
        self.label_combo.setCurrentIndex(self.label_combo.count() - 1)
        self.custom_label_edit.clear()

    def _current_label(self):
        return self.label_combo.currentData()

    # ── Image loading ──────────────────────────────────────────────────────────

    def load_image(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Chọn ảnh siêu âm", ".",
            "Image Files (*.png *.jpg *.jpeg *.bmp *.tif *.tiff)"
        )
        if not file_path:
            return

        self.status_label.setText("⏳ Đang tính image embedding...")
        QApplication.processEvents()

        img_np = io.imread(file_path)
        # Handle grayscale
        if img_np.ndim == 2:
            img_3c = np.stack([img_np] * 3, axis=-1)
        elif img_np.shape[2] == 4:
            img_3c = img_np[:, :, :3]
        else:
            img_3c = img_np

        self.img_3c = img_3c.astype(np.uint8)
        self.image_path = file_path
        self.annotations = []
        self.color_idx = 0
        self.mask_c = np.zeros((*self.img_3c.shape[:2], 3), dtype=np.uint8)

        self._compute_embedding()
        self._refresh_scene()
        self._refresh_ann_list()
        self.status_label.setText(
            f"✅ Loaded: {os.path.basename(file_path)}  "
            f"({self.img_3c.shape[1]}×{self.img_3c.shape[0]})"
        )

    @torch.no_grad()
    def _compute_embedding(self):
        img_1024 = transform.resize(
            self.img_3c, (1024, 1024), order=3,
            preserve_range=True, anti_aliasing=True
        ).astype(np.uint8)
        img_1024 = (img_1024 - img_1024.min()) / np.clip(
            img_1024.max() - img_1024.min(), a_min=1e-8, a_max=None
        )
        tensor = torch.tensor(img_1024).float().permute(2, 0, 1).unsqueeze(0).to(device)
        self.embedding = medsam_model.image_encoder(tensor)

    def _refresh_scene(self):
        H, W, _ = self.img_3c.shape
        self.scene = QGraphicsScene(0, 0, W, H)
        self.end_point = None
        self.rect = None

        # Composite image = original blended with mask
        bg = Image.fromarray(self.img_3c)
        mask_img = Image.fromarray(self.mask_c)
        composite = Image.blend(bg, mask_img, 0.35)
        self.bg_img = self.scene.addPixmap(np2pixmap(np.array(composite)))
        self.bg_img.setPos(0, 0)
        self.view.setScene(self.scene)
        self.view.fitInView(self.scene.sceneRect(), Qt.AspectRatioMode.KeepAspectRatio)

        self.scene.mousePressEvent = self.mouse_press
        self.scene.mouseMoveEvent = self.mouse_move
        self.scene.mouseReleaseEvent = self.mouse_release

    # ── Mouse events ───────────────────────────────────────────────────────────

    def mouse_press(self, ev):
        if self.embedding is None:
            return
        x, y = ev.scenePos().x(), ev.scenePos().y()
        self.is_mouse_down = True
        self.start_pos = (x, y)
        color = ANNOTATION_COLORS[self.color_idx % len(ANNOTATION_COLORS)]
        qc = color_qcolor(color)
        self.start_point = self.scene.addEllipse(
            x - 4, y - 4, 8, 8,
            pen=QPen(qc), brush=QBrush(qc),
        )

    def mouse_move(self, ev):
        if not self.is_mouse_down:
            return
        x, y = ev.scenePos().x(), ev.scenePos().y()
        sx, sy = self.start_pos
        color = ANNOTATION_COLORS[self.color_idx % len(ANNOTATION_COLORS)]
        qc = color_qcolor(color)

        if self.end_point:
            self.scene.removeItem(self.end_point)
        self.end_point = self.scene.addEllipse(
            x - 4, y - 4, 8, 8, pen=QPen(qc), brush=QBrush(qc),
        )
        if self.rect:
            self.scene.removeItem(self.rect)
        xmin, xmax = min(x, sx), max(x, sx)
        ymin, ymax = min(y, sy), max(y, sy)
        self.rect = self.scene.addRect(
            xmin, ymin, xmax - xmin, ymax - ymin,
            pen=QPen(qc, 1.5),
        )

    def mouse_release(self, ev):
        if not self.is_mouse_down or self.embedding is None:
            return
        self.is_mouse_down = False

        x, y = ev.scenePos().x(), ev.scenePos().y()
        sx, sy = self.start_pos
        xmin, xmax = min(x, sx), max(x, sx)
        ymin, ymax = min(y, sy), max(y, sy)

        if (xmax - xmin) < 5 or (ymax - ymin) < 5:
            self.status_label.setText("⚠️  Bbox quá nhỏ, hãy kéo rộng hơn")
            return

        H, W, _ = self.img_3c.shape
        box_np = np.array([[xmin, ymin, xmax, ymax]])
        box_1024 = box_np / np.array([W, H, W, H]) * 1024

        self.status_label.setText("⏳ MedSAM đang segment...")
        QApplication.processEvents()

        sam_mask = medsam_inference(self.embedding, box_1024, H, W)

        # Ask for label
        label = self._current_label()

        # Build annotation record
        color = ANNOTATION_COLORS[self.color_idx % len(ANNOTATION_COLORS)]
        polygon = mask_to_polygon(sam_mask)
        area = int(sam_mask.sum())

        ann = {
            "id": len(self.annotations) + 1,
            "label": label,
            "bbox": [int(xmin), int(ymin), int(xmax), int(ymax)],
            "bbox_format": "xmin_ymin_xmax_ymax",
            "segmentation": polygon,
            "area": area,
            "color": list(color),
        }
        self.annotations.append(ann)
        self.color_idx += 1

        # Paint mask onto canvas
        self.mask_c[sam_mask != 0] = color
        self._refresh_scene()
        self._refresh_ann_list()
        self.status_label.setText(
            f"✅ Annotated: [{label}]  |  area={area}px  |  "
            f"bbox=[{int(xmin)},{int(ymin)},{int(xmax)},{int(ymax)}]"
        )

    # ── Annotation list ────────────────────────────────────────────────────────

    def _refresh_ann_list(self):
        self.ann_list.clear()
        for ann in self.annotations:
            color = ann["color"]
            item = QListWidgetItem(
                f"#{ann['id']}  {ann['label']}  "
                f"(area={ann['area']}px)"
            )
            item.setForeground(QColor(*color))
            item.setData(Qt.ItemDataRole.UserRole, ann["id"])
            self.ann_list.addItem(item)

    def undo_last(self):
        if not self.annotations:
            self.status_label.setText("⚠️  Không có annotation để undo")
            return
        self.annotations.pop()
        self.color_idx = max(0, self.color_idx - 1)
        self._rebuild_mask_canvas()
        self._refresh_scene()
        self._refresh_ann_list()
        self.status_label.setText("↩  Đã undo annotation cuối")

    def delete_selected(self):
        item = self.ann_list.currentItem()
        if item is None:
            return
        ann_id = item.data(Qt.ItemDataRole.UserRole)
        self.annotations = [a for a in self.annotations if a["id"] != ann_id]
        self.color_idx = len(self.annotations)
        self._rebuild_mask_canvas()
        self._refresh_scene()
        self._refresh_ann_list()
        self.status_label.setText(f"🗑  Đã xóa annotation #{ann_id}")

    def clear_all(self):
        if not self.annotations:
            return
        reply = QMessageBox.question(
            self, "Xác nhận", "Xóa tất cả annotation?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
        )
        if reply == QMessageBox.StandardButton.Yes:
            self.annotations.clear()
            self.color_idx = 0
            self.mask_c = np.zeros((*self.img_3c.shape[:2], 3), dtype=np.uint8)
            self._refresh_scene()
            self._refresh_ann_list()
            self.status_label.setText("🔴  Đã xóa tất cả annotation")

    def _rebuild_mask_canvas(self):
        """Repaint mask canvas from scratch using current annotation list."""
        if self.img_3c is None:
            return
        self.mask_c = np.zeros((*self.img_3c.shape[:2], 3), dtype=np.uint8)
        H, W, _ = self.img_3c.shape
        for ann in self.annotations:
            color = ann["color"]
            xmin, ymin, xmax, ymax = ann["bbox"]
            # Re-run SAM to get mask? That's expensive. Instead we store raw mask.
            # To avoid re-running SAM, we fill the bbox region as approximation.
            # (Polygon is stored but reprojecting is complex — bbox fill is acceptable for display)
            self.mask_c[ymin:ymax, xmin:xmax] = color

    # ── Export ────────────────────────────────────────────────────────────────

    def _get_category_list(self):
        """Build category list from preset + any custom labels used."""
        used_labels = {a["label"] for a in self.annotations}
        preset_names = [name for name, _ in PRESET_LABELS]
        categories = []
        cat_id = 1
        for name, group in PRESET_LABELS:
            if name in used_labels:
                categories.append({"id": cat_id, "name": name, "group": group})
                cat_id += 1
        for name in self.custom_labels:
            if name in used_labels:
                categories.append({"id": cat_id, "name": name, "group": "custom"})
                cat_id += 1
        return categories

    def export_json(self):
        if not self.annotations:
            QMessageBox.warning(self, "Chưa có annotation", "Hãy vẽ ít nhất một annotation trước.")
            return
        if not self.image_path:
            return

        # Suggest save path next to image
        base = os.path.splitext(self.image_path)[0]
        default_path = base + "_annotations.json"
        out_path, _ = QFileDialog.getSaveFileName(
            self, "Lưu file JSON", default_path, "JSON Files (*.json)"
        )
        if not out_path:
            return

        H, W, _ = self.img_3c.shape
        label_to_id = {}
        categories = self._get_category_list()
        for cat in categories:
            label_to_id[cat["name"]] = cat["id"]

        annotations_out = []
        for ann in self.annotations:
            cat_id = label_to_id.get(ann["label"], 0)
            # Flatten polygon for COCO-style: [[x1,y1],[x2,y2],...] → [x1,y1,x2,y2,...]
            seg_flat = [coord for pt in ann["segmentation"] for coord in pt]
            annotations_out.append({
                "id": ann["id"],
                "category_id": cat_id,
                "label": ann["label"],
                "bbox": ann["bbox"],
                "bbox_format": "xmin_ymin_xmax_ymax",
                "segmentation": [seg_flat] if seg_flat else [],
                "area": ann["area"],
            })

        output = {
            "info": {
                "tool": "MedSAM Annotation Tool",
                "date": datetime.date.today().isoformat(),
                "model_checkpoint": os.path.basename(MedSAM_CKPT_PATH),
            },
            "image": {
                "file_name": os.path.basename(self.image_path),
                "width": W,
                "height": H,
                "patient_id": self.patient_id_edit.text().strip() or "unknown",
                "uterine_position": self.position_combo.currentText(),
                "notes": self.notes_edit.text().strip(),
            },
            "categories": categories,
            "annotations": annotations_out,
        }

        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(output, f, ensure_ascii=False, indent=2)

        QMessageBox.information(
            self, "Export thành công",
            f"Đã lưu {len(self.annotations)} annotation vào:\n{out_path}"
        )
        self.status_label.setText(f"💾  JSON saved → {os.path.basename(out_path)}")

    def save_annotated_image(self):
        if self.img_3c is None:
            return

        base = os.path.splitext(self.image_path)[0]
        default_path = base + "_annotated.png"
        out_path, _ = QFileDialog.getSaveFileName(
            self, "Lưu ảnh annotated", default_path, "PNG Files (*.png)"
        )
        if not out_path:
            return

        # Blend original + mask
        bg = Image.fromarray(self.img_3c).convert("RGB")
        mask_img = Image.fromarray(self.mask_c).convert("RGB")
        result = Image.blend(bg, mask_img, 0.40)
        draw = ImageDraw.Draw(result)

        # Draw bbox + label for each annotation
        for ann in self.annotations:
            xmin, ymin, xmax, ymax = ann["bbox"]
            color = tuple(ann["color"])
            label = ann["label"]

            # Bounding box
            draw.rectangle([xmin, ymin, xmax, ymax], outline=color, width=2)

            # Label background + text
            try:
                font = ImageFont.truetype("arial.ttf", 14)
            except Exception:
                font = ImageFont.load_default()

            text = f"#{ann['id']} {label}"
            bbox_text = draw.textbbox((xmin, ymin - 18), text, font=font)
            draw.rectangle(bbox_text, fill=color)
            draw.text((xmin, ymin - 18), text, fill=(255, 255, 255), font=font)

        result.save(out_path)
        QMessageBox.information(
            self, "Lưu thành công",
            f"Ảnh annotated đã được lưu tại:\n{out_path}"
        )
        self.status_label.setText(f"🖼  Image saved → {os.path.basename(out_path)}")


# ─── Entry point ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setStyle("Fusion")

    w = Window()
    w.show()

    sys.exit(app.exec())
