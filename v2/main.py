import glob
import os

import cv2
import numpy as np
from openpyxl import Workbook
from rapidocr_onnxruntime import RapidOCR

# Cấu hình
INPUT_DIR = 'input'
OUTPUT_DIR = 'output'
VUNG_CAM_DIR = os.path.join(OUTPUT_DIR, 'vung_cam')
DA_XU_LY_DIR = os.path.join(OUTPUT_DIR, 'da_xu_ly')
EXCEL_PATH = os.path.join(DA_XU_LY_DIR, 'ocr_texts.xlsx')

os.makedirs(INPUT_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(VUNG_CAM_DIR, exist_ok=True)
os.makedirs(DA_XU_LY_DIR, exist_ok=True)

OCR = RapidOCR()

FAN_POLYGON_FRACS = [
    (0.48, 0.06),
    (0.38, 0.10),
    (0.27, 0.18),
    (0.15, 0.39),
    (0.13, 0.67),
    (0.20, 0.84),
    (0.50, 0.90),
    (0.80, 0.84),
    (0.87, 0.67),
    (0.85, 0.39),
    (0.73, 0.18),
    (0.62, 0.10),
]

DOPPLER_POLYGON_FRACS = [
    (0.10, 0.40),
    (0.12, 0.36),
    (0.88, 0.36),
    (0.90, 0.40),
    (0.90, 0.91),
    (0.86, 0.94),
    (0.14, 0.94),
    (0.10, 0.91),
]


def scaled_points(width, height, points_frac):
    return np.array(
        [[int(width * x), int(height * y)] for x, y in points_frac],
        dtype=np.int32,
    )


def polygon_mask(height, width, points_frac):
    mask = np.zeros((height, width), dtype=np.uint8)
    points = scaled_points(width, height, points_frac)
    cv2.fillPoly(mask, [points], 255)
    return mask, points


def box_mask(height, width, box_points):
    mask = np.zeros((height, width), dtype=np.uint8)
    cv2.fillPoly(mask, [box_points], 255)
    return mask


def protect_regions(height, width):
    fan_mask, fan_points = polygon_mask(height, width, FAN_POLYGON_FRACS)
    doppler_mask, doppler_points = polygon_mask(height, width, DOPPLER_POLYGON_FRACS)
    protected_mask = cv2.bitwise_or(fan_mask, doppler_mask)
    return protected_mask, fan_points, doppler_points


def remove_text_and_export(image_path):
    image = cv2.imread(image_path)
    if image is None:
        return None, []

    height, width, _ = image.shape
    protected_mask, fan_points, doppler_points = protect_regions(height, width)
    ocr_result, _ = OCR(image_path)
    redacted = image.copy()
    rows = []

    for item in ocr_result or []:
        box, text, score = item
        text = (text or '').strip()
        if not text:
            continue

        box_points = np.array(box, dtype=np.int32)
        box_mask = box_mask_fn(height, width, box_points)
        overlap_mask = cv2.bitwise_and(box_mask, protected_mask)
        touches_protected = int(cv2.countNonZero(overlap_mask) > 0)

        redact_mask = cv2.bitwise_and(box_mask, cv2.bitwise_not(protected_mask))
        if cv2.countNonZero(redact_mask) > 0:
            redacted = cv2.inpaint(redacted, redact_mask, 3, cv2.INPAINT_TELEA)

        x, y, w_cnt, h_cnt = cv2.boundingRect(box_points)
        rows.append([
            os.path.basename(image_path),
            text,
            float(score),
            x,
            y,
            x + w_cnt,
            y + h_cnt,
            'yes' if touches_protected else 'no',
        ])

    output_name = f'cleaned_{os.path.basename(image_path)}'
    output_path = os.path.join(DA_XU_LY_DIR, output_name)
    cv2.imwrite(output_path, redacted)
    return output_path, rows


def save_protected_region_preview(image_path):
    image = cv2.imread(image_path)
    if image is None:
        return None

    height, width, _ = image.shape
    drawing = image.copy()
    fan_polygon = scaled_points(width, height, FAN_POLYGON_FRACS).reshape((-1, 1, 2))
    fan_overlay = drawing.copy()
    cv2.fillPoly(fan_overlay, [fan_polygon], (255, 0, 0))
    cv2.addWeighted(fan_overlay, 0.10, drawing, 0.90, 0, drawing)
    cv2.polylines(drawing, [fan_polygon], True, (255, 0, 0), 4)

    doppler_polygon = scaled_points(width, height, DOPPLER_POLYGON_FRACS).reshape((-1, 1, 2))
    cv2.polylines(drawing, [doppler_polygon], True, (0, 255, 255), 3)

    output_name = f'roi_{os.path.basename(image_path)}'
    output_path = os.path.join(VUNG_CAM_DIR, output_name)
    cv2.imwrite(output_path, drawing)
    return output_path


def box_mask_fn(height, width, box_points):
    mask = np.zeros((height, width), dtype=np.uint8)
    cv2.fillPoly(mask, [box_points], 255)
    return mask


def save_excel(rows):
    # Aggregate detected texts per image: join multiple texts with ' | '
    from collections import defaultdict

    grouped = defaultdict(list)
    for row in rows:
        # row format: [file, text, confidence, x1, y1, x2, y2, touches_protected]
        fname = row[0]
        text = row[1]
        conf = row[2]
        touches = row[7]
        grouped[fname].append((text, conf, touches))

    workbook = Workbook()
    worksheet = workbook.active
    worksheet.title = 'OCR_Texts'
    worksheet.append(['file', 'texts', 'confidences', 'touches_protected'])

    for fname, items in grouped.items():
        texts = ' | '.join([t for t, _, _ in items])
        confs = ' | '.join([str(c) for _, c, _ in items])
        touches_any = 'yes' if any(t == 'yes' for _, _, t in items) else 'no'
        worksheet.append([fname, texts, confs, touches_any])

    workbook.save(EXCEL_PATH)


def verify_outputs(expected_images):
    missing_cleaned = []
    unreadable_cleaned = []
    missing_protected = []
    for image_path in expected_images:
        cleaned_name = f'cleaned_{os.path.basename(image_path)}'
        cleaned_path = os.path.join(DA_XU_LY_DIR, cleaned_name)
        protected_name = f'roi_{os.path.basename(image_path)}'
        protected_path = os.path.join(VUNG_CAM_DIR, protected_name)

        if not os.path.exists(cleaned_path):
            missing_cleaned.append(cleaned_path)
            continue

        if cv2.imread(cleaned_path) is None:
            unreadable_cleaned.append(cleaned_path)

        if not os.path.exists(protected_path):
            missing_protected.append(protected_path)

    excel_exists = os.path.exists(EXCEL_PATH)
    print(f'Processed {len(expected_images)} image(s).')
    print(f'Excel saved: {EXCEL_PATH if excel_exists else "missing"}')
    if missing_cleaned:
        print('Missing processed images:')
        for path in missing_cleaned:
            print(f'- {path}')
    if unreadable_cleaned:
        print('Unreadable processed images:')
        for path in unreadable_cleaned:
            print(f'- {path}')
    if missing_protected:
        print('Missing protected-region previews:')
        for path in missing_protected:
            print(f'- {path}')

    return excel_exists and not missing_cleaned and not unreadable_cleaned and not missing_protected


# Chạy
images = sorted(glob.glob(os.path.join(INPUT_DIR, '*.*')))
all_rows = []

for image_path in images:
    save_protected_region_preview(image_path)
    _, rows = remove_text_and_export(image_path)
    all_rows.extend(rows)

save_excel(all_rows)

if not verify_outputs(images):
    raise SystemExit(1)