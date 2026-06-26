import cv2
import pytesseract
import os
import re
import json
import csv
import numpy as np
from datetime import datetime

# Cấu hình Tesseract
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

def preprocess_image(image_path):
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Phóng to để đọc chữ nhỏ
    gray = cv2.resize(gray, None, fx=3, fy=3, interpolation=cv2.INTER_LANCZOS4)
    inverted = cv2.bitwise_not(gray)
    return inverted

def fix_decimal(value_str):
    """Sửa lỗi thiếu dấu chấm thập phân (147 -> 1.47)"""
    val = re.sub(r'[^\d]', '', value_str)
    if len(val) >= 3 and int(val) > 10: return f"{val[0]}.{val[1:]}"
    if len(val) == 2 and int(val) > 5: return f"0.{val}"
    return value_str

def fix_impossible_date(date_str):
    """Sửa lỗi ngày vô lý (34/12 -> 31/12)"""
    match = re.search(r'(\d{1,2})/(\d{1,2})/(\d{4})', date_str)
    if match:
        day, month, year = int(match.group(1)), int(match.group(2)), int(match.group(3))
        if day > 31: day = 31
        if month > 12: month = 12
        return f"{day:02d}/{month:02d}/{year}"
    return date_str

def extract_clinical_data(image_path):
    processed_img = preprocess_image(image_path)
    text = pytesseract.image_to_string(processed_img, config=r'--oem 3 --psm 11')
    
    # Sửa các lỗi đọc chữ/số phổ biến
    text_cleaned = text.replace('wSd', 'w5d').replace('wS d', 'w5d').replace('|', '')
    
    data = {
        "filename": os.path.basename(image_path),
        "timestamp": "",
        "fhr_bpm": "",
        "gestational_age": "",
        "measurements": "",
        "safety_indices": ""
    }

    measure_list = []
    safety_list = []

    # 1. Quét Safety Indices (MI, TIS, TIB)
    safety_matches = re.findall(r'(M[I1]|TIS|TIB|T1S)\s*([\d.]+)', text_cleaned, re.I)
    for label, val in safety_matches:
        fixed_val = fix_decimal(val) if '.' not in val else val
        safety_list.append(f"{label.upper().replace('1','I')}:{fixed_val}")

    # Tách dòng kiểu truyền thống để tương thích Python cũ
    lines = text_cleaned.split('\n')
    
    for line in lines:
        line = line.strip()
        if not line or any(x in line.upper() for x in ["PK ", "BS ", "PHONG KHAM"]): 
            continue

        # 2. Ngày giờ
        dt_match = re.search(r'(\d{2}:\d{2}:\d{2})\s+\w{2}\s+(\d{1,2}/\d{1,2}/\d{4})', line)
        if dt_match:
            time_part = dt_match.group(1)
            date_part = fix_impossible_date(dt_match.group(2))
            data["timestamp"] = f"{date_part} {time_part}"

        # 3. Nhịp tim (FHR)
        fhr_match = re.search(r'FHR\s*[=:-]?\s*(\d+)|(\d+)\s*bpm', line, re.I)
        if fhr_match:
            val = fhr_match.group(1) if fhr_match.group(1) else fhr_match.group(2)
            if val and 60 <= int(val) <= 220: 
                data["fhr_bpm"] = val
            elif val and len(val) > 3: 
                data["fhr_bpm"] = val[:3]

        # 4. Các phép đo (CRL, D1...)
        m_match = re.search(r'(CRL|BPD|FL|AC|D\d+)\s*[=:-]?\s*([\d.]+)\s*mm', line, re.I)
        if m_match:
            measure_list.append(f"{m_match.group(1).upper()}:{m_match.group(2)}mm")

        # 5. Tuổi thai (GA)
        ga_match = re.search(r'(\d+)w([\dS]d)', line, re.I)
        if ga_match:
            week = ga_match.group(1)
            if len(week) > 1 and week[0] == week[1]: week = week[0]
            day = ga_match.group(2).lower().replace('s', '5')
            data["gestational_age"] = f"{week}w{day}"

    data["measurements"] = ", ".join(sorted(list(set(measure_list))))
    data["safety_indices"] = ", ".join(sorted(list(set(safety_list))))
    return data

def main():
    input_folder = 'input'
    output_folder = 'output'
    if not os.path.exists(output_folder): os.mkdir(output_folder)
    
    files = [f for f in os.listdir(input_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    if not files:
        print("Folder input trong!")
        return

    results = []
    for file in files:
        print(f"Processing: {file}...")
        try:
            res = extract_clinical_data(os.path.join(input_folder, file))
            results.append(res)
        except Exception as e:
            print(f"Loi tai file {file}: {e}")

    fieldnames = ["filename", "timestamp", "fhr_bpm", "gestational_age", "measurements", "safety_indices"]
    
    with open(os.path.join(output_folder, 'clinical_report.csv'), 'w', newline='', encoding='utf-8-sig') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)
    
    print("\n--- HOAN THANH! KIEM TRA FILE EXCEL TRONG OUTPUT ---")

if __name__ == "__main__":
    main()