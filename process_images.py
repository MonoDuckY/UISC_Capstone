import cv2
import numpy as np
import os
import xml.etree.ElementTree as ET
from xml.dom import minidom


def prettify_xml(elem):
    """Giúp định dạng file XML có xuống dòng và thụt lề đẹp mắt"""
    rough_string = ET.tostring(elem, 'utf-8')
    reparsed = minidom.parseString(rough_string)
    return reparsed.toprettyxml(indent="\t")


def build_caliper_mask(img, templates_dir, threshold=0.72, use_box_mask=False, extra_mask_px=0):
    """
    Quét đa mẫu để tìm dấu đo và tạo mask nhị phân.

    use_box_mask=False:
        Mask bám theo hình dấu đo đã phát hiện.

    use_box_mask=True:
        Mask là hình chữ nhật bao quanh toàn bộ bounding box của dấu đo.

    extra_mask_px:
        Mở rộng mask thêm N pixel.
        0 = mặc định, không mở rộng thêm ngoài phần dilation có sẵn.
    """
    all_boxes = []
    mask = np.zeros(img.shape[:2], dtype=np.uint8)
    extra_mask_px = max(0, int(extra_mask_px))

    if not os.path.exists(templates_dir):
        return mask, all_boxes

    valid_extensions = (".jpg", ".jpeg", ".png", ".bmp")
    template_files = [f for f in os.listdir(templates_dir) if f.lower().endswith(valid_extensions)]

    if not template_files:
        return mask, all_boxes

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
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
        if tpl is None:
            continue

        h, w = tpl.shape[:2]

        _, tpl_mask = cv2.threshold(tpl, 220, 255, cv2.THRESH_BINARY)
        res = cv2.matchTemplate(gray_filtered, tpl, cv2.TM_CCOEFF_NORMED)
        loc = np.where(res >= threshold)

        for pt in zip(*loc[::-1]):
            x_start, y_start = pt[0], pt[1]

            # Khử nhiễu vùng biên chứa chữ số ở góc màn hình 1024x768
            if x_start < 80 or y_start > 650:
                continue

            if y_start + h > img.shape[0] or x_start + w > img.shape[1]:
                continue

            xmin = max(0, x_start - 2)
            ymin = max(0, y_start - 2)
            xmax = min(img.shape[1], x_start + w + 2)
            ymax = min(img.shape[0], y_start + h + 2)

            is_duplicate = False
            for b in all_boxes:
                if abs(b['xmin'] - xmin) < 8 and abs(b['ymin'] - ymin) < 8:
                    is_duplicate = True
                    break

            if is_duplicate:
                continue

            gray_roi = gray[y_start:y_start + h, x_start:x_start + w]

            _, dynamic_mask = cv2.threshold(gray_roi, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

            kernel = np.ones((3, 3), np.uint8)

            if use_box_mask:
                # Box mask: dùng hình chữ nhật bao quanh toàn bộ bounding box của dấu đo.
                # Cách này xóa mạnh hơn, phù hợp khi mask theo hình dấu đo vẫn còn viền đen.
                bx1 = max(0, xmin - extra_mask_px)
                by1 = max(0, ymin - extra_mask_px)
                bx2 = min(img.shape[1], xmax + extra_mask_px)
                by2 = min(img.shape[0], ymax + extra_mask_px)
                mask[by1:by2, bx1:bx2] = 255
            else:
                # Shape mask: mask bám theo hình dấu đo.
                # Mở rộng template mask để ăn hết phần viền tối/đen còn sót quanh dấu đo.
                # iterations=2 giúp Local Median Fill và Inpainting không chừa lại khung đen 1px.
                tpl_mask_dilated = cv2.dilate(tpl_mask, kernel, iterations=2)
                final_roi_mask = cv2.bitwise_and(tpl_mask_dilated, dynamic_mask)

                # Mở rộng lần cuối thêm 1px để phủ kín biên của dấu đo trước khi fill/xóa.
                final_roi_mask = cv2.dilate(final_roi_mask, kernel, iterations=1)

                # Người dùng có thể mở rộng thêm N pixel. 0 = mặc định.
                if extra_mask_px > 0:
                    extra_kernel_size = 2 * extra_mask_px + 1
                    extra_kernel = np.ones((extra_kernel_size, extra_kernel_size), np.uint8)
                    final_roi_mask = cv2.dilate(final_roi_mask, extra_kernel, iterations=1)

                # Đưa mask cục bộ vào mask toàn ảnh
                mask_roi = mask[y_start:y_start + h, x_start:x_start + w]
                mask[y_start:y_start + h, x_start:x_start + w] = cv2.bitwise_or(mask_roi, final_roi_mask)

            all_boxes.append({
                'name': class_name,
                'xmin': xmin,
                'ymin': ymin,
                'xmax': xmax,
                'ymax': ymax
            })

    return mask, all_boxes


def highlight_by_mask(img, mask):
    """Tô đỏ vùng dấu đo dựa trên mask."""
    result = img.copy()
    result[mask > 0] = [0, 0, 255]
    return result


def inpaint_fast_marching(img, mask, radius=3):
    """
    Xóa dấu đo bằng Fast Marching / Telea Inpainting.
    Đây là lựa chọn nhanh, có sẵn trong OpenCV.
    """
    if mask is None or np.count_nonzero(mask) == 0:
        return img.copy()
    return cv2.inpaint(img, mask, radius, cv2.INPAINT_TELEA)


def inpaint_navier_stokes(img, mask, radius=3):
    """
    Xóa dấu đo bằng Navier-Stokes Inpainting.
    Phù hợp để so sánh với Fast Marching.
    """
    if mask is None or np.count_nonzero(mask) == 0:
        return img.copy()
    return cv2.inpaint(img, mask, radius, cv2.INPAINT_NS)


def fill_with_local_median(img, mask, kernel_size=7):
    """
    Xóa dấu đo bằng cách thay pixel trong mask bằng median cục bộ.
    Đây là baseline đơn giản, không phải AI.
    """
    if mask is None or np.count_nonzero(mask) == 0:
        return img.copy()

    if kernel_size % 2 == 0:
        kernel_size += 1

    median_img = cv2.medianBlur(img, kernel_size)
    result = img.copy()
    result[mask > 0] = median_img[mask > 0]
    return result


def apply_editing_mode(img, mask, mode):
    """
    mode:
    1 = tô đỏ như code cũ
    2 = Fast Marching / Telea
    3 = Navier-Stokes
    4 = Local Median Fill
    """
    if mode == 1:
        return highlight_by_mask(img, mask)
    elif mode == 2:
        return inpaint_fast_marching(img, mask, radius=3)
    elif mode == 3:
        return inpaint_navier_stokes(img, mask, radius=3)
    elif mode == 4:
        return fill_with_local_median(img, mask, kernel_size=9)
    else:
        print("Lựa chọn không hợp lệ, mặc định dùng tô đỏ.")
        return highlight_by_mask(img, mask)


def highlight_and_extract_all_boxes(img, templates_dir, threshold=0.72, mode=1, use_box_mask=False, extra_mask_px=0):
    """
    Hàm wrapper giữ tên cũ để không phá luồng xử lý.

    mode:
    1 = tô đỏ ảnh gốc như bình thường
    2 = xóa dấu đo bằng Fast Marching / Telea
    3 = xóa dấu đo bằng Navier-Stokes
    4 = xóa dấu đo bằng Local Median Fill

    use_box_mask:
        False = mask theo hình dấu đo
        True = mask hình chữ nhật bao quanh bounding box

    extra_mask_px:
        số pixel mở rộng thêm cho mask hoặc bounding box
    """
    mask, all_boxes = build_caliper_mask(
        img,
        templates_dir,
        threshold,
        use_box_mask=use_box_mask,
        extra_mask_px=extra_mask_px
    )
    processed_img = apply_editing_mode(img, mask, mode)
    return processed_img, all_boxes


def save_to_combined_xml(output_path, filename, img_shape, boxes, folder_name="output"):
    """Ghi cấu trúc Pascal VOC XML chứa cả nhãn plus và x_mark chung nhau"""
    annotation = ET.Element('annotation')

    ET.SubElement(annotation, 'folder').text = folder_name
    ET.SubElement(annotation, 'filename').text = filename
    ET.SubElement(annotation, 'path').text = os.path.abspath(output_path)

    source = ET.SubElement(annotation, 'source')
    ET.SubElement(source, 'database').text = "Unknown"

    size = ET.SubElement(annotation, 'size')
    ET.SubElement(size, 'width').text = str(img_shape[1])
    ET.SubElement(size, 'height').text = str(img_shape[0])
    ET.SubElement(size, 'depth').text = str(img_shape[2])

    ET.SubElement(annotation, 'segmented').text = "0"

    for box in boxes:
        obj = ET.SubElement(annotation, 'object')
        ET.SubElement(obj, 'name').text = box['name']
        ET.SubElement(obj, 'pose').text = "Unspecified"
        ET.SubElement(obj, 'truncated').text = "0"
        ET.SubElement(obj, 'difficult').text = "0"

        bndbox = ET.SubElement(obj, 'bndbox')
        ET.SubElement(bndbox, 'xmin').text = str(box['xmin'])
        ET.SubElement(bndbox, 'ymin').text = str(box['ymin'])
        ET.SubElement(bndbox, 'xmax').text = str(box['xmax'])
        ET.SubElement(bndbox, 'ymax').text = str(box['ymax'])

    pretty_xml = prettify_xml(annotation)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(pretty_xml)


def ask_processing_mode():
    print("\nChọn chế độ xử lý dấu đo:")
    print("1. Tô đỏ dấu đo như bình thường")
    print("2. Xóa dấu đo bằng Fast Marching / Telea Inpainting")
    print("3. Xóa dấu đo bằng Navier-Stokes Inpainting")
    print("4. Xóa dấu đo bằng Local Median Fill")
    choice = input("Nhập lựa chọn (1/2/3/4): ").strip()

    try:
        mode = int(choice)
    except ValueError:
        mode = 1

    if mode not in [1, 2, 3, 4]:
        mode = 1

    mode_names = {
        1: "Highlight Red",
        2: "Fast Marching / Telea Inpainting",
        3: "Navier-Stokes Inpainting",
        4: "Local Median Fill"
    }
    print(f"Đang sử dụng chế độ: {mode_names[mode]}\n")
    return mode


def ask_mask_options():
    print("Chọn kiểu mask:")
    print("1. Shape mask: mask theo hình dấu đo hiện tại")
    print("2. Box mask: mask hình chữ nhật bao quanh toàn bộ bounding box")
    mask_choice = input("Nhập lựa chọn mask (1/2): ").strip()

    use_box_mask = mask_choice == "2"

    px_text = input("Nhập số pixel muốn mở rộng mask/box (0 = mặc định): ").strip()
    try:
        extra_mask_px = int(px_text)
    except ValueError:
        extra_mask_px = 0

    if extra_mask_px < 0:
        extra_mask_px = 0

    print(f"Kiểu mask: {'Box mask' if use_box_mask else 'Shape mask'}")
    print(f"Mở rộng thêm: {extra_mask_px}px\n")

    return use_box_mask, extra_mask_px


def main():
    input_dir = "input"
    templates_dir = "templates"

    mode = ask_processing_mode()
    use_box_mask, extra_mask_px = ask_mask_options()

    base_output_dir = "output"

    out_images_dir = os.path.join(base_output_dir, "output_images")
    out_xmls_dir = os.path.join(base_output_dir, "output_xmls")
    out_combined_dir = os.path.join(base_output_dir, "output_combined")

    if not os.path.exists(input_dir):
        os.makedirs(input_dir)
        print(f"Đã tạo thư mục '{input_dir}'. Hãy bỏ ảnh vào đây và chạy lại.")
        return

    for d in [base_output_dir, out_images_dir, out_xmls_dir, out_combined_dir, templates_dir]:
        if not os.path.exists(d):
            os.makedirs(d)

    valid_extensions = (".jpg", ".jpeg", ".png", ".bmp", ".tiff")
    images = [f for f in os.listdir(input_dir) if f.lower().endswith(valid_extensions)]

    if not images:
        print(f"Không tìm thấy ảnh nào trong thư mục '{input_dir}'!")
        return

    print(f"Tìm thấy {len(images)} ảnh. Đang xử lý và xuất vào thư mục tổng 'output'...")

    for filename in images:
        input_path = os.path.join(input_dir, filename)
        base_name = os.path.splitext(filename)[0]

        img = cv2.imread(input_path)
        if img is None:
            print(f"Lỗi: Không thể đọc ảnh {filename}")
            continue

        img_to_process = img.copy()
        processed_img, all_boxes = highlight_and_extract_all_boxes(
            img_to_process,
            templates_dir,
            threshold=0.6,
            mode=mode,
            use_box_mask=use_box_mask,
            extra_mask_px=extra_mask_px
        )

        img_only_path = os.path.join(out_images_dir, filename)
        cv2.imwrite(img_only_path, processed_img)

        xml_only_path = os.path.join(out_xmls_dir, f"{base_name}.xml")
        save_to_combined_xml(xml_only_path, filename, img.shape, all_boxes, folder_name="output_xmls")

        combined_img_path = os.path.join(out_combined_dir, filename)
        combined_xml_path = os.path.join(out_combined_dir, f"{base_name}.xml")

        cv2.imwrite(combined_img_path, processed_img)
        save_to_combined_xml(combined_xml_path, filename, img.shape, all_boxes, folder_name="output_combined")

        print(f"-> Đã xử lý: {filename} ({len(all_boxes)} đối tượng)")

    print(f"\n[THÀNH CÔNG] Toàn bộ dữ liệu đã được gom vào thư mục: '{base_output_dir}/'")


if __name__ == "__main__":
    main()
