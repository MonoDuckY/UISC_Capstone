import cv2
import numpy as np
import scipy.ndimage as ndimage

def adaptive_gamma_map_filter(img, ksize, enl):
    """
    Thuật toán Local Adaptive Gamma MAP Filter cho ảnh siêu âm.
    """
    img_float = img.astype(np.float32) / 255.0
    
    mean = ndimage.uniform_filter(img_float, size=ksize)
    mean_sq = ndimage.uniform_filter(img_float**2, size=ksize)
    variance = np.maximum(mean_sq - mean**2, 1e-8)
    
    cu = 1.0 / np.sqrt(enl)
    cmax = np.sqrt(2.0) * cu
    
    ci = np.sqrt(variance) / (mean + 1e-8)
    
    alpha = (1.0 + cu**2) / (ci**2 - cu**2 + 1e-8)
    b = alpha - enl - 1.0
    d = mean**2 * b**2 + 4.0 * alpha * enl * mean * img_float
    f = (b * mean + np.sqrt(np.maximum(d, 0))) / (2.0 * alpha + 1e-8)
    
    result = np.zeros_like(img_float)
    
    mask1 = ci <= cu
    result[mask1] = mean[mask1]
    
    mask2 = (ci > cu) & (ci < cmax)
    result[mask2] = f[mask2]
    
    mask3 = ci >= cmax
    result[mask3] = img_float[mask3]
    
    result = np.clip(result * 255.0, 0, 255).astype(np.uint8)
    return result

def on_trackbar(val):
    pass

# --- HÀM MAIN ---
if __name__ == "__main__":
    image_path = "0267.png" 
    img_origin = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    
    if img_origin is None:
        print(f"Lỗi: Không tìm thấy ảnh '{image_path}'. Vui lòng kiểm tra lại thư mục.")
        exit()

    window_name = "Gamma MAP & CLAHE Tuning - SET490_G77"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 1000, 600)

    # Khởi tạo 3 Trackbars
    cv2.createTrackbar("K_Size (1=3, 2=5...)", window_name, 1, 5, on_trackbar) # Ksize
    cv2.createTrackbar("ENL (x0.1)", window_name, 15, 50, on_trackbar)         # ENL
    cv2.createTrackbar("CLAHE Limit (x0.1)", window_name, 15, 50, on_trackbar) # CLAHE MỚI

    print("Giao diện đã khởi chạy.")
    print("👉 Nhấn phím 'S' để LƯU ảnh hiện tại.")
    print("👉 Nhấn phím 'ESC' để THOÁT.")

    while True:
        # Lấy giá trị từ Trackbar
        k_val = cv2.getTrackbarPos("K_Size (1=3, 2=5...)", window_name)
        enl_val = cv2.getTrackbarPos("ENL (x0.1)", window_name)
        clahe_val = cv2.getTrackbarPos("CLAHE Limit (x0.1)", window_name)
        
        # Xử lý ngoại lệ để tránh lỗi
        if k_val == 0: k_val = 1
        if enl_val == 0: enl_val = 1
        if clahe_val == 0: clahe_val = 1
        
        # Quy đổi giá trị thực tế
        actual_ksize = 2 * k_val + 1
        actual_enl = enl_val / 10.0
        actual_clahe = clahe_val / 10.0
        
        # 1. Chạy thuật toán Gamma MAP
        img_filtered = adaptive_gamma_map_filter(img_origin, actual_ksize, actual_enl)
        
        # 2. Khởi tạo và chạy thuật toán CLAHE động theo thanh trượt
        clahe = cv2.createCLAHE(clipLimit=actual_clahe, tileGridSize=(8, 8))
        img_enhanced = clahe.apply(img_filtered)
        
        # Ghép ảnh gốc và ảnh sau xử lý
        combined_display = np.hstack((img_origin, img_enhanced))
        
        # Cập nhật thông số hiển thị
        text = f"K: {actual_ksize}x{actual_ksize} | ENL: {actual_enl} | CLAHE: {actual_clahe} | 'S' de Luu"
        cv2.putText(combined_display, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        cv2.imshow(window_name, combined_display)
        
        key = cv2.waitKey(100) & 0xFF
        
        if key == 27: 
            break
            
        elif key == ord('s') or key == ord('S'):
            # Tên file tự động lưu trữ cả thông số CLAHE để dễ quản lý
            output_filename = f"filtered_k{actual_ksize}_enl{actual_enl}_clahe{actual_clahe}.png"
            cv2.imwrite(output_filename, img_enhanced)
            print(f"✅ Đã xuất ảnh thành công: {output_filename}")

    cv2.destroyAllWindows()