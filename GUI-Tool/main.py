import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from PIL import Image, ImageTk
import cv2
import easyocr
import pandas as pd
import numpy as np
import os
import re
import threading
import torch

class UltrasoundCleanerApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Tool Nhận Diện & Xóa Chữ Siêu Âm (Hỗ trợ GPU Fast-Mode)")
        self.root.geometry("1400x800")

        self.file_paths = []
        self.reader = None
        
        # --- THIẾT LẬP GIAO DIỆN (UI) TỐI ƯU HÓA LAYOUT ---
        
        # 1. Khung bên trái (Cột điều khiển)
        left_frame = tk.Frame(root, width=280, bg="#f4f4f4")
        left_frame.pack(side=tk.LEFT, fill=tk.Y, padx=5, pady=5)
        left_frame.pack_propagate(False) 

        self.btn_select = tk.Button(left_frame, text="1. Chọn Ảnh (Upload)", font=("Arial", 10, "bold"), bg="#4CAF50", fg="white", command=self.upload_images, height=1)
        self.btn_select.pack(fill=tk.X, pady=(10, 5))

        tk.Label(left_frame, text="Danh sách file:", bg="#f4f4f4", font=("Arial", 9)).pack(anchor=tk.W)
        
        list_frame = tk.Frame(left_frame)
        list_frame.pack(fill=tk.BOTH, expand=True, pady=2)
        scrollbar = tk.Scrollbar(list_frame)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.listbox = tk.Listbox(list_frame, font=("Arial", 9), yscrollcommand=scrollbar.set)
        self.listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.config(command=self.listbox.yview)
        self.listbox.bind('<<ListboxSelect>>', self.on_select_image)

        self.btn_process = tk.Button(left_frame, text="2. Bắt Đầu Xử Lý", font=("Arial", 10, "bold"), bg="#2196F3", fg="white", command=self.start_processing, height=1)
        self.btn_process.pack(fill=tk.X, pady=(5, 5))

        self.progress = ttk.Progressbar(left_frame, orient=tk.HORIZONTAL, mode='determinate')
        self.progress.pack(fill=tk.X, pady=2)

        self.status_label = tk.Label(left_frame, text="Sẵn sàng...", bg="#f4f4f4", fg="#333333", font=("Arial", 9), wraplength=260)
        self.status_label.pack(fill=tk.X, pady=2)

        self.has_gpu = torch.cuda.is_available()
        hardware_text = "NVIDIA GPU (CUDA)" if self.has_gpu else "CPU (Chậm)"
        hardware_color = "#2e7d32" if self.has_gpu else "#c62828"
        tk.Label(left_frame, text=f"Hardware: {hardware_text}", fg=hardware_color, bg="#f4f4f4", font=("Arial", 9, "italic")).pack(pady=(0, 10))

        # 2. Khung bên phải (Khu vực hiển thị ảnh) - CHUYỂN SANG BỐ CỤC TRÁI/PHẢI
        right_frame = tk.Frame(root, bg="white")
        right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=5, pady=5)

        # Cấu hình lưới (Grid): Ép 2 cột chia nhau đúng tỷ lệ 50:50
        right_frame.columnconfigure(0, weight=1, uniform="half")
        right_frame.columnconfigure(1, weight=1, uniform="half")
        right_frame.rowconfigure(1, weight=1)

        # Tiêu đề
        self.lbl_input_title = tk.Label(right_frame, text="ẢNH GỐC (INPUT)", font=("Arial", 10, "bold"), bg="#ffebee", fg="#c62828")
        self.lbl_input_title.grid(row=0, column=0, sticky="ew", padx=(0, 2), pady=(0, 5))
        
        self.lbl_output_title = tk.Label(right_frame, text="ẢNH ĐÃ XÓA CHỮ (OUTPUT)", font=("Arial", 10, "bold"), bg="#e3f2fd", fg="#1565c0")
        self.lbl_output_title.grid(row=0, column=1, sticky="ew", padx=(2, 0), pady=(0, 5))

        # Khung chứa ảnh
        self.panel_input = tk.Label(right_frame, bg="#2c3e50", text="[ Chưa có ảnh ]", fg="white")
        self.panel_input.grid(row=1, column=0, sticky="nsew", padx=(0, 2))

        self.panel_output = tk.Label(right_frame, bg="#2c3e50", text="[ Chưa xử lý ]", fg="white")
        self.panel_output.grid(row=1, column=1, sticky="nsew", padx=(2, 0))

    def upload_images(self):
        files = filedialog.askopenfilenames(title="Chọn ảnh siêu âm", filetypes=[("Image Files", "*.jpg *.jpeg *.png")])
        if files:
            self.file_paths = list(files)
            self.listbox.delete(0, tk.END)
            for f in self.file_paths:
                self.listbox.insert(tk.END, os.path.basename(f))
            
            self.status_label.config(text=f"Đã tải {len(self.file_paths)} ảnh.")
            self.listbox.selection_set(0)
            self.on_select_image(None)

    def on_select_image(self, event):
        selection = self.listbox.curselection()
        if not selection: return
        idx = selection[0]
        input_path = self.file_paths[idx]
        filename = os.path.basename(input_path)
        output_path = os.path.join("output_images", f"cleaned_{filename}")

        self.root.after(50, lambda: self.display_image(input_path, self.panel_input))
        
        if os.path.exists(output_path):
            self.root.after(50, lambda: self.display_image(output_path, self.panel_output))
        else:
            self.panel_output.config(image='', text="[ Chưa xử lý ]")

    def display_image(self, path, panel):
        try:
            img = Image.open(path)
            
            self.root.update_idletasks() 
            panel_width = panel.winfo_width()
            panel_height = panel.winfo_height()
            
            if panel_width < 10 or panel_height < 10:
                panel_width, panel_height = 500, 500 
                
            img.thumbnail((panel_width, panel_height), Image.Resampling.LANCZOS)
            img_tk = ImageTk.PhotoImage(img)
            
            panel.config(image=img_tk, text="")
            panel.image = img_tk 
        except Exception as e:
            panel.config(text=f"Lỗi hiển thị: {e}", image='')

    def start_processing(self):
        if not self.file_paths:
            messagebox.showwarning("Cảnh báo", "Upload ảnh trước!")
            return
        
        self.btn_process.config(state=tk.DISABLED, bg="gray")
        self.btn_select.config(state=tk.DISABLED, bg="gray")
        
        self.progress['value'] = 0
        self.progress['maximum'] = len(self.file_paths)
        
        threading.Thread(target=self.process_images_thread, daemon=True).start()

    def process_images_thread(self):
        device_name = "GPU" if self.has_gpu else "CPU"
        self.root.after(0, self.status_label.config, {'text': f"Đang nạp ({device_name})..."})
        
        if self.reader is None:
            self.reader = easyocr.Reader(['vi', 'en'], gpu=self.has_gpu)

        output_dir = 'output_images'
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        all_text_data = []

        for i, img_path in enumerate(self.file_paths):
            filename = os.path.basename(img_path)
            self.root.after(0, self.status_label.config, {'text': f"Đang xử lý: {filename} ({i+1}/{len(self.file_paths)})"})
            
            img = cv2.imread(img_path)
            if img is None: 
                self.update_progress(i + 1)
                continue
            
            img_h, img_w = img.shape[:2]
            
            img_ocr = cv2.resize(img, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
            results = self.reader.readtext(img_ocr, text_threshold=0.05, low_text=0.05)
            
            mask = np.zeros((img_h, img_w), dtype=np.uint8)
            detected_texts = []
            
            for (bbox, text, prob) in results:
                tl = (np.min(bbox, axis=0) / 2).astype(int)
                br = (np.max(bbox, axis=0) / 2).astype(int)
                
                clean_text = re.sub(r'[^\w\s\.,:\-\/%]', '', text).strip()
                if clean_text and len(clean_text) >= 2: 
                    detected_texts.append(clean_text)
                
                center_x = (tl[0] + br[0]) / 2
                center_y = (tl[1] + br[1]) / 2
                
                if (0.18 * img_w < center_x < 0.85 * img_w) and (0.12 * img_h < center_y < 0.90 * img_h):
                    continue 
                
                pad = 8 
                x_min = max(0, tl[0] - pad)
                y_min = max(0, tl[1] - pad)
                x_max = min(img.shape[1], br[0] + pad)
                y_max = min(img.shape[0], br[1] + pad)
                
                roi = img[y_min:y_max, x_min:x_max]
                roi_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
                
                _, roi_thresh = cv2.threshold(roi_gray, 80, 255, cv2.THRESH_BINARY)
                mask[y_min:y_max, x_min:x_max] = cv2.bitwise_or(mask[y_min:y_max, x_min:x_max], roi_thresh)
                
            kernel = np.ones((5, 5), np.uint8)
            mask = cv2.dilate(mask, kernel, iterations=2)
            
            cleaned_img = cv2.inpaint(img, mask, 5, cv2.INPAINT_TELEA)
            
            all_text_data.append({
                'Tên File': filename,
                'Nội dung text': ' | '.join(detected_texts)
            })
            
            out_path = os.path.join(output_dir, f"cleaned_{filename}")
            cv2.imwrite(out_path, cleaned_img)
            
            self.update_progress(i + 1)
            self.root.after(0, self.listbox.selection_clear, 0, tk.END)
            self.root.after(0, self.listbox.selection_set, i)
            self.root.after(0, self.on_select_image, None)

        if all_text_data:
            self.root.after(0, self.status_label.config, {'text': "Đang lưu Excel..."})
            excel_path = 'KetQua_NhanDien.xlsx'
            pd.DataFrame(all_text_data).to_excel(excel_path, index=False)
        
        self.root.after(0, self.finish_processing)

    def update_progress(self, val):
        def update():
            self.progress['value'] = val
        self.root.after(0, update)

    def finish_processing(self):
        self.status_label.config(text="Hoàn thành!")
        self.btn_process.config(state=tk.NORMAL, bg="#2196F3")
        self.btn_select.config(state=tk.NORMAL, bg="#4CAF50")
        messagebox.showinfo("Thành công", "Đã xử lý xong!")

if __name__ == "__main__":
    root = tk.Tk()
    app = UltrasoundCleanerApp(root)
    root.mainloop()