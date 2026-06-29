@echo off
setlocal enabledelayedexpansion
echo ========================================
echo         SETUP TOOL DETECT SIÊU ÂM
echo ========================================

:: 1. Tạo virtual environment
echo [1/5] Tạo virtual environment...
python -m venv venv

:: Kích hoạt venv để cài thư viện vào môi trường tách biệt
call venv\Scripts\activate.bat

:: 2. Kiểm tra GPU hỗ trợ CUDA qua Python cơ bản của hệ thống
echo [2/5] Đang kiểm tra cấu hình phần cứng (GPU CUDA)...

:: Tạo một file python tạm thời để test CUDA
echo import torch > test_cuda.py 2>nul
echo print(torch.cuda.is_available()) >> test_cuda.py 2>nul

:: Cài tạm torch bản cpu/nhẹ để kiểm tra nếu hệ thống chưa có, hoặc dùng luôn python để test lệnh nhanh
python -c "import ctypes; set_cuda = ctypes.windll.nvcuda; print('CUDA_FOUND')" > cuda_check.txt 2>&1

findstr "CUDA_FOUND" cuda_check.txt >nul
if %errorlevel%==0 (
    echo    =^> Tìm thấy NVIDIA GPU! Đang tiến hành cấu hình bản CUDA...
    set HAS_GPU=1
) else (
    echo    =^> Không tìm thấy NVIDIA GPU hoặc driver CUDA chưa cài. Cấu hình bản CPU...
    set HAS_GPU=0
)

:: Xóa file tạm sau khi check xong
del cuda_check.txt >nul 2>&1
del test_cuda.py >nul 2>&1

:: 3. Cài đặt thư viện dựa trên cấu hình máy
echo [3/5] Cài đặt các thư viện lõi (OpenCV, Numpy, Ultralytics)...
pip install opencv-python numpy ultralytics

echo [4/5] Cài đặt EasyOCR và gói PyTorch phù hợp...
if !HAS_GPU!==1 (
    echo    - Đang cài đặt PyTorch với CUDA 12.1 để kích hoạt GPU cho EasyOCR...
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
    echo    - Đang cài đặt EasyOCR...
    pip install easyocr
) else (
    echo    - Đang cài đặt EasyOCR phiên bản CPU tiêu chuẩn...
    pip install easyocr
)

:: 4. Tạo các thư mục làm việc nếu chưa có
echo [5/5] Tạo cấu trúc thư mục dự án...
mkdir inputs outputs models 2>nul

echo.
echo ========================================
echo SETUP HOÀN TẤT!
echo - Chạy 'run.bat' để sử dụng tool
echo ========================================
pause