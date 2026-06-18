@echo off
title Chay Tool Xoa Chu Trong Anh
echo ==========================================
echo    KHỞI CHẠY TOOL NHẬN DIỆN VÀ XÓA CHỮ
echo ==========================================
echo.

REM Kiem tra neu chua chay setup thi canh bao
if not exist "env\Scripts\activate.bat" (
    echo [LỖI] Chua tim thay moi truong 'env'. 
    echo Vui long chay file '1_setup.bat' truoc de cai dat!
    echo.
    pause
    exit
)

echo [*] Dang kich hoat moi truong ao...
call env\Scripts\activate.bat

echo.
echo [*] Dang tien hanh chay code Python...
echo ==========================================
python main.py

echo.
echo ==========================================
echo Chuong trinh da hoan tat! 
echo Vui long kiem tra thu muc 'output_images' va file Excel.
pause