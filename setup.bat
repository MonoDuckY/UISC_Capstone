@echo off
title Cai Dat Moi Truong OCR
echo ===================================================
echo   DANG KHOI TAO MOI TRUONG VA CAI DAT THU VIEN
echo ===================================================

:: 1. Tao moi truong ao (Dung 'python' thay vi 'python3' tren Windows)
echo [INFO] Dang tao moi truong ao 'env_ocr'...
python -m venv env_ocr

:: 2. Kich hoat moi truong ao tren Windows
echo [INFO] Dang kich hoat moi truong ao...
call env_ocr\Scripts\activate.bat

:: 3. Nang cap pip va cai dat thu vien qua requirements.txt
echo [INFO] Dang nang cap pip va cai dat cac thu vien can thiet...
python -m pip install --upgrade pip
python -m pip install -r requirements.txt

echo ===================================================
echo   === Da cai dat xong moi truong ao va cac thu vien! ===
echo ===================================================
pause