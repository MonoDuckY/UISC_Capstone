@echo off
title Ultrasound Processor

REM 1. Kiem tra Python
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo [LOI] Khong tim thay Python!
    pause
    exit /b
)

REM 2. Tao moi truong ao
if not exist "env\" (
    python -m venv env
)

REM 3. Kich hoat va cai thu vien
call env\Scripts\activate.bat
python -m pip install --upgrade pip
pip install PyQt6 opencv-python numpy numba

REM 4. Chay app
echo Dang khoi chay ung dung...
python ultrasound_processor.py

pause