@echo off
title Real-ESRGAN RUN ONLY

echo ==========================
echo Real-ESRGAN RUN DEMO
echo ==========================

:: ==========================
:: 1. CHECK VENV
:: ==========================
if not exist venv (
    echo [ERROR] venv not found!
    echo Please run setup first (install bat).
    pause
    exit /b
)

:: ==========================
:: 2. ACTIVATE ENV
:: ==========================
call venv\Scripts\activate.bat

:: ==========================
:: 3. RUN INFERENCE
:: ==========================
echo Running Real-ESRGAN...

python inference_realesrgan.py -n RealESRGAN_x4plus -i inputs -o results --fp32

:: ==========================
:: 4. DONE
:: ==========================
echo ==========================
echo DONE - check /results folder
echo ==========================
pause