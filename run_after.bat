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
    echo Please run run_first.bat first.
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
set INPUT_PATH="data\raw data"
if "%~1"=="test" (
    set INPUT_PATH="data\raw data\test_0001.png"
) else if not "%~1"=="" (
    set INPUT_PATH=%1
)

echo Running Real-ESRGAN on %INPUT_PATH%...

venv\Scripts\python.exe inference_realesrgan.py -n RealESRGAN_x4plus -i %INPUT_PATH% -o "data\processed data" --suffix processed --fp32

:: ==========================
:: 4. DONE
:: ==========================
echo ==========================
echo DONE - check data\processed data folder
echo ==========================
:: pause