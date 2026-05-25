@echo off
title Real-ESRGAN Stable Runner

echo ==========================
echo Real-ESRGAN AUTO RUNNER (FIXED)
echo ==========================

:: ==========================
:: 1. FORCE PYTHON 3.10
:: ==========================
echo [1/5] Checking Python 3.10...

where py -3.10 >nul 2>nul
if %errorlevel% neq 0 (
    echo [ERROR] Python 3.10 not found!
    echo Install Python 3.10 and try again.
    pause
    exit /b
)

set PY=py -3.10

:: ==========================
:: 2. DELETE OLD VENV
:: ==========================
echo [2/5] Resetting environment...
if exist venv (
    rmdir /s /q venv
)

%PY% -m venv venv

call venv\Scripts\activate.bat

:: ==========================
:: 3. CORE PIP FIX
:: ==========================
echo [3/5] Upgrading pip tools...
python -m pip install --upgrade pip==23.2 setuptools wheel

:: ==========================
:: 4. INSTALL COMPATIBLE STACK
:: ==========================
echo [4/5] Installing stable ML stack...

:: IMPORTANT: NumPy must be <2
pip install numpy==1.26.4

:: PyTorch stable combo for Python 3.10
pip install torch==2.1.2+cpu torchvision==0.16.2+cpu --index-url https://download.pytorch.org/whl/cpu

:: Core Real-ESRGAN stack (WORKING combo)
pip install basicsr==1.4.2
pip install facexlib==0.3.0
pip install gfpgan==1.3.8
pip install realesrgan==0.3.0

:: utilities
pip install opencv-python tqdm pyyaml scipy

:: ==========================
:: 5. FINAL PATCH FIX
:: ==========================
echo [5/5] Fixing known compatibility issues...

pip uninstall -y numpy
pip install numpy==1.26.4

:: force reinstall torchvision compatibility fix
pip install torchvision==0.16.2+cpu --force-reinstall

:: ==========================
:: MODEL DOWNLOAD + RUN
:: ==========================
echo Downloading model...
if not exist weights mkdir weights

powershell -Command "Invoke-WebRequest https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth -OutFile weights\RealESRGAN_x4plus.pth"

echo Running Real-ESRGAN...
python inference_realesrgan.py -n RealESRGAN_x4plus -i inputs -o results --fp32

echo ==========================
echo DONE
echo ==========================
pause