@echo off
echo ==========================
echo SwinIR FINAL SETUP (FIXED)
echo ==========================

:: ==========================
:: CREATE VENV WITH PYTHON 3.10
:: ==========================
if not exist venv (
    echo [INFO] Creating virtual environment with Python 3.10...
    py -3.10 -m venv venv
)

call venv\Scripts\activate

echo [2] Upgrade tools...
python -m pip install --upgrade pip setuptools wheel

echo [3] Install PyTorch (stable CPU)...
pip install torch==2.1.2+cpu torchvision==0.16.2+cpu --index-url https://download.pytorch.org/whl/cpu

echo [4] FORCE NumPy 1.x (CRITICAL)...
pip install numpy==1.26.4

echo [5] Install dependencies WITHOUT breaking numpy...
pip install opencv-python --no-deps
pip install tqdm timm einops pyyaml scipy==1.15.3

echo [6] Fix missing SwinIR model file...

IF NOT EXIST models (
    mkdir models
)

:: Download correct file
curl -L -o models\network_swinir.py https://raw.githubusercontent.com/JingyunLiang/SwinIR/main/models/network_swinir.py

echo [7] Done
echo ==========================
echo SETUP COMPLETE (STABLE)
echo ==========================
pause