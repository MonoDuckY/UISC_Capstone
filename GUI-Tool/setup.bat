@echo off
title Setup Moi Truong Tool Xoa Chu
cd /d "%~dp0"

echo ==========================================
echo    CAI DAT MOI TRUONG VA THU MUC
echo ==========================================
echo.

echo [1/4] Kiem tra va tao thu muc...
if not exist "output_images" (
    mkdir output_images
    echo - Da tao thu muc 'output_images'
)

echo.
if exist "env\Scripts\activate.bat" goto bo_qua_tao_moi

echo [2/4] Dang tao moi truong ao (virtual environment)...
python -m venv env
goto buoc_tiep_theo

:bo_qua_tao_moi
echo [2/4] Moi truong ao 'env' da ton tai, bo qua tao moi.

:buoc_tiep_theo
echo.
echo [*] Dang kich hoat moi truong va nang cap PIP...
call env\Scripts\activate.bat
python -m pip install --upgrade pip

echo.
echo [3/4] Dang kiem tra phan cung de cai dat AI Core (PyTorch)...
REM Kiem tra xem may co card NVIDIA hay khong bang cach bat loi (errorlevel)
nvidia-smi >nul 2>&1
if errorlevel 1 goto cai_dat_cpu

:cai_dat_gpu
echo - Phat hien Card do hoa NVIDIA!
echo - Dang tai va cai dat PyTorch phien ban GPU (Fast-Mode)...
python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
goto buoc_cuoi_cung

:cai_dat_cpu
echo - Khong tim thay Card NVIDIA hoac chua co Driver, he thong se chay bang CPU.
echo - Dang tai va cai dat PyTorch phien ban CPU (Standard-Mode)...
python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

:buoc_cuoi_cung
echo.
echo [4/4] Dang cai dat cac thu vien phu tro tu requirements.txt...
python -m pip install -r requirements.txt

echo.
echo ==========================================
echo SETUP HOAN TAT! 
echo Ban co the chay file '2_run.bat' de su dung tool.
pause