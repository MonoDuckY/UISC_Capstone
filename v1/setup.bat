@echo off
echo [1/3] Dang tao moi truong ao (venv)...
python -m venv venv

echo [2/3] Dang kich hoat moi truong ao va cai thu vien...
call venv\Scripts\activate
pip install pytesseract opencv-python pillow

echo [3/3] Tao thu muc dau vao va dau ra...
if not exist "input" mkdir input
if not exist "output" mkdir output

echo ===========================================
echo DA CAI DAT XONG! 
echo Hay copy anh vao thu muc 'input' roi chay run.bat
echo ===========================================
pause