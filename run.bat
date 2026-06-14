@echo off
title Script Xoa Chu Anh Sieu Am
echo ===================================================
echo   DANG KICH HOAT MOI TRUONG VA CHAY SCRIPT OCR...
echo ===================================================

:: 1. Kiem tra xem thu muc moi truong ao da ton tai chua, neu chua thi tu dong tao
if not exist env_ocr (
    echo [INFO] Khong tim thay thu muc 'env_ocr'. Dang tu dong tao moi truong ao...
    python -m venv env_ocr
)

:: 2. Kich hoat moi truong ao
echo [INFO] Dang kich hoat moi truong ao 'env_ocr'...
call env_ocr\Scripts\activate.bat

:: 3. Cai dat thu vien bang cach goi truc tiep qua python module (Khac phuc loi thieu pip)
echo [INFO] Dang kiem tra va cai dat cac thu vien can thiet...
python -m pip install --upgrade pip
python -m pip install -r requirements.txt

:: 4. Chay file python
echo.
echo [INFO] Chay script xu ly anh...
echo ===================================================
python run.py

:: 5. Giu cua so cmd lai de xem ket qua
echo ===================================================
echo   QUA TRINH XU LY HOAN THANH!
echo ===================================================
pause