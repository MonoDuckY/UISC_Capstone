@echo off
title Setup Moi Truong Tool Xoa Chu
echo ==========================================
echo    CÀI ĐẶT MÔI TRƯỜNG VÀ THƯ MỤC
echo ==========================================
echo.

echo [1/3] Kiem tra va tao thu muc anh...
if not exist "input_images" (
    mkdir input_images
    echo - Da tao thu muc 'input_images'
)
if not exist "output_images" (
    mkdir output_images
    echo - Da tao thu muc 'output_images'
)

echo.
if exist "env\Scripts\activate.bat" goto bo_qua_tao_moi

echo [2/3] Dang tao moi truong ao (virtual environment)...
python -m venv env
goto buoc_tiep_theo

:bo_qua_tao_moi
echo [2/3] Moi truong ao 'env' da ton tai, bo qua tao moi.

:buoc_tiep_theo
echo.
echo [3/3] Dang kich hoat va cai dat thu vien tu requirements.txt...
call env\Scripts\activate.bat
pip install -r requirements.txt

echo.
echo ==========================================
echo SETUP HOÀN TẤT! 
echo Vui long copy anh can xu ly vao thu muc 'input_images'
echo sau do chay file '2_run.bat' de su dung tool.
pause