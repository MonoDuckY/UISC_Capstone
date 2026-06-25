@echo off
setlocal
cd /d "%~dp0"
echo --- Dang chay Cong cu Phat hien ROI Sieu am ---
echo.

if not exist venv\Scripts\python.exe goto no_venv

echo Dang chay 'main.py' trong moi truong ao...
venv\Scripts\python.exe main.py
if errorlevel 1 goto run_failed

echo.
echo --- Hoan thanh! Kiem tra ket qua trong thu muc 'output'. ---
echo.
pause
endlocal
exit /b 0

:no_venv
echo Loi: Khong tim thay moi truong ao. Vui long chay 'setup.bat' truoc.
pause
endlocal
exit /b 1

:run_failed
echo.
echo Loi: Chay chuong trinh that bai.
pause
endlocal
exit /b 1