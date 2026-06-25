@echo off
setlocal
cd /d "%~dp0"
echo --- Thiet lap Moi truong ^& Cai dat Thu vien ---
echo.

where py >nul 2>nul
if errorlevel 1 goto no_python

if exist venv\Scripts\python.exe goto install_deps

echo Dang tao moi truong ao (venv)...
py -3 -m venv venv
if errorlevel 1 goto create_failed

:install_deps

echo Dang cai dat cac thu vien trong moi truong ao...
venv\Scripts\python.exe -m pip install --upgrade pip
if errorlevel 1 goto pip_failed

venv\Scripts\python.exe -m pip install -r requirements.txt
if errorlevel 1 goto requirements_failed

echo.
echo --- Thiet lap hoan tat! ---
echo.
pause
endlocal
exit /b 0

:no_python
echo Loi: Khong tim thay Python launcher 'py'. Vui long cai dat Python va them no vao PATH.
pause
endlocal
exit /b 1

:create_failed
echo Loi: Khong the tao moi truong ao.
pause
endlocal
exit /b 1

:pip_failed
echo Loi: Khong the cap nhat pip trong venv.
pause
endlocal
exit /b 1

:requirements_failed
echo Loi: Khong the cai dat cac thu vien.
pause
endlocal
exit /b 1