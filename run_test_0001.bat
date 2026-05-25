@echo off
echo ==============================================
echo SwinIR DEMO RUN - TEST 0001
echo ==============================================

cd /d "%~dp0"

:: Check if virtual environment exists, if not run setup
if not exist venv (
    echo [INFO] Virtual environment not found. Running setup first...
    call setup.bat
)

call venv\Scripts\activate

echo Running SwinIR on test_0001.png...
python main_test_swinir.py ^
--task real_sr ^
--scale 4 ^
--model_path model_zoo\003_realSR_BSRGAN_DFO_s64w8_SwinIR-M_x4_GAN.pth ^
--folder_lq "data\raw data" ^
--image_filter test_0001.png ^
--folder_output "data\processed data" ^
--output_suffix _processed

echo ==============================================
echo Running calculate_metrics...
echo ==============================================
python calculate_metrics.py

echo ==============================================
echo DONE!
echo Results saved to: data\processed data\test_0001_processed.png
echo Metric report saved to: data\metrics_report.csv
echo ==============================================
pause
