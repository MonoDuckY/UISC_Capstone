@echo off
echo ==========================
echo SwinIR RUN
echo ==========================

cd /d "%~dp0"

call venv\Scripts\activate

echo Running SwinIR...

python main_test_swinir.py ^
--task real_sr ^
--scale 4 ^
--model_path model_zoo\003_realSR_BSRGAN_DFO_s64w8_SwinIR-M_x4_GAN.pth ^
--folder_lq "data\raw data" ^
--image_filter test_0001.png ^
--folder_output "data\processed data" ^
--output_suffix _processed

echo ==========================
echo DONE
echo ==========================
pause