# Shorter version for lazy

Too bad...I'm lazier

# 

# SwinIR Image Super-Resolution Demo

This project demonstrates **image super-resolution using SwinIR**.  
It provides simple scripts to set up the environment and run image upscaling.

\---

## 📌 Features

* One-click environment setup
* Easy demo execution
* CPU-compatible (no GPU required)
* Uses pretrained SwinIR model

\---

## 📁 Project Structure

```
SwinIR/
│
├── run\\\_demo.bat          # Run inference
├── setup.bat             # Setup environment
│
├── main\\\_test\\\_swinir.py
├── models/
├── utils/
│
├── model\\\_zoo/            # Pretrained model (.pth)
├── testsets/
│   └── real\\\_sr/
│       └── LR/           # Input images
│
├── results/              # Output images (auto generated)
└── venv/                # Virtual environment
```

\---

## ⚙️ Setup

Run the setup script:

```
setup.bat
```

This will:

* Create a Python 3.10 virtual environment
* Install PyTorch (CPU)
* Install dependencies
* Fix NumPy compatibility

\---

## ▶️ Run Demo

After setup:

```
run\\\_demo.bat
```

\---

## 🖼️ Input Images

Put your images here:

```
testsets/real\\\_sr/LR/
```

Supported formats:

* .png
* .jpg
* .jpeg

\---

## 📤 Output

Results will be saved in:

```
results/swinir\\\_real\\\_sr\\\_x4/
```

\---

## ⚠️ Notes

* Requires **Python 3.10**
* Keep NumPy at version **1.26.4**
* Input folder must contain images

\---

## 🧠 Model

Model used:

```
003\\\_realSR\\\_BSRGAN\\\_DFO\\\_s64w8\\\_SwinIR-M\\\_x4\\\_GAN.pth
```

\---

## 🛠️ Troubleshooting

### No images processed

Ensure:

```
testsets/real\\\_sr/LR/
```

contains images.

### NumPy error

```
pip uninstall numpy -y
pip install numpy==1.26.4
```

### Module errors

Check project structure and run from root directory.

\---

## 📚 Reference

Official SwinIR repository:
https://github.com/JingyunLiang/SwinIR

\---

## 👨‍💻 Author

Capstone Project - SwinIR Demo

\---

## ✅ Status

* Setup working
* Demo working
* Ready for use

