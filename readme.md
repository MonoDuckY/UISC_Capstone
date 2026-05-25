

# &#x20;Shorter version for lazy 

Clone/Download zip->run run\_first.bat and wait for 15min(result in result ofc) -> want more test?(Change inputs in inputs ofc) run run\_after.bat

**NOTE**: Don't worry if you see many red line, it not a bug, it a feature; When occur an error, don't ask me, I don't know either :v 



# &#x20;Real-ESRGAN Windows Setup (Stable + Fixed)

This project runs **Real-ESRGAN image upscaling** on Windows with a fully automated setup and run script.

It fixes common issues like:

* Python version conflicts (3.13 unsupported)
* NumPy 2.x crashes
* torchvision compatibility errors
* basicsr installation failures

\---

# &#x20;Features

* One-click environment setup
* CPU-safe PyTorch installation
* Stable dependency versions
* Simple input/output workflow
* Run-only mode after setup
* Works fully offline after installation

\---

# &#x20;Requirements

* Windows 10 / 11
* Python **3.10 (IMPORTANT)**
* Git (optional)
* At least 4GB RAM

\---

# &#x20;IMPORTANT

❌ Do NOT use Python 3.11 / 3.12 / 3.13  
✔ Only use Python 3.10 for stability

\---

# &#x20;Project Structure

```
Real-ESRGAN/
│
├── venv/                      # virtual environment (auto-created)
├── inference\\\_realesrgan.py   # main inference script
├── run\\\_demo.bat              # run-only script
├── setup.bat                 # full installer (if included)
├── inputs/                   # put images here
├── results/                  # output images
└── weights/
    └── RealESRGAN\\\_x4plus.pth
```

\---

# &#x20;Installation (First Time Only)

Run:

```
run\_first.bat
```

This will:

* Create virtual environment
* Install PyTorch (CPU version)
* Install Real-ESRGAN dependencies
* Download model weights

\---

# &#x20;Run Real-ESRGAN

After setup is complete:

```
run\_after.bat
```

Or manually:

```bash
python inference\\\_realesrgan.py -n RealESRGAN\\\_x4plus -i inputs -o results --fp32
```

\---

# &#x20;How to Use

1. Put images inside:

```
inputs/
```

2. Run:

```
run\_first.bat
```

3. Check output in:

```
results/
```

\---

# &#x20;Model Used

* RealESRGAN\_x4plus (General image upscaling x4)

Download source:
https://github.com/xinntao/Real-ESRGAN

\---

# &#x20;Fixed Issues

This setup solves:

### &#x20;NumPy 2.x crash

✔ Locked to numpy==1.26.4

### &#x20;torchvision missing module error

✔ Fixed by using torchvision==0.16.2+cpu

### &#x20;basicsr build failure

✔ Prebuilt wheel versions only

### &#x20;Python 3.13 incompatibility

✔ Forced Python 3.10 usage

\---

# &#x20;Performance

|Mode|Speed|Quality|
|-|-|-|
|CPU (--fp32)|Slow|High|
|GPU (optional)|Fast|High|

\---

# &#x20;Example Command

```bash
python inference\\\_realesrgan.py -n RealESRGAN\\\_x4plus -i inputs -o results --fp32
```

\---

# &#x20;License

This project uses:

* Real-ESRGAN (BSD-style license)
* PyTorch (BSD-style license)

