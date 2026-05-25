import os
import urllib.request

MODEL_DIR = "weights"
os.makedirs(MODEL_DIR, exist_ok=True)

MODEL_URLS = {
    "RealESRGAN_x4plus": "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth"
}

def download_model(name):
    if name not in MODEL_URLS:
        raise ValueError(f"Unknown model: {name}")

    url = MODEL_URLS[name]
    save_path = os.path.join(MODEL_DIR, f"{name}.pth")

    if os.path.exists(save_path):
        print(f"[OK] Model already exists: {save_path}")
        return save_path

    print(f"[DOWNLOADING] {name}...")
    print(url)

    urllib.request.urlretrieve(url, save_path)
    print(f"[DONE] Saved to {save_path}")
    return save_path


if __name__ == "__main__":
    download_model("RealESRGAN_x4plus")