# Ultrasound Caliper Detection & XML Annotation Generator

An automated computer vision tool built with OpenCV for detecting ultrasound caliper markers (`plus` and `x_mark`) and automatically generating Pascal VOC XML annotations. This tool is designed to accelerate dataset preparation for Deep Learning object detection models such as YOLO, Faster R-CNN, SSD, and similar frameworks.

The project uses a dedicated Python virtual environment to ensure dependency isolation and prevent conflicts with system-wide Python installations.

---

## Features

* **Isolated Virtual Environment**

  * Creates a local `.venv` environment to keep project dependencies separate from your global Python installation.

* **Multi-Template Matching**

  * Detects caliper markers using multiple template samples simultaneously for improved robustness.

* **Adaptive Otsu Thresholding**

  * Reduces color bleeding and improves contour extraction accuracy.

* **Automatic Pascal VOC XML Generation**

  * Generates annotation files containing both `plus` and `x_mark` detections in a single XML file per image.

* **Organized Output Structure**

  * Stores generated files in dedicated output folders:

    * `output/output_images/`
    * `output/output_xmls/`
    * `output/output_combined/`

---

## Installation & Setup

Before using the tool, initialize the project's Python virtual environment.

### Prerequisites

* Windows operating system
* Python 3.x installed
* Python added to the system `PATH`

### First-Time Setup

1. Clone or download this repository.
2. Run **`setup.bat`**.
3. The setup script will automatically:

   * Create a local virtual environment (`.venv`)
   * Upgrade `pip`
   * Install all required dependencies from `requirements.txt`

After setup is complete, the environment is ready for use.

---

## Usage

### Step 1: Prepare Input Images

Place all ultrasound images to be processed inside the `input/` directory.

### Step 2: Prepare Templates

Place all caliper template images inside the `templates/` directory.

### Step 3: Run the Tool

Double-click **`run.bat`**.

The script will:

* Activate the local virtual environment
* Execute the image processing pipeline
* Generate annotated images and XML files

### Step 4: Review Results

Generated files will be available in the `output/` directory.

---

## Advanced Configuration

### Adjusting Detection Sensitivity

If detections are missing or too many false positives are generated, adjust the template matching threshold.

Open `process_images.py` and locate:

```python
processed_img, all_boxes = highlight_and_extract_all_boxes(
    img_to_process,
    templates_dir,
    threshold=0.72
)
```

Modify the `threshold` value according to your needs:

| Threshold Range | Effect                                                                |
| --------------- | --------------------------------------------------------------------- |
| 0.65 – 0.68     | Higher sensitivity. Better for faint or low-contrast markers.         |
| 0.70 – 0.75     | Balanced performance (recommended).                                   |
| 0.75 – 0.80     | Stricter matching. Reduces false positives but may miss weak markers. |

---

### Adding New Templates

Different ultrasound machines may use slightly different caliper designs. You can improve detection accuracy by adding additional templates.

#### Creating a New Template

1. Crop an undetected caliper marker from an ultrasound image.
2. Recommended template size: **17×17 to 21×21 pixels**.
3. Clean the image using an image editor:

   * Background: pure black (`#000000`)
   * Marker: pure white (`#FFFFFF`)
4. Save the template inside the `templates/` folder.

#### Naming Rules

Template filenames determine the generated annotation label:

| Filename Example | Generated Label |
| ---------------- | --------------- |
| `plus_faint.png` | `plus`          |
| `plus_small.png` | `plus`          |
| `x_v2.png`       | `x_mark`        |
| `x_alt.png`      | `x_mark`        |

As long as the filename contains:

* `plus` → labeled as `plus`
* `x` → labeled as `x_mark`

the XML annotation will be generated automatically with the correct class name.

---

## Output Overview

For each processed image, the tool generates:

* Annotated preview image showing detected markers
* Pascal VOC XML annotation file
* Combined output package for dataset preparation

This allows the generated dataset to be used directly for training object detection models with minimal manual labeling effort.
