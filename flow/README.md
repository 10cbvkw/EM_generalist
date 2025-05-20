# Flow Version User Guide

This document provides instructions on how to run the Python scripts located in the `flow/inference` directory. 

### 1. Overview

The `inference` directory contains scripts for the following tasks:

- **2D Super-Resolution (`2d_sr.py`)**
- **Image Deblurring (`deblur.py`)**
- **Image Denoising (`denoise.py`)**
- **3D Super-Resolution (`3d_sr.py`)**

### 2. **Installation instructions**:

Clone the EM-Generalist repository, navigate to the flow directory, create and activate a new environment, then install the required packages using pip.

```bash
# Clone the repository to local
git clone https://github.com/YourRepo/EM-Generalist.git
# Navigate into the flow subdirectory
cd EM-Generalist/flow
# Create a new virtual environment
conda create -n flow python=3.11
conda activate flow
# Install the requiredpackages listed in requirements.txt
pip install -r requirements.txt
```
### 3. Instructions for use

Before running any task, you need to download the pretrained flow prior model:

```bash
# Step 1: Download the pretrained model from Hugging Face
# Step 2: Move the model into the expected path
mv flow_weights.pt model/flow_weights.pt
```

Once the model is placed correctly, run inference using task-specific scripts and parameters. For example, for **2D super-resolution** using the flow method:

```bash
python inference/2d_sr.py   --input ./path/to/input_image.tif   --alpha 0.3   --factor 2
```

> Replace `./path/to/input_image.tif` with your own test image path. 

#### Other tasks:

| Task                  | Script Name  | Notable Args                     |
| --------------------- | ------------ | -------------------------------- |
| 2D denoising          | `denoise.py` | `--input`, `--alpha`             |
| 2D deblurring         | `deblur.py`  | `--input`, `--alpha`, `--factor` |
| 2D super resolution   | `2d_sr.py`   | `--input`, `--alpha`, `--factor` |
| 3D volume restoration | `3d_sr.py`   | `--input`, `--alpha`, `--factor` |

`--alpha`: Data fidelity weight coefficient. Default: `0.3`（Typically 0.1-0.5）.

`--factor`: Factor for super-resolution / Sigma for Gaussian blur kernel

#### Output

- The output will be saved to the default results directory(e.g., `results/2d_sr`).
- Processed images are generally saved as PNG files (e.g., `output.png`).
- The `3d_sr.py` script saves its main output as a NumPy array file (`output.npy`) and also saves sample PNG images.
- The pixel values of the output images are typically scaled to the 0-255 range and saved as 8-bit unsigned integers.
