# EM generalist: A Physics-Driven Diffusion Foundation Model for Electron Microscopy

## Overview

EM generalist is a cutting-edge diffusion-based foundation model designed to revolutionize electron microscopy (EM) and volumetric EM (vEM) image processing. Our physics-driven approach leverages over 1.7 million EM/vEM images to provide zero-shot solutions for diverse imaging challenges.

## Key Features

✔ **Multi-task Solution**  
- Denoising
- 2D Super-resolution reconstruction
- Deblurring
- 3D isotropic volume reconstruction

✔ **Technical Advantages**  
- Zero-shot reconstruction (no paired training needed)
- A unified solution for all tasks
- A foundation model for omni-sample type scanning elecron microscopy (SEM) data

## Usage Options
- The well-trained model weights can be downloaded at https://huggingface.co/10cbvkw/EM_generalist

### 🌐 Online Demo (Recommended)
For most users, we recommend our web platform:
🔗 [generativemicroscope.com](https://generativemicroscope.com)  
✅ No installation required  
✅ User-friendly interface  

### 💻 Local Deployment
This repository provides resources for:
1. Local deployment
2. Model training
3. Physical model embedding

# EM Generalist Code – User Guide

### 1. System requirements

- ✅ **Dependencies**: Python 3.9+, PyTorch ≥ 1.12, Diffusers = 0.29.0, Accelerate ≥ 0.16, NumPy, Matplotlib, scikit-image, tifffile
- ✅ **Operating systems tested**: Ubuntu 20.04, Ubuntu 22.04
- ✅ **Hardware**: 
  - Recommended: NVIDIA GPU with ≥ 16GB memory (e.g., RTX A6000, V100)
  - CPU mode is supported but significantly slower

### 2. Installation guide

- ✅ **Installation instructions**:
```bash
git clone https://github.com/YourRepo/EM-Generalist.git
cd EM-Generalist
conda create -n emgen python=3.9
conda activate emgen
pip install -r requirements.txt
```
- ✅ **Typical install time**: ~10 minutes on a standard desktop environment

### 3. Instructions for use

Before running any task, you need to download the pretrained diffusion prior model:

```bash
# Step 1: Download the pretrained model from Hugging Face
wget https://huggingface.co/10cbvkw/EM_generalist/resolve/main/prior_model.pt

# Step 2: Create the required directory if it doesn't exist
mkdir -p diffusion/train/exp/

# Step 3: Move the model into the expected path
mv prior_model.pt diffusion/train/exp/prior_model.pt
```

Once the model is placed correctly, run inference using task-specific scripts and parameters. For example, for **2D super-resolution** using the DPS method:

```bash
python 2D_sr.py   --path ./path/to/input_image.tif   --gamma 1.0   --factor 2
```

> Replace `./path/to/input_image.tif` with your own test image path. The output will be saved to the default results directory.

#### Other tasks:

| Task                  | Script Name                | Notable Args                     |
|-----------------------|----------------------------|----------------------------------|
| 2D denoising          | `denoise.py`               | `--path`, `--gamma`              |
| 2D deblurring         | `deblur.py`                | `--path`, `--gamma`, `--factor`  |
| 2D super resolution   | `2D_sr.py`                 | `--path`, `--gamma`, `--factor`  |
| 3D volume restoration | `3D_sr.py`                 | `--path`, `--gamma`, `--factor`  |

---

# EM Generalist Web Application – User Guide

*For the user guide video, please refer to ./online_app_user_guide.mp4 in this repository.

This web application provides a zero-shot solution for electron microscopy and volume electron microscopy (vEM) image restoration, supporting multiple tasks such as denoising, deblurring, and 2D/3D super-resolution using a unified diffusion-based foundation model.

## 1. Job Submission Interface

After visiting [generativemicroscope.com](https://generativemicroscope.com), users will be redirected to the application interface.
Due to sleep-based deployment, please note that the **first launch may take up to one minute** to activate the backend service. Once initialized, the application will operate normally.

As shown in **Figure 1**, users may upload input images through the left-hand panel of the **Job Submit** tab.

- **Supported file formats:** `.tif`, `.mrc`, `.npy`
- **Input requirements:** Image data should be single-channel (grayscale) and normalized within the value range **[0, 255]**

Users can:
- **Drag-and-drop** the file into the upload box  
- Or **click to open** the file selection dialog  

Once uploaded, the file is ready for task assignment and processing.

## 2. Using Example Files

In the **"Example File"** tab (see **Figure 2**), users may directly access a curated set of benchmark image samples:

- These examples reflect the typical test cases used across various publications
- Each file is named based on its **sample type**, **task category**, and **hyperparameter configuration**, such as:
  - `denoise_liver_stepsize_0.5.tif`
  - `2Dsr_x3_stepsize_0.1.tif`
  - `deblur_sigma_1_stepsize_0.1.tif`

Users can click to download, inspect, and re-upload them to the **Job Submit** tab for trial processing.

## 3. Job Parameters

On the right-hand panel of the **Job Submit** tab is the **Job Parameters** section:

- **Mode**:  
  Select one of the following tasks:
  - `denoise`
  - `deblur`
  - `super_resolution_2D`
  - `super_resolution_3D`

- **Stepsize**:  
  This is the **only hyperparameter** that requires manual tuning. It governs the update magnitude during inference sampling.
  
  - Recommended default: **0.3 to 0.5**
  - If set to `0`, the system will **automatically estimate** a suitable value based on the noise level or degradation type.
  - For **optimal performance on specific datasets**, it is recommended to **manually tune** this parameter.

Once the file and parameters are set, click **Upload and Process** to submit the task.

## 4. Job Execution and Results

- Processing time varies based on task complexity and image size. In general, each job takes a few minutes (see detailed timing benchmarks in the associated paper).
- After processing, the restored image appears in the **Submit Result** panel at the bottom.
- Click the **download icon** on the top-right of the result panel to download the output image.

## 5. Job List and History Tracking

In the **"Job List"** tab (see **Figure 3**), users can track their previously submitted tasks.

- Each entry includes:
  - **Job ID** (clickable for detailed view)
  - **Status** (e.g., Finished)
  - **Create Time** (in UTC+8 format)
  - **Execution Duration** (hh:mm:ss)
  
- During the **review period**, account login is not required — all prior jobs are stored and accessible.

Users may click on the **Job ID** to review both the original and processed image results.

## Acknowledgment

This tool was developed to facilitate reproducibility and accelerate electron microscopy workflows by providing accessible restoration solutions using a generalized diffusion model.

For details on model training, benchmark results, and recommended configurations, please refer to the original publication.
