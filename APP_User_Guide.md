
# EM Generalist Web Application – User Guide

**A physics-driven diffusion foundation model for electron microscopy (EM)**

## Overview

This web application provides a zero-shot solution for electron microscopy and volume electron microscopy (vEM) image restoration, supporting multiple tasks such as denoising, deblurring, and super-resolution using a unified diffusion-based foundation model.

---

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

---

## 2. Using Example Files

In the **"Example File"** tab (see **Figure 2**), users may directly access a curated set of benchmark image samples:

- These examples reflect the typical test cases used across various publications
- Each file is named based on its **sample type**, **task category**, and **hyperparameter configuration**, such as:
  - `denoise_liver_stepsize_0.5.tif`
  - `2Dsr_x3_stepsize_0.1.tif`
  - `deblur_sigma_1_stepsize_0.1.tif`

Users can click to download, inspect, and re-upload them to the **Job Submit** tab for trial processing.

---

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

---

## 4. Job Execution and Results

- Processing time varies based on task complexity and image size. In general, each job takes a few minutes (see detailed timing benchmarks in the associated paper).
- After processing, the restored image appears in the **Submit Result** panel at the bottom.
- Click the **download icon** on the top-right of the result panel to download the output image.

---

## 5. Job List and History Tracking

In the **"Job List"** tab (see **Figure 3**), users can track their previously submitted tasks.

- Each entry includes:
  - **Job ID** (clickable for detailed view)
  - **Status** (e.g., Finished)
  - **Create Time** (in UTC+8 format)
  - **Execution Duration** (hh:mm:ss)
  
- During the **review period**, account login is not required — all prior jobs are stored and accessible.

Users may click on the **Job ID** to review both the original and processed image results.

---

## Acknowledgment

This tool was developed to facilitate reproducibility and accelerate electron microscopy workflows by providing accessible restoration solutions using a generalized diffusion model.

For details on model training, benchmark results, and recommended configurations, please refer to the original publication.
