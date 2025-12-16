# 👁️ Image Dehazing and Contrast Enhancement (AOD-Net & DCP)

This project investigates and compares classical (physics-based) and deep learning-based approaches for single-image dehazing and contrast enhancement. [cite_start]We implemented the **Dark Channel Prior (DCP)** method as a physical baseline [cite: 74, 85, 86] [cite_start]and the **All-in-One Dehazing Network (AOD-Net)** as the deep learning solution[cite: 74, 90].

[cite_start]The goal was to provide hands-on experience in solving a computer vision problem and analyze the trade-offs between physical models and learning-based techniques[cite: 7, 78, 79].

## 📄 Project Report and Source Code

| Component | Status | Details |
| :--- | :--- | :--- |
| **Final Report Title** | Complete | [cite_start]Image Dehazing and Contrast Enhancement for Outdoor Vision [cite: 66] |
| **Source Code Link** | Pending | [**Insert Your GitHub or Google Drive Link Here**] |
| **Group Number** | Assigned | [cite_start]28 [cite: 67] |

## 🧑‍💻 Developer Contributions

[cite_start]The project was divided into three distinct roles to cover all core requirements[cite: 27, 29].

| Developer | Role | Main Deliverable | Key Responsibilities |
| :--- | :--- | :--- | :--- |
| **Developer 1 (Tasfiq Jasimuddin)** | **Classical Dehazing Baseline** | [cite_start]Complete DCP implementation in Python with CLAHE post-processing[cite: 103, 105]. | [cite_start]Transmission map and atmospheric light estimation[cite: 111, 122]. |
| **Developer 2 (Saad Wasim)** | **AOD-Net Architecture & Debugging** | Implementation of the AOD-Net model, architectural fixing (Sigmoid activation), and hyperparameter tuning. | [cite_start]Fixing the convergence trap and managing GPU training[cite: 90]. |
| **Developer 3 (Rohan Uppal)** | **Evaluation & Data Pipeline** | [cite_start]Setup of the unified SOTS-Outdoor test data pipeline, metric calculation (PSNR/SSIM/VQS), and final analysis of results[cite: 76, 94]. | [cite_start]Evaluation metrics and comparison framework[cite: 44]. |

## ⚙️ Key Technical Findings & Model State (AOD-Net)

The AOD-Net development involved significant debugging and system migration due to initial hardware constraints.

### 1. Hardware Instability & Environment

* Initial development on local machines using consumer-grade GPUs was hampered by **unstable CUDA driver compatibility issues** (e.g., RTX 3070-class), forcing a temporary pivot to slow CPU-based testing.
* The project was successfully migrated to the **stable NVIDIA T4 GPU platform** via Google Colab for reliable, accelerated training.

### 2. AOD-Net Convergence Analysis

The hyper-tuned training revealed a critical limitation of the model's loss function:

| Issue | Solution Implemented | Status |
| :--- | :--- | :--- |
| **Architectural Bug** | [cite_start]Replaced the final layer's activation with **`torch.sigmoid`** to guarantee physical bounds on $K(x)$[cite: 63]. | **Fixed.** Confirmed active learning (First Batch Diff: >0.204). |
| **Training Convergence** | Implemented **Hyperparameter Kick** ($\text{LR}=5\text{e-}3, \epsilon=1\text{e-}5$) over 50 epochs to force global convergence. | **Stalled.** The loss plateaued at $\approx \mathbf{0.0454}$ (see log below), indicating the model converged to an **Identity Mapping Trap** (i.e., outputting the hazy input). |

**Original 50-Epoch Log (Demonstrating Convergence Stall)**

**Final State:** The AOD-Net architecture is proven correct and stable, but its final Average PSNR is $\mathbf{15.98 \text{ dB}}$. This result demonstrates the necessity of complex loss functions (e.g., Perceptual Loss) over simple Mean Squared Error (MSE) for robust image restoration.


### 1. Prerequisites

* Python 3.8+
* The required packages are listed in `requirements.txt`.

```bash
pip install -r requirements.txt