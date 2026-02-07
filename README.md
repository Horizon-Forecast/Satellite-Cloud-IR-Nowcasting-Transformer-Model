# Horizon Forecast: Satellite Cloud IR Nowcasting Transformer Model

![Braude Logo](Phase%20-%20A/Images/braude_logo.png)

> **Project Code:** 26-1-R-1  
> **Institution:** Braude College of Engineering  
> **Department:** Software Engineering

---

## 🌩️ Project Overview
**Horizon Forecast** is a deep learning-based meteorological nowcasting system designed to predict short-term weather events (0-4 hours) with high spatiotemporal precision.

Unlike traditional AI models that treat weather forecasting as simple video prediction (resulting in "blurry" forecasts), this project utilizes a **Physics-Informed Cascaded Architecture**. By explicitly modeling the thermodynamic drivers (Wind & Temperature) before predicting the visual manifestation (Clouds & Rain), the model achieves sharper, physically consistent forecasts for the complex climatic region of Israel and the Eastern Mediterranean.

![Blurry vs Sharp Comparison](Phase%20-%20A/Images/concept_comparison.jpg)
*Figure 1: Conceptual comparison between standard MSE loss (Left) and our Physics-Informed approach (Right).*

---

## 👥 The Team
* **Students:**
    * **Or Mordechay Hod**
    * **Gilad Boudman**
* **Advisors:**
    * Mrs. Elena Kramer
    * Dr. Dan Lemberg

---

## 🧠 System Architecture
The system employs a **Cascaded Dual-Supervision Network**. It does not merely map past pixels to future pixels; it learns the *forces* driving the weather system.

![System Architecture](Phase%20-%20A/Images/architecture_diagram.jpg)

### The Two-Stage Inference Process:
1.  **Stage 1 (The Physics Drivers):**
    * **Input:** Latent features from the **SimVPv2 Encoder**.
    * **Task:** Predicts the invisible thermodynamic state: **Surface Wind Speed** and **Temperature**.
    * **Loss:** Masked MSE (Calculated only at active ground station coordinates).
2.  **Stage 2 (The Manifestation):**
    * **Input:** Fused tensor of Latent Visuals + Predicted Physics (from Stage 1).
    * **Task:** Predicts **Cloud Structure** and **Rain Intensity**.
    * **Innovation:** Uses **SaTformer** logic to classify rain into 64 probability buckets, preventing the "zero-inflation" problem.

---

## 🌍 Data Sources & Fusion
We fuse three distinct geospatial layers into a unified tensor for training:

![Data Fusion Strategy](Phase%20-%20A/Images/data_fusion.jpg)

* **🛰️ Top Layer (Dense Input):** EUMETSAT Meteosat Second Generation (MSG) Satellite Imagery.
    * *Channels:* IR 10.8µm (Thermal) & WV 6.2µm (Water Vapor).
* **📡 Middle Layer (Sparse Ground Truth):** IMS (Israel Meteorological Service) Ground Stations.
    * *Data:* Rain Intensity, Wind Speed, Surface Temperature.
* **🏔️ Bottom Layer (Static Context):** NASA SRTM Topography.
    * *Purpose:* Provides orographic lift context (mountain effects on rain).

---

## 🛠️ Technology Stack
* **Core Engine:** Python 3.9+, PyTorch 2.1
* **Training Framework:** PyTorch Lightning
* **Data Engineering:** EUMDAC (Satellite API), Pandas, Xarray, Dask
* **Geospatial Processing:** Rasterio (Projection alignment)
* **Hardware:** NVIDIA CUDA (GPU Acceleration)

---

## 📅 Project Timeline
The project development is divided into three strategic phases:

![Project Timeline](Phase%20-%20A/Images/development_process.png)

* **Phase A (Design):** Literature Review, Architecture Design, Mathematical Formulation. *(Completed)*
* **Phase B (Data Engineering):** Building the pipeline for EUMETSAT/IMS synchronization and tensor construction. *(Feb 2026 - Mar 2026)*
* **Phase C (Model & Eval):** Training the Neural Network, Hyperparameter Tuning, and Final Evaluation. *(Mar 2026 - Jun 2026)*

---

## 🚀 Getting Started
*(Instructions for future deployment during Phase B/C)*

### Prerequisites
```bash
pip install torch torchvision pytorch-lightning rasterio xarray eumdac