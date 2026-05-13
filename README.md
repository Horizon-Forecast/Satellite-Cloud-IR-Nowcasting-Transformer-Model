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
* **Core Engine:** Python 3.13, PyTorch 2.11 + CUDA 12.8
* **Architecture:** SimVPv2 Encoder (gSTA blocks) + SaTformer-style classification head — implemented from scratch in raw PyTorch.
* **Data Engineering:** EUMDAC (Satellite API), Pandas, PyArrow (Parquet), Rasterio, SciPy (DEM mosaic + zoom).
* **Mixed Precision:** BF16 autocast on A5000/H100; FP16 + GradScaler fallback on consumer GPUs.
* **Hardware Used:** NVIDIA RTX A5000 24GB (primary local training), RTX 3070 8GB (inference fallback).

---

## 📅 Project Timeline
The project development is divided into three strategic phases:

![Project Timeline](Phase%20-%20A/Images/development_process.png)

* **Phase A — Design:** Literature Review, Architecture Design, Mathematical Formulation. ✅ *Completed (Feb 2026)*
* **Phase B — Data Engineering:** EUMETSAT/IMS synchronization, DEM mosaic, sample-index construction, normalization stats, rain-class weights. ✅ *Completed (May 2026)*
* **Phase C — Model & Evaluation:** Training the Cascaded Dual-Supervision Network, ablation study (`--no-cascade`), per-horizon evaluation against persistence / optical-flow / climatology baselines, qualitative case studies. 🚧 *In progress (May–Jun 2026)*

---

## 📁 Repository Structure
```
Horizon Forecast/
├── README.md                ← this file (project landing page)
├── Phase - A/               ← Phase A book (PDF) + presentation + figures
└── Phase B/                 ← Phase B + Phase C source code
    ├── entry_point.py       ← main training CLI
    ├── data_prep.py         ← data preparation CLI
    ├── run_viz.py           ← inference + visualization CLI
    ├── requirements.txt
    └── src/
        ├── data/    (dataset.py, prep.py)
        ├── models/  (model.py — 30.6M params, gSTA + transformer)
        ├── train/   (train.py — HorizonLoss, Trainer, FP16 inference loader)
        ├── eval/    (evaluate.py, baselines.py, inference.py)
        └── viz/     (visualize.py)
```
Large directories (`data/raw/`, `data/processed/`, `checkpoints/`, `venv/`) are intentionally **not committed** — they are reproduced locally by running the data preparation and training pipelines.

---

## 🚀 Getting Started

### Prerequisites
* Python **3.13**
* NVIDIA GPU with CUDA 12.x driver (RTX 3060+ / A5000 / H100 etc.)
* ~50 GB free disk for raw + processed data, ~5 GB for venv

### 1. Clone and set up the virtual environment
```bash
git clone https://github.com/Horizon-Forecast/Satellite-Cloud-IR-Nowcasting-Transformer-Model.git
cd Satellite-Cloud-IR-Nowcasting-Transformer-Model/Phase\ B
py -3.13 -m venv venv
venv\Scripts\activate                      # Windows
# source venv/bin/activate                 # Linux/macOS
pip install torch --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
```

### 2. Provide the raw data
The model requires three data sources under `Phase B/data/raw/`:
* `IR_108 with WV_062 Tif (EUMETVIEW)/Raw/YYYYMM/*.tif` — EUMETSAT SEVIRI imagery (IR 10.8 µm + WV 6.2 µm)
* `GroundTruth(IMS)/*.csv` — Israel Meteorological Service ground-station data
* `ElevationData(NASA)/*.hgt` — NASA SRTM digital elevation model tiles
* `stations_locations.csv` — IMS station GPS coordinates

### 3. Run the data preparation pipeline (one-time, ~30 min)
```bash
python data_prep.py                   # runs all 10 steps; idempotent — safe to re-run
python data_prep.py --steps verify    # just check artifacts
```
Produces `data/processed/` with the 256×256 DEM, station mask, per-timestamp IMS snapshots, train/val/test index CSVs, normalization stats, and rain-class weights.

### 4. Train
```bash
python entry_point.py --device-id 0 --precision bf16 --rollout-max 16 --max-epochs 80
```
Important flags:
* `--rollout-max 16`     — train all the way to the 4-hour autoregressive horizon (16 × 15 min)
* `--no-cascade`         — **ablation mode**: zero the drivers fed to the Manifestation Head (research-question control run)
* `--multihorizon-every 20` — every 20 epochs, run per-horizon evaluation (CSI/POD/FAR/SSIM at T+15, +30, +60, +120, +180, +240)
* `--smoke`              — 2-epoch sanity check; runs the full pipeline end-to-end fast

Checkpoints land under `Phase B/checkpoints/`:
* `gpu0_best.pt`         — lowest single-step validation loss
* `gpu0_best_rollout.pt` — best `0.5·SSIM@T+60 + 0.5·CSI@T+60` over the multi-horizon evaluations
* `gpu0_epoch_XXXX.pt`   — every 5 epochs (resume point)

### 5. Generate forecast visualizations
```bash
python run_viz.py --ckpt checkpoints/gpu0_best.pt --split val --n 5
```
Produces a multi-panel PNG per sample in `viz_output/`: input history, Stage 1 driver maps, Stage 2 cloud + rain predictions, ground-truth comparison.

### 6. Standalone evaluation on the test set
```bash
python -m src.eval.evaluate    # multi-horizon × multi-threshold table
python -m src.eval.baselines   # persistence, optical flow, climatology baselines
```

---

## 🧪 Research Question
> Does explicitly predicting thermodynamic atmospheric drivers (**Wind**, **Temperature**) in a first-stage cascade significantly improve the prediction of storm manifestation (**Clouds**, **Rain Intensity**) in the second stage — compared to end-to-end regression?

The codebase supports both training modes via the `--no-cascade` ablation flag. Phase C compares the two head-to-head and reports **per-horizon CSI / POD / FAR** at multiple rain-intensity thresholds, plus three reference baselines (Persistence, Farneback Optical Flow, Monthly Climatology).