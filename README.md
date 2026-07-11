# Horizon Forecast — Driver-First Cascade Nowcasting

> **Project Code:** 26-1-R-1
> **Institution:** Braude College of Engineering — Software Engineering Department
> **Students:** Or Mordechay Hod, Gilad Boudman
> **Advisors:** Mrs. Elena Kramer, Dr. Dan Lemberg

A deep-learning **nowcasting** system for Israel: from the last hour of weather-satellite
imagery it forecasts the near future of **wind, temperature, cloud structure, and rain**
over a 256×256 grid, in 15-minute steps.

## The idea — Driver-First Cascade

Instead of mapping past satellite frames straight to future rain, the model first predicts the
physical **drivers** of the weather (surface wind and temperature), then conditions a second
stage on those drivers to predict the **clouds and rain** — *understand why the weather moves
before predicting where it goes.*

We test this with a clean **ablation**: the same network with the driver link switched off
(`--no-cascade`). On the strictly held-out test set (Jul 2024 – Dec 2025), the cascade beats
the ablation on rain skill while cloud quality stays tied:

| Model (T+15) | Rain CSI | Heavy-rain CSI | Cloud SSIM |
|---|---|---|---|
| **Driver-First Cascade** | **0.135** | **0.114** | 0.81 |
| End-to-End Ablation | 0.089 | 0.017 | 0.82 |

The driver-first ordering improves rain CSI by **~52%** at every horizon. The low *absolute*
CSI is a ground-truth-density limit (rain is verified only at ~100 sparse stations), not a
model failure — the full analysis is in the project book.

## Architecture (~30M params, `Phase B/src/models/model.py`)

```
input (B, 12, 256, 256)              4 frames × [IR, WV, DEM]
  → SimVPv2 Encoder (gSTA blocks)    → shared latent representation
  → Stage 1: PhysicsDriverHead       → wind + temperature (supervised by dense ERA5)
  → Stage 2: ManifestationHead(latent, drivers)
        ├ cloud head → IR + WV        (verified against dense satellite imagery)
        └ rain head  → 64 intensity-class logits  (verified against sparse IMS stations)
```

## Data

| Source | Role |
|---|---|
| EUMETSAT SEVIRI (IR 10.8 µm + WV 6.2 µm) | model input (dense) |
| NASA SRTM (elevation) | model input (static) |
| ERA5 reanalysis (wind, temperature) | dense driver ground truth |
| IMS stations (rain) | sparse rain ground truth |

Grid: 29–34°N, 34–36°E. Splits: train 2020–2023, validation 2024 H1, held-out test
2024-07 → 2025-12 (52,566 samples).

## Poster

<p align="center">
  <a href="Phase%20B/documents/Capstone%20Project%20%E2%80%93%20Phase%202_26-1-R-1_Poster.pdf">
    <img src="Phase%20B/documents/poster.png" alt="Horizon Forecast — project poster" width="480">
  </a>
</p>

<p align="center"><em>Click the poster for the full-resolution PDF.</em></p>

## Repository layout

```
Horizon Forecast/
├── README.md            ← this file
├── Phase A/             ← Phase A proposal book + figures
└── Phase B/             ← the delivered system
    ├── entry_point.py   ← training CLI
    ├── run_viz.py       ← load a checkpoint → render a forecast dashboard
    ├── requirements.txt
    ├── src/             ← the package (data, models, train, eval, viz)
    ├── scripts/         ← eval, weight export, demo builder, training launchers
    ├── data/            ← data-download + preprocessing scripts, the 4 committed artifacts
    ├── weights/         ← the two delivered models (FP16, driver_first.pt / end_to_end.pt)
    ├── demo/            ← standalone offline viewer (.exe) + walkthrough video
    ├── experiments/     ← neighborhood-CSI analysis
    └── documents/       ← project book + poster (PDF) and poster.png
```

Large artifacts — `checkpoints/`, the heavy `data/` subdirectories, and `logs/` — are
git-ignored and rebuilt locally.

## Quick start

Full instructions are in the project book (`Phase B/documents/`) — User Guide (Appendix A) and
Maintenance Guide (Appendix B). In short, clone the repo and set up the environment from
`Phase B/`:

```bash
git clone https://github.com/Horizon-Forecast/Satellite-Cloud-IR-Nowcasting-Transformer-Model.git
cd Satellite-Cloud-IR-Nowcasting-Transformer-Model/"Phase B"
python -m venv venv
venv\Scripts\activate                     # Windows (use source venv/bin/activate on Linux/macOS)
pip install torch --index-url https://download.pytorch.org/whl/cu128   # match your CUDA driver
pip install -r requirements.txt
```

The two trained models ship in `weights/`. Reproduce the held-out test results (requires the
processed dataset — rebuild it from the raw sources per Appendix B):

```bash
python scripts/eval_all_test.py --sub 10000 --horizons 1,2,3,4,6,8
```

Rebuild the processed dataset from raw downloads (after obtaining API keys — see Appendix B):

```bash
python -m src.data.prep
```

Retrain from scratch (two-phase frozen schedule, requires the processed dataset and a GPU —
full commands and flags in Appendix B):

```bash
python entry_point.py --era5-path data/era5_npy --freeze-stage mani --lambda-cloud 0 --lambda-rain 0 --ckpt-dir checkpoints/phase1
python entry_point.py --era5-path data/era5_npy --resume-ckpt checkpoints/phase1/gpu0_best.pt --resume-weights-only --freeze-stage encoder_phys --lambda-thermo 0 --lambda-cloud 1.0 --lambda-rain 0.5 --ckpt-dir checkpoints/phase2
```

## Demo

- **Standalone offline viewer:** `Phase B/demo/dist/HorizonForecastDemo.exe` — a self-contained
  Windows executable (no Python, no GPU) that steps through curated test cases, comparing the
  Driver-First Cascade against the ablation across every lead time.
- **Narrated walkthrough video:** [`Phase B/demo/video/Demo_Video.mp4`](Phase%20B/demo/video/Demo_Video.mp4).

## Research question

> Does explicitly predicting the thermodynamic drivers (wind, temperature) in a first-stage
> cascade measurably improve rain nowcasting, compared to an otherwise identical end-to-end
> model?

Answered by the controlled `--no-cascade` ablation: on this data, yes — the driver-first
ordering improves rain forecasting specifically (not the image as a whole), by ~52% CSI.
