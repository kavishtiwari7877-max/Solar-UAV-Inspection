# Solar UAV Inspection & Anomaly Detection System

An enterprise computer vision and thermographic analytics framework designed for autonomous drone aerial inspection of large-scale solar photovoltaic (PV) plants.

---

## 📁 Project Architecture

The codebase follows a modular, production-ready directory structure:

```
Solar_UAV_Inspection/
│
├── data/
│   ├── original/                     # Raw aerial drone survey photos
│   │   ├── panel_01.jpg              # Thermal IR survey frame (Hotspots & Diode Defect)
│   │   └── panel_02.jpg              # Optical RGB survey frame (Cracks & Soiling)
│   │
│   ├── processed/                    # CLAHE preprocessed & normalized images
│   │
│   └── ground_truth/
│       └── annotations.csv           # Ground truth bounding boxes for model evaluation
│
├── outputs/
│   ├── annotated/                    # Images with AI bounding boxes & severity badges
│   │
│   └── reports/
│       ├── inspection_results.csv    # Defect registry & work-orders for maintenance
│       ├── inspection_report.html    # Standalone interactive browser audit report
│       └── inspection_report.pdf     # Engineering audit PDF report
│
├── src/
│   ├── __init__.py                   # Package exports
│   ├── preprocessing.py              # CLAHE enhancement, deskewing, panel segmentation
│   ├── detection.py                  # Multi-anomaly detection (Hotspot, Diode, Crack, Soiling)
│   ├── visualization.py              # Visual overlays, side-by-side comparative views
│   ├── reporting.py                  # PDF (ReportLab), HTML, and CSV report generators
│   ├── evaluation.py                 # Ground truth benchmarking (Precision, Recall, F1, IoU)
│   └── pipeline.py                   # High-level pipeline coordinator
│
├── tests/
│   ├── test_preprocessing.py         # Unit tests for preprocessing & segmentation
│   ├── test_detection.py             # Unit tests for anomaly detectors
│   └── test_reporting.py             # Unit tests for report generation
│
├── main.py                           # CLI entrypoint to execute batch inspection
├── requirements.txt                  # Python dependencies
├── README.md                         # Project documentation
└── .gitignore                        # Git exclusion rules
```

---

## 🚀 Quick Start

### 1. Installation
Install the required dependencies:
```bash
pip install -r requirements.txt
```

### 2. Run Batch Inspection
Execute the end-to-end pipeline on `data/original/`:
```bash
python main.py
```
This will:
1. Ingest `panel_01.jpg` (Thermal IR) and `panel_02.jpg` (Optical RGB).
2. Apply CLAHE contrast enhancement and save normalized imagery to `data/processed/`.
3. Segment all photovoltaic modules and detect anomalies.
4. Save annotated defect overlays to `outputs/annotated/`.
5. Generate `outputs/reports/inspection_results.csv`, `outputs/reports/inspection_report.html`, and `outputs/reports/inspection_report.pdf`.
6. Compute Precision, Recall, and F1-score against `data/ground_truth/annotations.csv`.

---

## 🔍 Modules Overview

### 1. `src/preprocessing.py`
- **`ImagePreprocessor`**: Ingestion, auto-detection of Thermal IR vs. Optical RGB modality, CLAHE contrast equalization, perspective deskewing, and false-color thermal colormap generation.
- **`PanelSegmenter`**: Contour analysis and adaptive thresholding to isolate each solar panel module while filtering out mounting racks and terrain clutter.

### 2. `src/detection.py`
- **Thermal Hotspots**: Point heating with differential temperature ($\Delta T$) measurement in °C compliant with IEC 62446-3.
- **Bypass Diode Failures**: Vertical 1/3 or 2/3 sub-string open/short-circuit detection.
- **Micro-Cracks**: High-frequency gradient analysis filtering out regular busbars.
- **Soiling & Dust**: Organic variance reduction and chromatic discoloration detection.

### 3. `src/evaluation.py`
- Benchmarks detection performance against `data/ground_truth/annotations.csv`.
- Computes Intersection-over-Union (IoU), True Positives (TP), False Positives (FP), False Negatives (FN), Precision, Recall, and F1-score.

### 4. `src/reporting.py`
- **`inspection_results.csv`**: Module-level defect log with coordinates, severity, and power loss projections.
- **`inspection_report.pdf`**: Multi-page engineering audit document with KPI cards and pilot sign-off blocks.
- **`inspection_report.html`**: Responsive browser summary with one-click print-to-PDF.

---

## 🧪 Running Unit Tests

Run the test suite with `pytest`:
```bash
pytest tests/ -v
```
