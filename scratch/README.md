# Scratch Workspace: Diagnostic & Tuning Tools

This directory contains experimental scripts, diagnostics, and tuning pipelines used to test spatial orientations, CT intensity windows, and model activation thresholds for the **ReXGroundingCT** challenge. 

To keep the workspace pristine, the scripts are organized into three subdirectories based on their functional domain.

---

## Workspace Structure

```text
scratch/
├── diagnostics/
│   ├── diagnose_spatial_alignment.py   # Axis permutations & flips search
│   ├── inspect_affines_and_shapes.py   # NIfTI affine & spacing validation
│   ├── inspect_metadata.py             # dataset.json & clinical prompt parser
│   ├── inspect_preprocess.py           # Preprocessing pipeline debugger
│   └── inspect_spatial_finding0.py     # Ground Glass localized scanner
│
├── intensity_tuning/
│   ├── tune_intensities_and_windows.py # Clipping window HU search
│   └── tune_thresholds_and_logits.py   # Sigmoid threshold & activation analyzer
│
└── tests/
    └── test_batch_validation.py        # Lightweight zero-shot runner
```

---

## Directory Descriptions & Scripts

### 📂 [scratch/diagnostics/](diagnostics)
*Focused on verifying coordinate alignment, image dimensions, coordinate spaces, and preprocessing outputs.*

*   **`diagnose_spatial_alignment.py`**: A spatial debugger that compares model predictions and ground truths. If they do not align, it automatically runs an exhaustive grid search over all 48 possible combinations of transpositions (axis swaps) and flips to find the correct alignment.
*   **`inspect_affines_and_shapes.py`**: A utility that loads raw and preprocessed NIfTIs, printing their internal shape arrays, coordinate spaces (e.g., RAS vs LPI), and Nibabel affine matrices. Used to detect hidden coordinate transpositions introduced by reading libraries (like `NibabelIOWithReorient`).
*   **`inspect_metadata.py`**: Inspects the main `dataset.json` structure, outputs lists of findings per scan, and computes the centers of mass and coordinate bounds for clinical annotations.
*   **`inspect_preprocess.py`**: Debugs the preprocessing pipeline (`scripts/data_prep/preprocess.py`) step-by-step, logging volume dimensions and intensity properties before/after spacing normalization.
*   **`inspect_spatial_finding0.py`**: Specifically isolates **Finding 0 (Ground Glass Opacities)**. It verifies that spatial bounding boxes match perfectly with the ground truth, validating that our coordinates are correct, and searches for prediction confidence patterns inside the pathology region.

---

### 📂 [scratch/intensity_tuning/](intensity_tuning)
*Focused on analyzing standard CT windowing operations, Hounsfield Unit (HU) clipping ranges, and probability calibration.*

*   **`tune_intensities_and_windows.py`**: Tests the impact of different Hounsfield Unit clipping windows (e.g., soft-tissue window `[-125, 275]`, pulmonary window `[-1000, 400]`, broad ranges, and intensity value shifts) on model prediction volume and Dice score accuracy.
*   **`tune_thresholds_and_logits.py`**: Conducts a parametric sweep over sigmoid activation thresholds (from `0.01` to `0.98`) across validation cases. It logs raw logit values, outputs mean/max probability statistics, and measures how different confidence levels affect the resulting Dice score and volume size.

---

### 📂 [scratch/tests/](tests)
*Focused on quick, end-to-end evaluation loops.*

*   **`test_batch_validation.py`**: A fast, lightweight baseline runner that executes zero-shot inference using the `VoxTell` model on a subset of the validation split (first 5 cases by default). It prints case-level Dice scores, ground truth volumes, prediction sizes, and overall averages to quickly verify pipeline correctness before kicking off full batch runs.

---

## ⚡ Safety & GPU Acceleration Note
All scripts in this workspace are configured with **automatic GPU isolation**. They load the environment from `.env` and configure `CUDA_VISIBLE_DEVICES` *before* importing PyTorch. 

When executing any of these scripts, they will automatically run on the compatible **NVIDIA RTX 6000 Ada** GPU (GPU 0), ensuring zero capability mismatch warnings and ultra-fast native GPU performance.
