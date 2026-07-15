# Scratch Directory: Phase 2 & 3 Validation and Diagnostic Tools

This directory contains experimental tools, temporary analysis scripts, and diagnostic utilities used during the fine-tuning, evaluation, and data exploration phases of the **ReXGroundingCT** challenge.

## Current Structure

### 1. `eval_scripts/`
*   **`run_mean_teacher_val_eval.py`**: Auxiliary script to quantitatively evaluate specific Mean Teacher checkpoints. It generates 3D predictions using a sliding window for both the Student and Teacher (EMA) networks, calculates the global Dice coefficient on the validation set, and compares them against the zero-shot baseline (v1.1).

### 2. `data_utils/`
*   **`diagnose_dataset.py`**: Diagnostic utility for inspecting dataset metadata, dimensionalities, and missing findings.
*   **`download_missing_val.py`**: Script to connect to the Hugging Face hub and download missing validation CT volumes.

### 3. `exp_004_analysis/`
*   **`exp_004_heatmaps.py`**: Script to generate Coronal and Axial Maximum Intensity Projections (MIP) to visualize the spatial distribution of findings.
*   **`exp_004_stats.py`**: Script to compute descriptive statistics comparing the sparse Train annotations against the exhaustive Val+Test annotations.

## Execution Notes

All scripts in this directory read the environment configuration from `.env` to inherit hardware isolation variables (such as `CUDA_VISIBLE_DEVICES`), ensuring predictable behavior and avoiding unintentional occupation of resources in use by other server users.
