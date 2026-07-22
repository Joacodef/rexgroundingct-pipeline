# Scratch Directory: Phase 1-3 Diagnostic & Analysis Tools

This directory contains experimental tools, temporary analysis scripts, and diagnostic utilities used during data exploration, baseline profiling, fine-tuning, and evaluation for the **ReXGroundingCT** challenge.

## Current Directory Structure

### 1. `phase_1_baseline_profiling/`
*   **`audit_prompts.py`**: Audits free-text prompt lengths, vocabulary size, measurement phrases, and clinical modifier adjectives between cases 1–50 and cases 51–200.
*   **`test_shape_hypothesis.py`**: Empirical 4D back-reorientation and shape audit script confirming spatial alignment within ~6 voxels and single-scan Dice up to 0.6240 (`train_13013_a_1`).
*   **`fast_split_eval.py` & `fast_discrepancy.py`**: Fast multi-process scripts quantifying the baseline performance drop between the paper split (cases 1–50: 0.2139 Dice) and the new MICCAI split (cases 51–200: 0.0491 Dice).
*   **`test_category_queries.py` & `test_category_mapping_eval.py`**: Benchmark scripts demonstrating text-encoder Out-of-Distribution activation recovery from **0.0000 -> 0.4124 Dice** when replacing verbose clinical report narratives with canonical entity queries.
*   **`test_cleaned_probabilities.py` & `test_raw_sigmoid_logits.py`**: Scripts extracting raw continuous sigmoid probability maps and pre-threshold logits across sliding window inference tiles.
*   **`check_normalized_early_eval.py`**: Fast NIfTI dataobj reader script computing intermediate Dice, Hit Rate, and empty mask rate on completed background prediction batches.

### 2. `eval_scripts/`
*   **`run_mean_teacher_val_eval.py`**: Auxiliary script to quantitatively evaluate specific Mean Teacher checkpoints. It generates 3D predictions using a sliding window for both the Student and Teacher (EMA) networks, calculates the global Dice coefficient on the validation set, and compares them against the zero-shot baseline (v1.1).
*   **`eval_single.py`**: Validation script to test and log inference performance on single isolated volumes.

### 3. `data_utils/`
*   **`diagnose_dataset.py`**: Diagnostic utility for inspecting dataset metadata, dimensionalities, and missing findings.
*   **`download_missing_val.py`**: Script to connect to the Hugging Face hub and download missing validation CT volumes.

### 4. `exp_004_analysis/`
*   **`exp_004_heatmaps.py`**: Script to generate Coronal and Axial Maximum Intensity Projections (MIP) to visualize the spatial distribution of findings.
*   **`exp_004_stats.py`**: Script to compute descriptive statistics comparing the sparse Train annotations against the exhaustive Val+Test annotations.

### 5. `diagnostics/`
*   **`check_volume_*.py`**: Scripts containing diagnostic utilities for asserting metadata, spatial orientations, and tensor sanity of individual CT volumes (e.g. volume 38, 53).

## Execution Notes

All scripts in this directory read the environment configuration from `.env` to inherit hardware isolation variables (such as `CUDA_VISIBLE_DEVICES`), ensuring predictable behavior and avoiding unintentional occupation of resources in use by other server users.
