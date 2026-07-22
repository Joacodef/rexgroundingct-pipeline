# Handshake: ReXGroundingCT Data & VoxTell Inference Audit Focus

This document serves as the active context bridge between sessions.

---

## 🎯 1. Operational Scope & Direction

All fine-tuning loops and loss modifications are **POSTPONED**. The active focus is 100% centered on:
1. **ReXGroundingCT Data Analysis**: Masks, 3D CT metadata, 14 finding categories, volume distributions, and free-text prompts.
2. **VoxTell Zero-Shot Inference Audit**: Preprocessing, spatial reorientation, sliding window tile overlap, probability logit distributions, and category-level failure modes.

---

## 📋 2. Completed Reorganization & Artifacts

* **Codebase Clean-Up**:
  * Moved legacy training code to [`scripts/archived_proof_of_concept/`](file:///home/jdeferrari/rex_project/scripts/archived_proof_of_concept/).
  * Moved legacy experiment logs to [`logs/experiments/archived_proof_of_concept/`](file:///home/jdeferrari/rex_project/logs/experiments/archived_proof_of_concept/).
  * Organized [`scratch/`](file:///home/jdeferrari/rex_project/scratch/) with active profiling scripts in [`scratch/phase_1_baseline_profiling/`](file:///home/jdeferrari/rex_project/scratch/phase_1_baseline_profiling/) and archived exploratory scripts in [`scratch/archived_proof_of_concept/`](file:///home/jdeferrari/rex_project/scratch/archived_proof_of_concept/).

* **Active Diagnostic Scripts & Reports**:
  * **[`text_shift_analysis.py`](file:///home/jdeferrari/rex_project/scratch/phase_1_baseline_profiling/text_shift_analysis.py)** -> **[`text_shift_analysis_report.md`](file:///home/jdeferrari/rex_project/logs/experiments/phase_1_baseline_profiling/text_shift_analysis_report.md)**: NLP analysis of free-text finding prompts.
  * **[`verify_official_pipeline.py`](file:///home/jdeferrari/rex_project/scratch/phase_1_baseline_profiling/verify_official_pipeline.py)**: Audit of official `NibabelIOWithReorient` and `VoxTellPredictor` inference against raw GT masks.
  * **[`exp_005_voxtell_baseline_full_val.md`](file:///home/jdeferrari/rex_project/logs/experiments/phase_1_baseline_profiling/exp_005_voxtell_baseline_full_val.md)**: Baseline 200-scan validation metrics log.

---

## 💻 3. Hardware & Operational Environment

* **GPU Allocation**: Pinned to **GPU 1** via `.env` (`CUDA_VISIBLE_DEVICES=1`, NVIDIA RTX PRO 6000 Blackwell).
* **Python Environment**: `.venv-voxtell` (Python 3.13 / CUDA 12.8).

---

## 🔬 4. Immediate Work Plan

1. **Category-Level Data Profiling**: Map validation findings across the 14 official ReXGroundingCT categories and profile GT volume sizes and instance counts.
2. **Category-Level Zero-Shot Baseline Benchmark**: Measure raw VoxTell zero-shot Dice across each of the 14 categories individually using official `NibabelIOWithReorient` inference.
3. **Probability Logit Profiling**: Inspect raw sigmoid probability maps to determine if false negatives stem from logit thresholding (`> 0.5`) or cross-attention activation drops.
