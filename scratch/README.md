# Scratch Directory: Data Profiling & VoxTell Inference Audit

This directory contains diagnostic utilities, analysis tools, and profiling scripts strictly focused on:
1. **ReXGroundingCT Data Analysis**: Masks, 3D CT images, 14 finding categories, volume distributions, and text prompts.
2. **VoxTell Inference & Preprocessing Audit**: Preprocessing requirements, input spatial orientation, sliding window tile overlap, probability thresholding, and failure mode profiling.

---

## Directory Structure

### 1. `phase_1_baseline_profiling/` (Active Profiling Tools)
* **`text_shift_analysis.py` & `audit_prompts.py`**: NLP prompt distribution profiling comparing free-text finding syntax across validation cases.
* **`test_shape_hypothesis.py`**: 4D Back-Reorientation spatial coordinate audit confirming spatial alignment in raw CT affine space.
* **`fast_split_eval.py` & `fast_200_eval.py`**: Fast multi-threaded evaluation scripts for raw baseline profiling.
* **`verify_official_pipeline.py`**: Verification script comparing official `NibabelIOWithReorient` reader/writer behavior against raw NIfTI GT masks.
* **`test_raw_sigmoid_logits.py` & `test_probability_thresholds.py`**: Continuous probability logit distribution analysis scripts.

### 2. `archived_proof_of_concept/` (Archived Preliminary Code)
Contains archived exploratory scripts from early proof-of-concept experiments (`data_utils/`, `diagnostics/`, `eval_scripts/`, `exp_004_analysis/`).

---

## Execution Notes
All active scripts load hardware and environment configuration from `.env` (`CUDA_VISIBLE_DEVICES=1` on GPU 1).
