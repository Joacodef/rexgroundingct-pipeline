# Project Status: ReXGroundingCT Challenge

**Date:** July 22, 2026  
**Associated Phase:** Data Profiling & VoxTell Inference Audit

---

## 1. Summary of Progress

*   **Repository Alignment**: Postponed all SPOCO / fine-tuning experiments to focus 100% on ReXGroundingCT data analysis and VoxTell zero-shot inference audit.
*   **Directory Reorganization**:
    *   Legacy training code moved to `scripts/archived_proof_of_concept/`.
    *   Legacy logs moved to `logs/experiments/archived_proof_of_concept/`.
    *   Legacy scratch tools moved to `scratch/archived_proof_of_concept/`.
*   **Inference & Data Audit**:
    *   Verified official `NibabelIOWithReorient` and `VoxTellPredictor` execution.
    *   Quantified free-text prompt syntax metrics in `logs/experiments/phase_1_baseline_profiling/text_shift_analysis_report.md`.

---

## 2. Current Operational Status

*   **Active Status:** Data Profiling & Zero-Shot VoxTell Inference Audit.
*   **Hardware Allocation:** Pinned to **GPU 1** via `.env` (`CUDA_VISIBLE_DEVICES=1`).
*   **Key Reports & Artifacts:** `logs/experiments/phase_1_baseline_profiling/text_shift_analysis_report.md`, `scratch/phase_1_baseline_profiling/verify_official_pipeline.py`.

---

## 3. Immediate Work Plan

### Task A: 14-Category Data & Performance Breakdown
*   Map validation findings to the 14 official challenge categories.
*   Compute per-category zero-shot Dice scores using official VoxTell inference.
*   Profile GT finding volumes, instance counts, and text prompt complexity per category.