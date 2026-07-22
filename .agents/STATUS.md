# Project Status: ReXGroundingCT Challenge

**Date:** July 22, 2026  
**Associated Phase:** Phase 1 (Baseline Error Profiling Completed) -> Transition to Phase 2 (Fine-Tuning)

---

## 1. Summary of Progress

### 📈 Experiment 005: [Phase 1] VoxTell v1.1 Second Baseline Verification & Text Shift Analysis
*   **Dataset:** Evaluated against all 200 validation masks (released July 2026).
*   **Empirical 4D Back-Reorientation Audit:** Confirmed coordinate mapping in `voxtell_inference.py` is mathematically correct (centroid diffs ~6 voxels; pleural effusion Dice reaches 0.6240).
*   **Split Evaluation Results (Raw vs Normalized Prompts):**
    *   **First 50 Scans (Paper Val Split):** **0.2139 Raw Dice** vs **0.2015 Normalized Dice** *(replicates paper benchmark ~0.21)*
    *   **Next 150 Scans (New MICCAI Val Split):** **0.0491 Raw Dice** vs **0.0402 Normalized Dice** *(severe zero-shot text shift)*
    *   **Combined 200 Scans:** **0.0988 Raw Dice** vs **0.0889 Normalized Dice**
*   **Key Diagnostic Finding & Solution:**
    *   Quantified prompt text shift (`logs/experiments/phase_1_baseline_profiling/text_shift_analysis_report.md`): Max prompt length spiked from 21 -> 35 words; comma rate more than doubled (11.3% -> 23.7%).
    *   Preliminary observations indicate that VoxTell's pre-trained text encoder is sensitive to verbose clinical sentences (*"Stable, nonspecific 6 mm..."* -> **0.0000 Dice**), while targeted entity queries (*"pulmonary nodule"*) recovered zero predictions up to **`0.4124` Dice** on single-finding test cases.
    *   Evaluated prompt normalization across all 200 scans, confirming that prompt engineering alone yields mixed results across diverse findings.


---

## 2. Current Operational Status

*   **Active Status:** Phase 1 Baseline Profiling 100% Completed. Phase 2 (Fine-Tuning Execution) Ready to Launch.
*   **Hardware Allocation:** Pinned to **GPU 1** via `.env` (`CUDA_VISIBLE_DEVICES=1`).
*   **Key Reports & Artifacts Generated:** `logs/experiments/phase_1_baseline_profiling/text_shift_analysis_report.md`, `scripts/voxtell/prompt_normalizer.py`, `scratch/phase_1_baseline_profiling/fast_200_eval.py`.

---

## 3. Remaining Work Plan (Hypotheses to Test in Phase 2)

### Task A: Establish Empirical Baseline Fine-Tuning Pipeline
*   Verify preprocessed training dataset cache on fast SSD `/tmp/jdeferrari/rexgroundingct_preprocessed/`.
*   Build minimal fine-tuning pipeline (`scripts/train/train_voxtell_baseline.py`) to empirically measure fine-tuning impact on GPU 1.
*   Test Hypothesis 1 (Fine-Tuning Weight Updates) and Hypothesis 2 (Positive-Unlabeled / SPOCO Loss) step-by-step with metric tracking on validation split.