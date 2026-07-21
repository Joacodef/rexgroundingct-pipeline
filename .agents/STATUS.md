# Project Status: ReXGroundingCT Challenge

**Date:** July 15, 2026  
**Associated Phase:** Phase 1 - Baseline Error Profiling & Alternative Scouting

---

## 1. Summary of Progress

### 📈 Experiment 005: [Phase 1] VoxTell v1.1 Second Baseline Verification (Active)
*   **Dataset:** Evaluated against the complete set of 200 validation masks (released July 2026).
*   **Inference Execution:** Successfully generated raw 4D NIfTI predictions on all 200 validation scans using a memory-leak-safe Bash Chunking Wrapper on the RTX PRO 6000 Blackwell GPU.
*   **Performance Anomaly:** The zero-shot evaluation yielded an **Average Dice of 0.0975** and a **Hit Rate of 24.14%**. This is a severe degradation compared to the leaderboard scores (>0.20) for the exact same pre-trained weights (`ibrahimhamamci/voxtell_v1.1`).
*   **Error Bucketing (Quantitative Analysis):** 
    *   Small findings (<1k voxels) dropped to ~7% Dice.
    *   Massive entities (e.g., Effusions) scored up to 25% Dice.
    *   The model suffers from severe *Instance Suppression Bias*, but the drop to 9.7% suggests technical implementation errors in the evaluation pipeline.

### 🧪 Technical Hypotheses for Baseline Degradation
1.  **Aggressive Sliding Window:** To speed up inference, the chunking wrapper used `--tile_step_size 0.75` (25% overlap). This likely destroyed context at the patch boundaries, obliterating the performance on small and medium findings.
2.  **Affine Orientation Alignment:** A potential discrepancy between how `voxtell_inference.py` back-reorients the predictions to the original scan space vs. how the official evaluation script processes the identity affine bug in the Ground Truth masks.

---

## 2. Current Operational Status

*   **Active Processes:** Running full 200-scan baseline validation using `0.5` tile step size via bash chunking wrapper.
*   **Hardware Allocation:** All inference operations are strictly pinned to **GPU 1** via `.env` (`CUDA_VISIBLE_DEVICES=1`). The previous zombie processes on the 48GB GPU 0 have been killed.

---

## 3. Remaining Work Plan (Next Immediate Steps)

### Task A: Rerun Full Baseline Validation (Active)
*   **Action:** Execute the full 200-scan validation batch again using `--tile_step_size 0.5` via the chunking wrapper.
*   **Goal:** Re-establish the baseline with the correct overlap, expecting the Dice to recover above `>0.20` as proven by the single-scan test.

### Task B: Qualitative Error Analysis (Visual Inspection)
*   Once the baseline numbers are technically sound (or if the anomaly persists), identify the 5 worst-performing scans from the validation set.
*   Extract 2D slices/heatmaps to visually inspect whether VoxTell generates hallucinated garbage masks or simply outputs empty zero-tensors (confirming the extreme False Negative rate of the Instance Suppression Bias).

### Task C: Proceed to Phase 2
*   Depending on the outcome of the qualitative analysis, formulate the first set of micro-experiments (e.g., modifying the standard loss function or applying thresholding/pseudo-labeling).