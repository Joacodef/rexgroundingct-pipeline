# Handshake: Phase 1 / Baseline Debugging & Split Performance Collapse Diagnosis

This document serves as an immediate operational and technical context bridge for active sessions.

---

## 1. Persistent Execution Status

*   **Completed Processes & Benchmarks:**
    *   Full 200-scan baseline validation rerun with `--tile_step_size 0.5` (`logs/execution_raw/voxtell_val_0.5.log`).
    *   Bucketed metrics calculation (`scripts/evaluate_bucketed.py` -> `logs/execution_raw/eval_bucketed_0.5.log`).
    *   Qualitative Error Analysis execution (`scripts/qualitative_error_analysis.py` -> `data/qualitative_analysis/`).
    *   **Empirical 4D Back-Reorientation & Shape Audit** (`scratch/phase_1_baseline_profiling/test_shape_hypothesis.py` on GPU 1): Confirmed spatial pipeline is mathematically correct. Spatial centroids align within ~6 voxels, and single-scan Dice reaches **0.6240** on aligned findings (`train_13013_a_1` pleural effusion).
    *   **Split Quantitative Evaluation** (`scratch/phase_1_baseline_profiling/fast_split_eval.py` & `fast_discrepancy.py`): Identified exact breakdown between original paper val split and new MICCAI challenge val split (`0.2139` vs `0.0491`).
    *   **Prompt Text Shift & Entity Extraction Analysis** (`scratch/phase_1_baseline_profiling/text_shift_analysis.py` -> `logs/experiments/phase_1_baseline_profiling/text_shift_analysis_report.md`): Quantified text shift metrics (max length 21 -> 35 words; comma rate 11.3% -> 23.7%). Initial evidence indicates that VoxTell's text encoder is sensitive to verbose clinical report sentences (*"Stable, nonspecific 6 mm..."* -> **0.0000 Dice**), while targeted entity queries (*"pulmonary nodule"*) recovered zero predictions up to **`0.4124` Dice** on single-finding test cases.

    *   **Full 200-Scan Normalized Baseline Evaluated** (`scratch/phase_1_baseline_profiling/fast_200_eval.py`): Completed all 200 validation scans with normalized text prompts (200-scan Average Dice: **`0.0889`**, Hit Rate: **`22.05%`**). Evaluated universal regex stripping vs targeted entity queries, confirming that zero-shot text normalization alone yields mixed results across diverse findings.

---

## 2. Hardware Topology & Environment Configuration

*   **GPU Isolation on `jumbito`:**
    *   **GPU 1 (RTX PRO 6000 Blackwell, 96 GB):** Designated operational GPU (`CUDA_VISIBLE_DEVICES=1`).
*   **Virtual Environment:** `.venv-voxtell` running Python 3.13.

---

## 3. Empirical Diagnostic Findings & Working Hypotheses

The 200-scan baseline evaluation yielded an **Overall Dice of 0.0988 (9.88%)** and **Hit Rate of 25.20%**. Empirical split evaluation revealed a critical distinction:

*   **First 50 Scans (Original Paper Val Split):**
    *   **Average Dice:** **`0.2139`** *(replicates paper benchmark ~0.21)*
    *   **Hit Rate ($\ge 0.1$):** **`49.57%`** (57/115 findings localized)
    *   **Empty Predictions:** 15.7% (18/115)
    *   **Median Prediction Size:** **6,451 voxels** (Median GT size: 3,687 voxels)
*   **Next 150 Scans (New MICCAI Challenge Val Split):**
    *   **Average Dice:** **`0.0491`** *(4.91%)*
    *   **Hit Rate ($\ge 0.1$):** **`14.66%`**
    *   **Empty Predictions:** **31.2% (83/266)**
    *   **Median Prediction Size:** **284 voxels** (Median GT size: 3,407 voxels)

### Empirical Findings:
1. **Instruction Template Matching**: In `voxtell/utils/text_embedding.py`, VoxTell wraps queries with `Instruct: Given an anatomical term query...`. Long radiology report narratives with non-diagnostic modifiers (*"Stable, nonspecific 6 mm"*) push text embeddings out-of-distribution, returning zero logits.
2. **Entity Query Recovery**: Converting long clinical descriptions to concise anatomical entity queries (*"pulmonary nodule"*) recovers zero predictions from **`0.0000` to `0.4124` Dice**.
3. **Spatial Location Preservation**: Retaining spatial clauses (*"in left lower lobe"*) preserves anatomical containment while stripping noisy modifier adjectives.

---

## 4. Action Plan & Next Hypotheses to Test (Phase 2 Fine-Tuning)

Phase 1 baseline profiling is **100% completed**. Zero-shot performance is fully documented across raw and normalized text prompt interfaces.

### Hypotheses to Test in Phase 2:
* **Hypothesis 1 (Instance Suppression Bias)**: We hypothesize that zero-shot model prediction suppression on complex findings is caused by training set partial annotations. We will test whether fine-tuning model weights on the 1,000 training scans improves validation Dice.
* **Hypothesis 2 (Positive-Unlabeled / SPOCO Loss)**: We hypothesize that treating unannotated regions as unlabelled rather than strict background negatives (via Positive-Unlabeled or SPOCO masked loss) will prevent performance degradation during fine-tuning.

### Immediate Action Plan:
1. **Verify SSD Training Cache**: Ensure the 1,000 preprocessed training volumes reside on `/tmp/jdeferrari/rexgroundingct_preprocessed/` to bypass CPU decompression bounds.
2. **Design Minimal Baseline Fine-Tuning Pipeline**: Implement a simple, measurable fine-tuning script (`scripts/train/train_voxtell_baseline.py`) to empirically test weight updating on GPU 1 before adding loss modifications.
3. **Launch Fine-Tuning Experiment on GPU 1**: Run training via `nohup` on GPU 1 (`RTX PRO 6000 Ada`).





