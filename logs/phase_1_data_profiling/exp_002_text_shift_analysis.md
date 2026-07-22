# Experiment Log 002: [Phase 1] Quantitative Analysis of Validation Prompt Text Shift

**Date**: July 22, 2026  
**Purpose**: Document the free-text prompt syntax, phrasing, and vocabulary distribution shift between the original 50-scan paper validation split and the 150-scan new MICCAI challenge validation split.

---

## 1. Quantitative Text Metrics

| Metric | Cases 1–50 (Paper Split) | Cases 51–200 (New MICCAI Split) | Shift / Factor |
|---|---|---|---|
| **Total Finding Prompts** | 115 | 266 | 2.31x |
| **Mean Word Count per Prompt** | 10.98 words | 12.00 words | +9.3% |
| **Max Word Count in Prompt** | 21 words | **35 words** | **1.67x spike** |
| **Mean Character Length** | 70.0 chars | 80.0 chars | +14.3% |
| **Prompts with Comma Punctuation** | 11.3% | **23.7%** | **+12.4% (More than 2x)** |
| **Prompts with Adjective Modifiers** | 40.0% | 43.2% | +3.2% |
| **Prompts with Measurements (mm/cm)** | 17.4% | 19.2% | +1.8% |
| **Type-Token Ratio (TTR)** | 0.1544 | 0.0846 | -45.2% |

---

## 2. Qualitative Phrasing & Syntactic Comparison

### A. Cases 1–50 Prompts (Original Paper Split)
* Short, direct anatomical entity queries with concise spatial descriptors:
  * *"Air cyst at the anterobasal level of the right lower lobe"* (11 words)
  * *"Focal consolidative density at the paramediastinal level of the right middle lobe"* (11 words)
  * *"Focal scarring changes at the posterobasal level of the left lung"* (11 words)
  * *"Patchy ground glass densities in the inferior lingula of the left upper lobe"* (12 words)

### B. Cases 51–200 Prompts (New MICCAI Challenge Split)
* Compound, multi-clause clinical report sentences with dense anatomical jargon and multi-adjective prefixes:
  * *"Bilateral multifocal ground-glass opacities in peripheral subpleural and peribronchovascular regions of both lungs"* (14 words; uses compound jargon `"peribronchovascular"`)
  * *"Stable, nonspecific 6 mm subpleural nodule in the lateral basal segment of the left lower lobe"* (15 words; multiple non-diagnostic modifiers `"Stable, nonspecific 6 mm"`)
  * *"Subcentimeter, minimal, nonspecific focal ground-glass opacities in the posterobasal segment of the right lower lobe and in the right middle lobe"* (20 words; 3 stacked adjectives + multi-lobe spatial conjunction)

---

## 4. Full 200-Scan Quantitative Comparison: Raw vs Universal Normalized Prompts

| Partition | Findings | Raw Prompts Dice | Raw Hit Rate | Normalized Prompts Dice | Normalized Hit Rate | Empty Preds (Norm) |
|---|---|---|---|---|---|---|
| **First 50 Scans (Paper Val Split)** | 115 | **`0.2139`** | **`49.57%`** | `0.2015` | `45.22%` | 20.0% |
| **Next 150 Scans (New MICCAI Split)** | 266 | `0.0491` | `14.66%` | `0.0402` | `12.03%` | 36.5% |
| **Combined 200 Scans** | 381 | **`0.0988`** | **`25.20%`** | `0.0889` | `22.05%` | 31.5% |

### Key Empirical Findings:
1. **Targeted Entity Queries vs Universal Regex Stripping**:
   * For specific focal findings (e.g. `train_18382_b_2`), replacing complex report sentences with canonical entity terms (*"pulmonary nodule"*) recovers zero predictions from **`0.0000` up to `0.4124` Dice**.
   * However, applying universal regex prompt stripping across all 266 findings strips spatial location clauses (*"in left lower lobe"*), causing the zero-shot model to output broad predictions across both lungs, which reduces spatial precision on isolated findings.
2. **Working Hypotheses for Phase 2 Fine-Tuning**:
   * Empirical evidence indicates that zero-shot text prompt engineering alone is insufficient to recover validation performance across diverse findings on cases 51–200.
   * Additional evidence must be collected during Phase 2 fine-tuning to evaluate whether weight updates and loss modifications (e.g. Positive-Unlabeled or SPOCO masked loss) improve spatial grounding without suppressing predictions.


