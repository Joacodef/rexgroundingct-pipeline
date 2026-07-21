# Experiment Log 004: [Phase 1] Statistical Analysis of ReXGroundingCT Masks [COMPLETED]

> [!NOTE]
> **PIVOT NOTICE:**  
> The original plan for Experiment 004 (Exhaustive Multi-Finding Grounding) was discarded in favor of conducting a deep statistical analysis of the dataset's segmentation masks. This helps us understand the true distribution of findings and the gap between sparse annotations in the Train split vs exhaustive annotations in the Validation/Test splits.

* **Date:** July 2026
* **Status:** Completed

---

## 1. Objective
Perform a comprehensive statistical and visual analysis of the ReXGroundingCT dataset to understand dataset imbalances, average findings per volume, and the spatial distribution (heatmaps) of different pathologies.

## 2. Experimental Setup
* **Descriptive Statistics:** Computed volume sizes, unique entities per split, and absolute/relative segmentation volumes across Train and Validation datasets.
* **Spatial Heatmaps:** Generated Maximum Intensity Projections (MIP) in Coronal and Axial planes for the 8 most frequent pathologies.
* **Artifacts Generated:** Plots for entity counts, volume distributions, and 3D heatmaps.

## 3. Results & Findings
All visual artifacts and statistical insights have been documented in the project documentation folder. 
The findings validate the Positive-Unlabeled (PU) strategy we employed in earlier experiments, as the Train set is verified to be sparsely annotated (average ~1 finding per scan) while Validation is exhaustively annotated (average ~3 findings per scan).

* **Detailed Report:** `docs/exp_004_analysis_walkthrough.md`
