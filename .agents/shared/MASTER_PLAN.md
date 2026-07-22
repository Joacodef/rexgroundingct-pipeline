# Master Plan — ReXGroundingCT Challenge 2026

**Primary Goal:** Top-3 on the leaderboard (September 2026) AND an original paper accepted at MICCAI 2026, built on rigorous data understanding and zero-shot baseline inference mastery.

> [!IMPORTANT]
> **Strategic Focus Shift**: All fine-tuning experiments, loss modifications (SPOCO, MPR consistency), and training loops are **POSTPONED**. The repository and active research are 100% centered on:
> 1. **In-Depth ReXGroundingCT Data Analysis** (images, 3D masks, free-text findings).
> 2. **VoxTell Zero-Shot Inference & Preprocessing Audit** (inputs, spatial orientation, sliding window tile overlap, probability logits, and category-level failure modes).

---

## 🔬 1. Core Research Pillars

### Pillar A: Comprehensive ReXGroundingCT Data Analysis
* **3D CT Image & Metadata Profiling**: Voxel spacings, orientation affines, physical dimensions, and intensity distributions.
* **Exhaustive vs Sparse Mask Analysis**: Quantitative comparison of ground-truth mask distributions between the training set (partially annotated) and validation set (exhaustively annotated).
* **The 14 Official Finding Categories**: Detailed error profiling across the 14 challenge categories:
  * *Non-focal (6)*: Bronchial wall thickening, Bronchiectasis, Emphysema, Septal thickening, Micronodules, Other non-focal.
  * *Focal (8)*: Linear opacities, Atelectasis / consolidation, Ground-glass opacity, Pulmonary nodules / masses, Pleural effusion / thickening, Honeycombing, Pneumothorax, Other focal.
* **Finding Volume & Multi-Instance Statistics**: Distribution of component counts, voxel volumes, and spatial centroids per finding category.
* **Free-Text Radiology Report Analysis**: Quantitative NLP analysis of finding descriptions in `dataset.json` (syntax, modifier adjectives, length, and anatomical jargon).

### Pillar B: In-Depth VoxTell Inference & Preprocessing Audit
* **Official Preprocessing & Reorientation**: Audit official `nnunetv2.imageio.nibabel_reader_writer.NibabelIOWithReorient` and `VoxTellPredictor` to ensure 100% fidelity with the authors' intended input pipeline.
* **Sliding Window Hyperparameter Sensitivity**: Evaluate tile step size (`tile_step_size` 0.5 vs 0.25), Gaussian tile weighting, and patch padding on prediction quality.
* **Continuous Logit & Threshold Profiling**: Analyze raw sigmoid output probabilities prior to binarization (`> 0.5`) to determine whether false negatives are caused by low probability magnitude or spatial misalignment.
* **Category-Level Failure Mode Profiling**: Systematically identify which of the 14 categories succeed zero-shot and which fail, analyzing spatial and text characteristics of failure cases.

---

## 🗓️ 2. Project Roadmap

### Phase 1: Deep Data Profiling of ReXGroundingCT 🟢 ACTIVE
* Perform comprehensive statistics on CT scans, 3D GT masks, and free-text prompts across all 200 validation cases.
* Map ground truth masks to the 14 official challenge categories.

### Phase 2: VoxTell Zero-Shot Inference & Preprocessing Audit 🟢 ACTIVE
* Benchmark VoxTell v1.1 on raw unmodified challenge prompts using official `NibabelIOWithReorient`.
* Profile raw logit probability distributions and optimal binarization thresholds per category.
* Perform fine-grained error analysis across the 14 categories.

### Phase 3: Fine-Tuning & Model Adaptations ⏳ POSTPONED
* *Postponed until Data Analysis and Inference Audit are 100% completed and validated.*
* Any future fine-tuning methods (standard supervised, Positive-Unlabeled, or consistency losses) will be formulated strictly as hypotheses to test step-by-step.

---

## 🚫 3. Archived Proof-of-Concept Code
All previous exploratory Mean Teacher / SPOCO training scripts and logs are archived in:
* **`scripts/archived_proof_of_concept/`**: Legacy training scripts (`train_mean_teacher.py`).
* **`logs/experiments/archived_proof_of_concept/`**: Legacy experiment logs (`exp_002`, `exp_003`).
* **`scratch/archived_proof_of_concept/`**: Legacy scratch scripts.
