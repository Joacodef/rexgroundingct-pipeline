# VoxTell Baseline Knowledge & Fine-Tuning Rationale

> [!IMPORTANT]
> This document details key findings about the pre-trained **VoxTell** baseline model (`voxtell_v1.1`) and establishes the mathematical/methodological justification for the Phase 2 Semi-Supervised Fine-Tuning pipeline.

---

## 🏁 1. VoxTell Pre-Training Distribution
According to the official VoxTell paper (*"VoxTell: Free-Text Promptable Universal 3D Medical Image Segmentation"*, cs.CV, Nov 2025):
* **Scale:** Pre-trained on a massive multi-modality 3D medical corpus of **62K+ volumetric scans** spanning CT, MRI, and PET across **1,000+ anatomical and pathological concepts**.
* **Instance-Focused Findings Dataset:** Critically, the authors **already included the ReXGroundingCT training split** (comprising 2,992 chest CT scans from the CT-RATE corpus) in their pre-training corpus to handle spatially grounded queries.
* **Supervision Style:** VoxTell was trained using **fully-supervised standard Dice + BCE losses** computed over the entire volume.

---

## ⚠️ 2. The Partial Annotation / Instance Suppression Problem
While VoxTell has already seen the ReXGroundingCT training images, its training protocol introduced a severe systematic bias:
1. **Sparsely Labeled Ground Truth:** The ReXGroundingCT training set is *partially annotated*. A maximum of **3 instances** per finding are annotated in any volume, while other real instances of the same finding are left unannotated as "background".
2. **Naive Background Penalty:** Because VoxTell was trained with fully-supervised losses over the entire volume, it treated all unannotated instances of a finding as "negative background". The loss function actively penalized the model for segmenting unannotated findings.
3. **Instance Suppression:** Consequently, the pre-trained VoxTell baseline model **suppresses extra instances** and restricts its predictions to only a few instances per scan. It is incapable of performing the *exhaustive* segmentation required by the MICCAI challenge validation and test splits (where all instances must be segmented).

---

## 🚀 3. Phase 2 Methodological Solution: Positive-Unlabeled (PU) Fine-Tuning
To push past the baseline's average Dice ceiling (~0.2138 zero-shot validation), we are implementing a **Semi-Supervised Fine-Tuning (SPOCO + MPR Consistency) pipeline** designed specifically to resolve the partial annotation limitation:

### A. SPOCO ROI Masked Loss (Strong Supervision)
* **How it works:** We confine the supervised DiceCE loss strictly to a **dilated region of interest (ROI)** surrounding the ground-truth annotated instances.
* **Why it helps:** By masking out unannotated regions from the supervised loss, the Student model is **no longer penalized** for segmenting unannotated findings elsewhere in the volume.

### B. Mean Teacher Framework
* **How it works:** We maintain two models: the active **Student** and an Exponential Moving Average (EMA) **Teacher** model (smoothing momentum $\alpha \approx 0.999$).
* **Why it helps:** The Teacher provides stable, slowly-evolving target predictions that act as soft pseudo-labels, helping the Student generalize.

### C. MPR multi-planar Consistency Loss (Weak Supervision)
* **How it works:** In unannotated regions (outside the SPOCO ROIs), we compute **2D Max Projections** along the three anatomical axes (Axial, Coronal, Sagittal) for both Student and Teacher predictions. We minimize the discrepancy (MSE/L1) between these projections.
* **Why it helps:** Projections project 3D structures into 2D space. Computing consistency over max projections makes the loss highly sensitive to small, dispersed false positives. It guides the model to maintain spatial consistency and suppress random noise *without* forcing a hard-zero label, allowing the model to naturally segment all real findings in the volume.
