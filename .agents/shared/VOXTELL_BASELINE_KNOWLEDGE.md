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

## ⚠️ 2. The Partial Annotation / Instance Suppression Hypothesis
Initial observations and training set metadata suggest a potential systematic bias:
1. **Sparsely Labeled Ground Truth:** The ReXGroundingCT training set is *partially annotated*. A maximum of **3 instances** per finding are annotated in any volume, while other real instances of the same finding are left unannotated as "background".
2. **Naive Background Penalty Hypothesis:** Because VoxTell was pre-trained with fully-supervised losses over entire volumes, it may have treated unannotated instances of a finding as "negative background". We hypothesize that the loss function penalized the model for segmenting unannotated findings.
3. **Instance Suppression Hypothesis:** Consequently, we hypothesize that pre-trained VoxTell baseline weights may suppress additional finding instances. Empirical testing during Phase 2 fine-tuning is required to measure whether weight updates resolve this behavior on MICCAI validation masks.

---

## 🔬 3. Phase 2 Working Hypotheses: Semi-Supervised Fine-Tuning Strategies

To evaluate whether fine-tuning model weights moves performance beyond the zero-shot baseline, we propose testing the following hypotheses empirically on GPU 1:

### A. SPOCO ROI Masked Loss Hypothesis
* **Hypothesis:** Confining supervised loss strictly to a dilated region of interest (ROI) surrounding ground-truth annotated instances will prevent penalizing predictions on unannotated findings elsewhere in the volume.

### B. Mean Teacher Framework Hypothesis
* **Hypothesis:** Maintaining an Exponential Moving Average (EMA) Teacher model will provide stable pseudo-labels to guide Student weight updates without divergence.

### C. MPR Multi-Planar Consistency Loss Hypothesis
* **Hypothesis:** Computing 2D max projection consistency across Axial, Coronal, and Sagittal views in unannotated regions will act as a spatial regularizer, helping suppress random noise without enforcing rigid zero background labels.

