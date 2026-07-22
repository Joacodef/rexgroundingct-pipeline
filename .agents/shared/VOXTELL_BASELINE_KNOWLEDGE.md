# VoxTell Baseline Knowledge & Inference Audit Reference

> [!IMPORTANT]
> This document details key specifications of the pre-trained **VoxTell** baseline model (`voxtell_v1.1`) and guidelines for its zero-shot inference audit (Phase 2), providing the foundation for Phase 3 model fine-tuning.

---

## 🏁 1. VoxTell Pre-Training Distribution
According to the official VoxTell paper (*"VoxTell: Free-Text Promptable Universal 3D Medical Image Segmentation"*, cs.CV, Nov 2025):
* **Scale:** Pre-trained on a massive multi-modality 3D medical corpus of **62K+ volumetric scans** spanning CT, MRI, and PET across **1,000+ anatomical and pathological concepts**.
* **Instance-Focused Findings Dataset:** Pre-trained using portions of CT-RATE for spatially grounded queries.
* **Supervision Style:** Trained using fully-supervised standard Dice + BCE losses computed over volumes.

---

## 🔍 2. Inference & Preprocessing Specifications

* **Official I/O Reader**: `nnunetv2.imageio.nibabel_reader_writer.NibabelIOWithReorient`.
* **Input Image Orientation**: Standard RAS space.
* **Text Prompt Format**: Passed through instruction template `Instruct: Given an anatomical term query, retrieve the precise anatomical entity and location it represents\nQuery: {text}`.
* **Predictor Architecture**: Uses BioClinicalBERT text encoder, 3D ResEncoder, and MaskFormer decoder stages.

---

## 🔬 3. Active Inference Audit Objectives

1. **Category-Level Performance Profiling**: Evaluate raw zero-shot performance across the 14 official ReXGroundingCT finding categories.
2. **Logit Magnitude Analysis**: Inspect continuous sigmoid probabilities prior to `> 0.5` binarization to measure confidence distributions.
3. **Sliding Window Sensitivity**: Evaluate the impact of tile step size (`tile_step_size`) on spatial coverage and boundary accuracy.
