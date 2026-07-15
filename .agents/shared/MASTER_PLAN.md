# Master Plan — ReXGroundingCT Challenge 2026

**Primary Goal:** Top-3 on the leaderboard (September 2026) AND an original paper accepted at MICCAI 2026.

---

## 🔬 1. Methodological Strategy (The Paper's "Contribution")

The training dataset has partial annotations (maximum of 3 instances per finding), whereas validation and test phases require exhaustive segmentation. To leverage this without being penalized by models that assume complete masks:

1. **Baseline:** VoxTell v1.1 (Frozen Text Encoder, decoder fine-tuning).
2. **Strong Supervision (ROIs):** Apply DiceCE loss only within dilated regions around the annotated instances (adapting the SPOCO principle from Wolny et al. 2022).
3. **Weak Supervision (Unannotated regions):** Implement a Mean Teacher framework. Apply MPR loss (SOUSA, Gao et al. 2022) over 3D max projections to penalize student false positives in unannotated areas, ensuring spatial consistency.

---

## 💻 2. Operational Constraints and Considerations

The pipeline is deployed across three environments: `jumbito` (compute.pln.villena.cl) and two SLURM-managed clusters.

### Execution and Persistence 
* **On `jumbito`:** In the absence of a queue manager, there is no job time limit. Long runs must use `tmux` (preferred over `screen`) or `nohup` (persistent execution).
* **On SLURM clusters:** Jobs are subject to queue wait times and strict wall-clock limits. The training loop must include checkpoint saving and resumption to recover from preemptions.

### I/O and Storage Management
* Massive 4D NIfTI processing must be isolated in fast, local filesystems (`/tmp` on `jumbito`; node-local scratch like `$SLURM_TMPDIR` on SLURM). 
* Temporary filesystems have automated deletion policies, so final artifacts must be synchronized to permanent storage at the end of every run.

### Execution Environment
* Use `uv` and virtual environments for strict dependency isolation. 
* If containerizing, note that SLURM clusters typically require Apptainer/Singularity, whereas `jumbito` supports rootless Docker.

### Multi-GPU Control (DDP)
* Resource allocation must be manual using `CUDA_VISIBLE_DEVICES`. 
* To avoid the `Address already in use` socket error when other users run distributed tasks, a dynamic or non-default `MASTER_PORT` (other than 29500) must be assigned in the launch scripts.

### Submission Format
* Mandatory 4D NIfTI identical to the original ground truth. 
* Post-inference stacking and **4D Back-Reorientation** must be robust to align with the original CT coordinate space.

---

## 🗓️ 3. Project Timeline & Status (Updated: July 2026)

### Phase 1: Securing Baseline and Methodology (May - June) - ✅ COMPLETED
* **Setup:** Environment secured with frozen dependencies on `jumbito`.
* **Preprocessing:** Bypassed/deprecated 1.5mm resampling. Execution runs strictly in native CT resolution.
* **Spatial Alignment:** Addressed the Identity Affine metadata bug in the Ground Truth segmentations via 4D Back-Reorientation.
* **Baseline Inference:** Native resolution batch inference with VoxTell v1.1 achieved a baseline of `~0.2138` Dice score and `~48.70%` Hit Rate.

### Phase 2: Mean Teacher Stabilization (Experiment 003) - ✅ COMPLETED
* **Integration:** Integrated the Teacher model's EMA update into the VoxTell training loop.
* **Stabilization:** Successfully diagnosed and patched the `fp16` underflow bug (NaN crash at Epoch 26) by applying L2 Gradient Norm Clipping, float32 upcasting, and softening consistency scaling.
* **Result:** Reached Epoch 50 successfully without NaNs. Model weights are generalized and ready for inference.

### Phase 3: Post-Launch Challenge Phase (July - September) - 🟢 ACTIVE
* **Official Evaluation (Task C):** Generate the first submission over the test split using the generalized weights from Experiment 003 and evaluate against the official leaderboard.
* **Ablation Studies:** Train variants by disabling the consistency loss or varying the weights to test the methodological impact for the paper.
* **Ensembling:** Combine the hybrid model with standard fine-tuned versions or average multi-scale inferences for the final submission.
