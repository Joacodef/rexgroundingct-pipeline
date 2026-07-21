# Master Plan — ReXGroundingCT Challenge 2026

**Primary Goal:** Top-3 on the leaderboard (September 2026) AND an original paper accepted at MICCAI 2026.

---

## 🔬 1. Methodological Strategy: Step-by-Step Empirical Approach

The previous strategy attempted a complex architecture (Mean Teacher + SPOCO + MPR Loss) that lacked empirical justification for each component. The new roadmap shifts to a strictly empirical, step-by-step methodology, focusing first on baseline error diagnosis before implementing any modifications.

1. **Untouchable Baseline:** VoxTell v1.1 will be tested as the initial hypothesis.
2. **Micro-Experiments:** We will test isolated, measured improvements (e.g., modifying the loss function slightly to measure suppression bias) rather than massive architectural changes.
3. **Data-Centric Focus:** Instead of model architecture overhauls, we will explore post-processing, thresholding, and naive pseudo-labeling techniques on partial annotations.
4. **Architectural Pivot (Contingency):** If error profiling reveals that VoxTell's pre-trained embeddings are fundamentally broken due to the "Instance Suppression Bias", we will pivot to building a lightweight text-conditioned model from scratch (e.g., nnU-Net).

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

### Phase 1: Baseline Error Profiling & Alternative Scouting (July) - 🟢 ACTIVE
*   **Quantitative Error Analysis:** Bucket errors by finding size, finding type, and anatomical region.
*   **Qualitative Error Analysis:** Visually inspect the worst-performing volumes to identify systematic biases (e.g., hallucinated masks, boundary issues, or Instance Suppression Bias).
*   **Alternative Model Scouting:** Identify and evaluate "pristine" foundation models (e.g., MedSAM, text-conditioned nnU-Net) that have never been fine-tuned on ReXGroundingCT masks, ensuring we have a clean slate ready for a potential pivot.
*   **Goal:** Decide if VoxTell is salvageable or if a pivot to the alternative pristine model is required.

### Phase 2: Micro-Experiments & Data-Centric Tweaks (August - September) - ⏳ PENDING
*   **Experiment A:** Train VoxTell with standard DiceCE to measure exact degradation caused by suppression bias.
*   **Experiment B:** Test thresholding, connected component filtering, or morphological operations.
*   **Experiment C:** Naive pseudo-labeling using the baseline model to fill in missing annotations, followed by standard training.

### Phase 3: Methodological Re-Evaluation (October) - ⏳ PENDING
*   Re-evaluate Gao/Wolny (SPOCO/MPR) techniques. Implement one at a time if necessary.
*   Explore simpler Positive-Unlabeled (PU) learning techniques as alternatives.

### Phase 4: Scaling and Paper Writing (November - December) - ⏳ PENDING
*   Scale the winning combination of micro-experiments to the full dataset on SLURM.
*   Perform hyperparameter sweeps on the stabilized architecture.
*   Draft the MICCAI paper centered on systematically evaluating partial-annotation solutions.
