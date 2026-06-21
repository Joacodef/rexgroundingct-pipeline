# Experiment Log 004: VoxTell v1.1 Exhaustive Multi-Finding Grounding [DRAFT]

> [!NOTE]
> **PARALLEL SERVER EXECUTION NOTE:**  
> This project is executed across two parallel environments:
> *   **`jumbito`**: Direct host execution using `.venv-voxtell` and `nohup`/`tmux`. GPU isolation is managed via `CUDA_VISIBLE_DEVICES`.
> *   **`ih-condor`**: SLURM-managed cluster where execution must run via `sbatch` / `srun` under the conda environment `voxtell`. Checkpointing is mandatory to survive queue preemption limits. 

* **Date:** Proposed for June 2, 2026  

* **Project Milestone:** Milestone 2 (June 15, 2026) - Exhaustive Multi-Concept Grounding Scaling  
* **Status:** Proposed / Draft

---

## 1. Objective & Hypothesis
In **Experiment 003**, we successfully stabilized the training loop by adding gradient clipping, class weighting, and consistency warmup. However, due to VRAM resource constraints on the Ada GPUs, we restricted the dataset to process only **one finding per volume** (`max_f = 1`). While this successfully prevents CUDA OOM, it restricts the network's cross-attention transformer decoder from learning joint multi-finding grounding during a single forward pass.

### Context:
In the ReXGroundingCT test split, volumes can contain multiple pathology prompts simultaneously (e.g. "atelectasis", "effusion", and "nodule"). A model trained to ground only one finding per volume has no cross-concept context to separate overlapping attention maps.

### Hypothesis:
By leveraging the **58 GB of free VRAM** on our newly integrated Blackwell GPU 1, we can remove the `max_f = 1` constraint and scale to **exhaustive multi-finding grounding** (up to `max_f = 5` simultaneous findings per scan). This will allow the Student model to learn rich, joint multi-class segmentations. 

We will use **Gradient Accumulation** to maintain a stable virtual batch size and prevent OOM spikes.

---

## 2. Proposed Experimental Setup
* **Model Baseline:** Pre-trained weights initialized from the final stabilized checkpoint of **Experiment 003** (`models/checkpoint_mean_teacher_final.pth`).
* **Multi-Finding Scale:** Scale `max_f` from `1` up to `5` (covers 98% of multi-finding volumes in the ReXGroundingCT corpus).
* **Gradient Accumulation:** Accumulate gradients over `4` steps to maintain a virtual batch size of `4` volumes, mitigating activation spikes during multi-finding backpropagation.
* **Hardware Isolation:** GPU 1 (RTX PRO 6000 Blackwell; `CUDA_VISIBLE_DEVICES=2`) utilizing the upgraded PyTorch `2.11.0+cu128` CUDA 12.8 environment.
* **Estimated VRAM Footprint:** ~24.5 GB VRAM (well within the Blackwell's 58 GB limit).

---

## 3. Required Code Modifications

### A. Add `--max-findings` to training arguments:
We will add a dynamic command-line parameter in [`scripts/voxtell/train_mean_teacher.py`](file:///home/jdeferrari/rex_project/scripts/voxtell/train_mean_teacher.py) to control the findings cap:
```python
parser.add_argument("--max-findings", type=int, default=5, 
                    help="Maximum number of simultaneous findings per volume (default: 5)")
```

### B. Update Dataloader and Validation Predictor:
Pass `args.max_findings` dynamically to `ReXDataset`:
```python
train_dataset = ReXDataset(..., max_findings=args.max_findings)
val_dataset = ReXDataset(..., max_findings=args.max_findings)
```

### C. Implement Gradient Accumulation:
Update the training step inside the loop to accumulate loss before updating parameters:
```python
# Scale loss by accumulation steps
loss = (loss_sup + w_con * loss_con) / args.accumulation_steps
loss.backward()

if (step + 1) % args.accumulation_steps == 0:
    torch.nn.utils.clip_grad_norm_(student_model.parameters(), max_norm=1.0)
    optimizer.step()
    optimizer.zero_grad()
    update_ema_variables(student_model, teacher_model, args.alpha)
```

---

## 4. Verification Plan

### Step 1: Memory & Execution Verification (Local Smoke Test)
Run a fast 5-epoch smoke test on GPU 1 with `--max-findings 5`:
```bash
.venv-voxtell/bin/python scripts/voxtell/train_mean_teacher.py --smoke-test --max-findings 5
```
*   Verify that peak VRAM stays below 30 GB.
*   Verify that the multi-angle projections (MPR) consistency loss handles multiple findings channels without shape mismatch errors.

### Step 2: Local Validation Comparison
Run a full validation evaluation on the 50 validation scans to compare results:
*   **Target metric:** Expect a substantial jump in **Average Dice** and **Hit Rate** over the single-finding Experiment 003 checkpoint due to rich cross-attention contextualization.

---

## 5. Timeline & Milestones
* **Launch Date:** Pending (Delayed due to Experiment 003 restart). The previous Exp 003 run was aborted at Epoch 8 due to a sliding window fp16 precision underflow bug (which caused false 0.0 Dice scores). A fresh 50-epoch run of Experiment 003 has been initiated. Exp 004 will launch immediately after its conclusion.
* **Monitoring:** Real-time logging of multi-class train/val losses to Weights & Biases.
* **Milestone 2 Target:** Reach **`0.32` Average Dice** on validation scans by June 15, 2026.
