# ReXGroundingCT Challenge 2026 — Data & Baseline Inference Audit Pipeline

Repository for participating in the **ReXGrounding Challenge @ MICCAI 2026**. The primary task is 3D segmentation of radiological findings in thoracic CT scans from free-text descriptions (free-text finding grounding).

> [!IMPORTANT]
> **Phased Research Roadmap**:
> 1. **Phase 1 — ReXGroundingCT Data Analysis**: Profiling 3D CT images, ground-truth masks (sparse training set vs exhaustive validation set), 14 finding categories, component volumes, and NLP prompt syntax.
> 2. **Phase 2 — VoxTell Zero-Shot Inference & Preprocessing Audit**: Evaluating official `NibabelIOWithReorient` and `VoxTellPredictor` execution, sliding window tile overlap, probability logit magnitudes, and per-category failure modes.
> 3. **Phase 3 — Model Fine-Tuning & Consistency Adaptations**: Adapting VoxTell weights with partial-annotation loss functions (SPOCO, MPR consistency, Mean Teacher).

---

## 📂 Project Structure

```text
REX_PROJECT/
├── .agents/                    # Agentic rules, host setup docs, and governance
│   ├── shared/                 # Server-agnostic master plan and paper digests
│   ├── AGENTS.md               # Repository operating rules & governance
│   ├── server_documentation.txt# Host server hardware setup & guides
│   ├── STATUS.md               # Local active state (untracked)
│   └── HANDSHAKE.md            # Local operational context bridge (untracked)
├── data/                       # Data storage (4D volumes and metadata)
├── logs/                       # Phase-organized logs and experiment reports
│   ├── phase_1_data_profiling/ # Phase 1 data analysis logs
│   ├── phase_2_inference_audit/# Phase 2 zero-shot inference audit logs
│   └── phase_3_fine_tuning/    # Phase 3 model fine-tuning logs
│       └── proof_of_concept/   # Early exploratory proof-of-concept logs
├── models/                     # Checkpoints and pretrained weights
├── scratch/                    # Phase-organized diagnostic scripts
│   ├── phase_1_data_profiling/ # Phase 1 prompt shift diagnostic scripts
│   ├── phase_2_inference_audit/# Phase 2 zero-shot evaluation tools
│   └── phase_3_fine_tuning/    # Phase 3 exploratory fine-tuning scripts
│       └── proof_of_concept/   # Exploratory training & evaluation tools
├── scripts/                    # Production pipeline scripts
│   ├── data_analysis/          # Dataset statistics & spatial heatmap generators
│   ├── data_prep/              # MONAI data preprocessing pipeline
│   ├── voxtell/                # VoxTell model training & inference pipelines
│   │   ├── training/           # Production fine-tuning & trainer modules
│   │   └── voxtell_inference.py# Zero-shot VoxTell inference pipeline
│   └── evaluate.py             # Official challenge metric calculator
└── README.md                   # Primary repository documentation
```

---

## ⚙️ Environment Setup & Hardware Isolation

The pipeline uses environment-based hardware isolation (`CUDA_VISIBLE_DEVICES`) to pin execution to target host GPUs. Specific hardware topology and server setup instructions are documented in `.agents/server_documentation.txt`.

### 1. Environment Configuration (`.env`)
The pipeline relies strictly on environment variables for path resolution:

```env
MODEL_DIR=./models/voxtell_v1.1
SEG_RAW_DIR=./data/raw/masks
IMG_RAW_DIR=./data/raw/images
DATA_PRED_DIR=./data/predictions
DATASET_JSON=./data/dataset.json

CUDA_VISIBLE_DEVICES=1
DEFAULT_DEVICE=cuda:0
```

---

## 🚀 Active Execution & Analysis Commands

### 1. Dataset Statistical Analysis & Spatial Heatmaps
Generate relative volume distributions, entity counts per finding, and 2D coronal/axial projection heatmaps across categories:
```bash
./.venv-voxtell/bin/python scripts/data_analysis/dataset_stats.py
./.venv-voxtell/bin/python scripts/data_analysis/mask_heatmaps.py
```

### 2. NLP Prompt Text Shift Analysis
Run quantitative NLP distribution analysis comparing free-text prompt syntax, word length, and adjective modifiers:
```bash
./.venv-voxtell/bin/python scratch/phase_1_data_profiling/text_shift_analysis.py
```

### 3. Official Reader/Writer Pipeline Verification
Verify official `NibabelIOWithReorient` and `VoxTellPredictor` execution against raw ground-truth NIfTI masks:
```bash
CUDA_VISIBLE_DEVICES=1 ./.venv-voxtell/bin/python scratch/phase_2_inference_audit/verify_official_pipeline.py
```

### 4. Batch Zero-Shot VoxTell Inference
Run zero-shot inference across validation scans:
```bash
CUDA_VISIBLE_DEVICES=1 ./.venv-voxtell/bin/python scripts/voxtell/voxtell_inference.py --split val --tile_step_size 0.5
```

### 5. Production Model Fine-Tuning
Run persistent Mean Teacher fine-tuning on target GPU:
```bash
WANDB_MODE=offline PYTHONUNBUFFERED=1 nohup .venv-voxtell/bin/python -u scripts/voxtell/training/train_mean_teacher.py --epochs 50 --wandb > logs/train_mean_teacher_50ep.log 2>&1 &
```

### 6. Fast Multi-Threaded Validation Metric Calculation
Compute primary ranking metrics (Average Dice, Hit Rate $\ge 0.1$, Empty Preds) across prediction volumes:
```bash
PYTHONPATH=. CUDA_VISIBLE_DEVICES=1 ./.venv-voxtell/bin/python scratch/phase_2_inference_audit/fast_200_eval.py
```

---

## 📝 Governance & Epistemic Modesty Guidelines

* **Epistemic Modesty**: All preliminary observations must use calibrated, modest phrasing (*"initial evidence suggests"*, *"preliminary tests indicate"*). Unproven fine-tuning methods are strictly framed as **hypotheses to be tested**.
* **Server-Agnostic Rules**: Repository-wide code and documentation in git remain strictly server-agnostic. Host-specific hardware topologies, GPU assignments, and paths are kept in local `.agents/` documentation.
* **SSD Storage Caching**: Heavy runtime inputs and intermediate volumes reside in fast local temporary storage specified in `.agents/server_documentation.txt`.
