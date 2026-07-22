# ReXGroundingCT Challenge 2026 — Data & Baseline Inference Audit Pipeline

Repository for participating in the **ReXGrounding Challenge @ MICCAI 2026**. The primary task is 3D segmentation of radiological findings in thoracic CT scans from free-text descriptions (free-text finding grounding).

> [!IMPORTANT]
> **Active Research Focus**: All model fine-tuning (SPOCO, MPR consistency, Mean Teacher) is **POSTPONED**. The repository is 100% centered on:
> 1. **In-Depth ReXGroundingCT Data Analysis**: Profiling 3D CT images, ground-truth masks (sparse training set vs exhaustive validation set), 14 finding categories, component volumes, and NLP prompt syntax.
> 2. **VoxTell Zero-Shot Inference & Preprocessing Audit**: Evaluating official `NibabelIOWithReorient` and `VoxTellPredictor` execution, sliding window tile overlap, probability logit magnitudes, and per-category failure modes.

---

## 📂 Project Structure

```text
REX_PROJECT/
├── .agents/                    # Agentic rules, status, and governance documentation
│   ├── AGENTS.md               # Operating rules & Epistemic Modesty guidelines
│   ├── HANDSHAKE.md            # Active transitional context bridge
│   ├── STATUS.md               # Live project status
│   └── shared/                 # Master plan & challenge rules
├── data/                       # Data storage (4D volumes and metadata)
│   ├── predictions/            # Inference outputs (4D masks in F,H,W,D format)
│   ├── raw/                    # Original raw NIfTI volumes
│   └── dataset.json            # Dataset partitions (train/val/test) and text prompts
├── logs/                       # Execution logs and experiment reports
│   ├── execution_raw/          # Raw terminal stdout/stderr dumps
│   └── experiments/            # Documented experiment logs
│       ├── phase_1_baseline_profiling/   # Active zero-shot & prompt analysis reports
│       └── archived_proof_of_concept/    # Legacy preliminary experiment logs
├── models/                     # Checkpoints and pretrained weights
│   ├── voxtell_v1.0/           # Base VoxTell v1.0 checkpoint
│   └── voxtell_v1.1/           # Pretrained VoxTell v1.1 checkpoint
├── requirements/               # Dependency configurations
│   └── voxtell.txt             # PyTorch, MONAI, transformers, and nnUNetv2 requirements
├── scratch/                    # Diagnostic tools and analytical scripts
│   ├── phase_1_baseline_profiling/   # Active data analysis & inference audit scripts
│   └── archived_proof_of_concept/    # Archived legacy scratch tools
├── scripts/                    # Executable pipeline scripts
│   ├── voxtell/                # VoxTell inference pipeline and prompt normalizer
│   │   ├── voxtell_inference.py# Batch zero-shot inference pipeline
│   │   └── prompt_normalizer.py# Hybrid NLP prompt parsing module
│   ├── evaluate.py             # Official metric evaluator (Dice, Hit Rate >= 0.1)
│   └── archived_proof_of_concept/    # Archived legacy training scripts (train_mean_teacher.py)
├── .env                        # Environment variable configuration (GPU 1 isolation, paths)
└── README.md                   # Primary repository documentation
```

---

## ⚙️ Environment Setup & Hardware Isolation

The pipeline is pinned to **GPU 1** (`NVIDIA RTX PRO 6000 Blackwell`, 96 GB VRAM) on the host system using the upgraded `.venv-voxtell` Python 3.13 environment with CUDA 12.8 support.

### 1. Environment Configuration (`.env`)
The pipeline relies strictly on environment variables for path resolution:

```env
MODEL_DIR=/home/jdeferrari/rex_project/models/voxtell_v1.1
SEG_RAW_DIR=/home/jdeferrari/rex_project/data/raw/masks
IMG_RAW_DIR=/home/jdeferrari/rex_project/data/raw/images
DATA_PRED_DIR=/home/jdeferrari/rex_project/data/predictions
DATASET_JSON=/home/jdeferrari/rex_project/data/dataset.json

CUDA_VISIBLE_DEVICES=1
DEFAULT_DEVICE=cuda:0
```

---

## 🚀 Active Execution & Analysis Commands

### 1. NLP Prompt Text Shift Analysis
Run quantitative NLP distribution analysis comparing free-text prompt syntax, word length, and adjective modifiers:
```bash
./.venv-voxtell/bin/python scratch/phase_1_baseline_profiling/text_shift_analysis.py
```

### 2. Official Reader/Writer Pipeline Verification
Verify official `NibabelIOWithReorient` and `VoxTellPredictor` execution against raw ground-truth NIfTI masks:
```bash
CUDA_VISIBLE_DEVICES=1 ./.venv-voxtell/bin/python scratch/phase_1_baseline_profiling/verify_official_pipeline.py
```

### 3. Batch Zero-Shot VoxTell Inference
Run zero-shot inference across validation scans on GPU 1:
```bash
CUDA_VISIBLE_DEVICES=1 ./.venv-voxtell/bin/python scripts/voxtell/voxtell_inference.py --split val --tile_step_size 0.5
```

### 4. Fast Multi-Threaded Validation Metric Calculation
Compute primary ranking metrics (Average Dice, Hit Rate $\ge 0.1$, Empty Preds) across prediction volumes:
```bash
PYTHONPATH=. CUDA_VISIBLE_DEVICES=1 ./.venv-voxtell/bin/python scratch/phase_1_baseline_profiling/fast_200_eval.py
```

---

## 📝 Governance & Epistemic Modesty Guidelines

* **Epistemic Modesty**: All preliminary observations must use calibrated, modest phrasing (*"initial evidence suggests"*, *"preliminary tests indicate"*). Unproven fine-tuning methods are strictly framed as **hypotheses to be tested**.
* **SSD Storage Caching**: Heavy runtime inputs and intermediate volumes reside on the fast SSD `/tmp` directory (`/tmp/jdeferrari/rexgroundingct_preprocessed/`).
