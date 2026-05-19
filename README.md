# Pipeline for ReXGroundingCT Challenge 2026

Repository for participating in the ReXGrounding Challenge @ MICCAI 2026. The main objective is the 3D segmentation of radiological findings from free-text descriptions (free-text finding grounding).

This pipeline implements advanced methodological adaptations (Mean Teacher, MPR Loss, SPOCO) to handle the problem of **partial annotations** in the training dataset.

## 📂 Project Structure

The repository architecture is designed to isolate the environment, heavy data, and execution configuration.

```text
REX_PROJECT/
├── data/                       # Data storage (4D volumes and metadata)
│   ├── predictions/            # Inference outputs (4D masks in F,H,W,D format)
│   ├── preprocessed/           # Standardized dataset (RAS, 1.5mm isotropic)
│   ├── raw/                    # Original NIfTI volumes
│   └── dataset.json            # Metadata, partitions (train/val/test) and prompts
├── models/                     # Checkpoints and caches
│   ├── .cache/                 # HuggingFace cache (e.g. Text Encoders)
│   ├── voxtell_v1.0/           # VoxTell base checkpoint
│   ├── voxtell_v1.1/           # Iterative checkpoint (fine-tuned)
│   └── config.json             # Hyperparameter configuration
├── notebooks/                  # Jupyter notebooks for EDA, sanity checks and visualizations
├── requirements/               # Modular dependency architecture
│   ├── base.txt                # Infrastructure, volumetric manipulation (MONAI) and monitoring
│   └── voxtell.txt             # Model-specific dependencies, CUDA and PyTorch compilation
├── scripts/                    # Executable pipeline
│   ├── data_prep/              # Preprocessing pipeline (orientation, resampling, clipping)
│   └── voxtell/                # Inference loops, baseline evaluation and fine-tuning
├── .env                        # Environment variables and secure relative path handling
├── .gitignore                  # Strict exclusion of virtual environments, NIfTIs and binaries
└── workspace.code-workspace    # Development environment configuration (e.g. VS Code)
```

## ⚙️ Environment Configuration (`uv`)

This project uses `uv` for isolated package and dependency management, ensuring mathematical reproducibility of metrics on any server or cluster.

1. **Install `uv`**:
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

2. **Create the isolated virtual environment (VoxTell)**:
```bash
uv venv .venv-voxtell --python 3.10
```

3. **Install frozen dependencies**:
```bash
uv pip install -r requirements/voxtell.txt --env-file .env
```

## 🚀 Execution Pipeline

### 1. Preprocessing (Format Assurance)

Standardizes raw volumes to the coordinate space expected by the model.

```bash
./.venv-voxtell/bin/python scripts/data_prep/preprocess.py
```

### 2. Batch Inference (Baseline)

Zero-shot model execution on the validation set.

> 💡 **Recommendation:** For inference or training on full datasets on remote servers, it is highly recommended to use terminal multiplexers (`screen` or `tmux`).

```bash
# Execution using the environment's isolated binary
CUDA_VISIBLE_DEVICES=0 ./.venv-voxtell/bin/python scripts/voxtell/run_baseline.py
```

### 3. Strict Evaluation

Calculation of metrics against exhaustive annotations. (Target baseline: Global Dice ~0.285).

```bash
./.venv-voxtell/bin/python rexrank_eval.py \
  --gt_dir data/preprocessed \
  --pred_dir data/predictions \
  --output_json data/eval_results.json \
  --dataset_json data/dataset.json
```

## 📝 Operational Considerations

* **I/O Handling:** 4D NIfTI processing is highly read/write intensive. It is strongly recommended to use fast file systems (SSD/NVMe) or ramdisks (`/tmp` in Linux environments) to interact with data folders during *runtime*.
* **Distributed Training:** If running DDP (Distributed Data Parallel) on shared clusters, it is vital to explicitly assign free ports (e.g. `MASTER_PORT=29501`) in the launch scripts to avoid network collisions with other users.
