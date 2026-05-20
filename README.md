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
│   └── voxtell/                
│       └── voxtell_inference.py # Batch Zero-Shot inference and 4D stacking
├── .env                        # Environment variables and secure relative path handling
├── .gitignore                  # Strict exclusion of virtual environments, NIfTIs and binaries
└── workspace.code-workspace    # Development environment configuration (e.g. VS Code)

```

## ⚙️ Environment Configuration (`uv`)

This project uses `uv` for isolated package and dependency management, ensuring mathematical reproducibility of metrics on any server or cluster.

1. **Install `uv**`:

```bash
curl -LsSf [https://astral.sh/uv/install.sh](https://astral.sh/uv/install.sh) | sh

```

2. **Create the isolated virtual environment (VoxTell)**:

```bash
uv venv .venv-voxtell --python 3.10

```

3. **Install frozen dependencies**:

> ⚠️ **Note:** `uv` only searches the first index that contains a given package by default. The `--index-strategy unsafe-best-match` flag is required so that packages like `torch` are resolved from the PyTorch index while the rest fall back to PyPI. The `--no-verify-hashes` flag is needed because the PyTorch wheel index does not publish hash metadata.

```bash
uv pip install \
  --no-verify-hashes \
  --extra-index-url [https://download.pytorch.org/whl/cu126](https://download.pytorch.org/whl/cu126) \
  --index-strategy unsafe-best-match \
  -r requirements/base.txt \
  -r requirements/voxtell.txt

```

## 🚀 Execution Pipeline

### 1. Environment Setup (.env)

The pipeline relies strictly on environment variables for path injection to ensure portability. Ensure your `.env` file at the root of the repository contains the following structure before execution:

```env
# Base paths
MODEL_DIR=/absolute/path/to/models/voxtell_v1.0
DATA_PREP_DIR=/absolute/path/to/data/preprocessed
DATA_PRED_DIR=/absolute/path/to/data/predictions
DATASET_JSON=/absolute/path/to/data/dataset.json

# GPU Isolation (Defaults to 0)
CUDA_VISIBLE_DEVICES=0
DEFAULT_DEVICE=cuda:0

```

### 2. Preprocessing (Format Assurance)

Standardizes raw volumes to the coordinate space expected by the model.

```bash
./.venv-voxtell/bin/python scripts/data_prep/preprocess.py

```

### 3. Batch Inference (Baseline)

Zero-shot model execution on the validation set. Generates strictly aligned 4D NIfTI masks `(F, H, W, D)`.

> 💡 **Recommendation:** For inference or training on full datasets on remote servers, it is highly recommended to use terminal multiplexers (`screen` or `tmux`).

```bash
# Execution using the environment's isolated binary
./.venv-voxtell/bin/python scripts/voxtell/voxtell_inference.py

```

### 4. Strict Evaluation

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
* **Execution Environments:** For inference or training on Jumbito, use terminal multiplexers (`screen` or `tmux`). For SLURM-based clusters, wrap the Python calls in `sbatch` submission scripts and utilize node-local scratch directories (`$SLURM_TMPDIR`).
* **Distributed Training:** If running DDP on shared clusters, it is vital to explicitly assign free ports (e.g. `MASTER_PORT=$((RANDOM % 10000 + 20000))`) in the launch scripts to avoid network collisions with other users.
