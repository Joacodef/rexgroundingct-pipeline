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

## ⚙️ Environment Configuration

Depending on which server environment you are running, choose the appropriate setup guide below.

### A. Official Cluster Setup (`ih-condor` SLURM + Conda)
This server uses Environment Modules and pre-installed Conda environments.

1. **Load and Activate Conda:**
   ```bash
   module load conda
   conda activate /mnt/workspace/jdeferrari/.conda/envs/voxtell
   ```
2. **Package Submissions (SLURM):**
   All resource-intensive experiments (preprocessing, training, inference) **must** be submitted as SLURM jobs. Use the scripts in `bash_scripts/`:
   * To preprocess data: `sbatch bash_scripts/preprocess.sh`
   * To train: `sbatch bash_scripts/train.sh`

---

### B. Local / Alternative Server Setup (`uv`)
For local systems or other servers that do not run SLURM/Conda, `uv` is used for isolated virtual environments.

1. **Install `uv`**:
   ```bash
   curl -LsSf https://astral.sh/uv/install.sh | sh
   ```
2. **Create the isolated virtual environment**:
   ```bash
   uv venv .venv-voxtell --python 3.10
   ```
3. **Install frozen dependencies**:
   ```bash
   uv pip install \
     --no-verify-hashes \
     --extra-index-url https://download.pytorch.org/whl/cu126 \
     --index-strategy unsafe-best-match \
     -r requirements/base.txt \
     -r requirements/voxtell.txt
   ```


## 🚀 Execution Pipeline

### 1. Environment Setup (.env)

The pipeline relies strictly on environment variables for path injection to ensure portability. To configure your environment:

1. Copy the provided template to create your local environment file:
   ```bash
   cp .env.example .env
   ```
2. Open `.env` and configure the absolute paths for your local directory layout and GPU hardware:
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

### 2. Preprocessing (Format Assurance & Native Resolution)

> [!IMPORTANT]
> **Note for VoxTell v1.1:** The `preprocess.py` script (which performs resampling to 1.5mm isotropic spacing and lung clipping) is deprecated for VoxTell v1.1 evaluation. Recent modifications proved that 1.5mm resampling severely degrades the Average Dice (reducing it to `~0.08`). 
> 
> Instead, the current VoxTell v1.1 pipeline performs inference **directly in native resolution** and applies a mathematical **4D Back-Reorientation** mechanism to align predictions with the Ground Truth.
> 
> To fully understand the theoretical details of this behavior, the Out-of-Distribution (OOD) Shift, and the implemented mathematical solution, please refer to the guide:
> 📄 **[VoxTell Preprocessing and Spatial Alignment Challenges](docs/voxtell_preprocessing_challenges.md)**

If for any reason you need to run the historic 1.5mm resampled preprocessing:

```bash
./.venv-voxtell/bin/python scripts/data_prep/preprocess.py
```

### 3. Batch Inference (Baseline)

Zero-shot model execution on the validation set. Generates strictly aligned 4D NIfTI masks `(F, H, W, D)`.

> 💡 **Recommendation:** For inference or training on full datasets on remote servers, it is highly recommended to use terminal multiplexers (preferably `tmux`).

```bash
# Execution using the environment's isolated binary
./.venv-voxtell/bin/python scripts/voxtell/voxtell_inference.py

```

### 4. Strict Evaluation

Calculation of metrics against exhaustive annotations. (Target baseline: Global Dice ~0.285).

```bash
./.venv-voxtell/bin/python scripts/evaluate.py

```

## 📝 Operational Considerations

* **I/O Handling:** 4D NIfTI processing is highly read/write intensive. It is strongly recommended to use fast file systems (SSD/NVMe) or ramdisks (`/tmp` in Linux environments) to interact with data folders during *runtime*.
* **Execution Environments:** For inference or training on Jumbito, use terminal multiplexers (`tmux` is preferred). For SLURM-based clusters, wrap the Python calls in `sbatch` submission scripts and utilize node-local scratch directories (`$SLURM_TMPDIR`).
* **Distributed Training:** If running DDP on shared clusters, it is vital to explicitly assign free ports (e.g. `MASTER_PORT=$((RANDOM % 10000 + 20000))`) in the launch scripts to avoid network collisions with other users.
