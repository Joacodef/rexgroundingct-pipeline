# Pipeline para ReXGroundingCT Challenge 2026

Repositorio para la participación en el ReXGrounding Challenge @ MICCAI 2026. El objetivo principal es la segmentación 3D de hallazgos radiológicos a partir de descripciones de texto libre (free-text finding grounding).

Este pipeline implementa adaptaciones metodológicas avanzadas (Mean Teacher, MPR Loss, SPOCO) para lidiar con el problema de **anotaciones parciales** en el dataset de entrenamiento.

## 📂 Estructura del Proyecto

La arquitectura del repositorio está diseñada para aislar el entorno, los datos pesados y la configuración de ejecución.

```text
REX_PROJECT/
├── data/                       # Almacenamiento de datos (volúmenes 4D y metadatos)
│   ├── predictions/            # Outputs de inferencia (máscaras 4D en formato F,H,W,D)
│   ├── preprocessed/           # Dataset estandarizado (RAS, 1.5mm isotrópico)
│   ├── raw/                    # Volúmenes NIfTI originales
│   └── dataset.json            # Metadatos, particiones (train/val/test) y prompts
├── models/                     # Checkpoints y cachés
│   ├── .cache/                 # Caché de HuggingFace (ej. Text Encoders)
│   ├── voxtell_v1.0/           # Checkpoint base de VoxTell
│   ├── voxtell_v1.1/           # Checkpoint iterativo (fine-tuned)
│   └── config.json             # Configuración de hiperparámetros
├── notebooks/                  # Jupyter notebooks para EDA, sanity checks y visualizaciones
├── requirements/               # Arquitectura modular de dependencias
│   ├── base.txt                # Infraestructura, manipulación volumétrica (MONAI) y monitoreo
│   └── voxtell.txt             # Dependencias específicas del modelo, compilación CUDA y PyTorch
├── scripts/                    # Pipeline ejecutable
│   ├── data_prep/              # Pipeline de preprocesamiento (orientación, resampleo, clipping)
│   └── voxtell/                # Loops de inferencia, evaluación baseline y fine-tuning
├── .env                        # Variables de entorno y manejo seguro de rutas relativas
├── .gitignore                  # Exclusión estricta de entornos virtuales, NIfTIs y binarios
└── workspace.code-workspace    # Configuración de entorno de desarrollo (ej. VS Code)

```

## ⚙️ Configuración del Entorno (`uv`)

Este proyecto utiliza `uv` para la gestión aislada de paquetes y dependencias, garantizando la reproducibilidad matemática de las métricas en cualquier servidor o clúster.

1. **Instalar `uv**`:
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh

```


2. **Crear el entorno virtual aislado (VoxTell)**:
```bash
uv venv .venv-voxtell --python 3.10

```


3. **Instalar dependencias congeladas**:
```bash
uv pip install -r requirements/voxtell.txt --env-file .env

```



## 🚀 Pipeline de Ejecución

### 1. Preprocesamiento (Asegurar Formato)

Estandariza los volúmenes crudos al espacio de coordenadas esperado por el modelo.

```bash
./.venv-voxtell/bin/python scripts/data_prep/preprocess.py

```

### 2. Inferencia Batch (Baseline)

Ejecución del modelo zero-shot sobre el validation set.

> 💡 **Recomendación:** Para inferencias o entrenamientos sobre datasets completos en servidores remotos, se recomienda utilizar multiplexores de terminal (`screen` o `tmux`).

```bash
# Ejecución utilizando el binario aislado del entorno
CUDA_VISIBLE_DEVICES=0 ./.venv-voxtell/bin/python scripts/voxtell/run_baseline.py

```

### 3. Evaluación Estricta

Cálculo de métricas contra las anotaciones exhaustivas. (Target baseline: Dice global ~0.285).

```bash
./.venv-voxtell/bin/python rexrank_eval.py \
  --gt_dir data/preprocessed \
  --pred_dir data/predictions \
  --output_json data/eval_results.json \
  --dataset_json data/dataset.json

```

## 📝 Consideraciones Operativas

* **Manejo de I/O:** El procesamiento de NIfTIs 4D es altamente intensivo en lectura/escritura. Se recomienda encarecidamente utilizar sistemas de archivos rápidos (SSD/NVMe) o ramdisks (`/tmp` en entornos Linux) para interactuar con las carpetas de datos durante el *runtime*.
* **Entrenamiento Distribuido:** Si se ejecuta DDP (Distributed Data Parallel) en clústeres compartidos, es vital asignar explícitamente puertos libres (ej. `MASTER_PORT=29501`) en los scripts de lanzamiento para evitar colisiones de red con otros usuarios.