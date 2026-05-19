import os
import json
from tqdm import tqdm
from dotenv import load_dotenv
from monai.transforms import Compose, LoadImaged, Orientationd, Spacingd, SaveImage, EnsureChannelFirstd
from monai.data import Dataset, DataLoader, decollate_batch

# Cargar variables de entorno desde el archivo .env
load_dotenv()

# Rutas dinámicas
DATASET_JSON = os.getenv("DATASET_JSON")

# Nota: Verifica que las carpetas internas coincidan con 'images' y 'segmentations'
IMG_DIR = os.getenv("IMG_RAW_DIR") 
SEG_DIR = os.getenv("SEG_RAW_DIR")

TMP_PREP_DIR = os.getenv("TMP_PREP_DIR")
DATA_PREP_DIR = os.getenv("DATA_PREP_DIR")

if TMP_PREP_DIR:
    # Modo Jumbito: La variable volátil existe. Se usa para I/O rápido.
    OUT_DIR = TMP_PREP_DIR
    print(f"[INFO] Modo Jumbito detectado. Escribiendo tensores en espacio volátil: {OUT_DIR}")
elif DATA_PREP_DIR:
    # Modo ih-condor: TMP_PREP_DIR fue eliminada del .env, se escribe directo al SSD del investigador.
    OUT_DIR = DATA_PREP_DIR
    print(f"[INFO] Modo ih-condor detectado. Escribiendo tensores en almacenamiento persistente: {OUT_DIR}")
else:
    raise ValueError("Error de configuración: No se detectó TMP_PREP_DIR ni DATA_PREP_DIR en el .env local.")

os.makedirs(OUT_DIR, exist_ok=True)

def main():
    # 1. Lectura estructurada del dataset.json
    with open(DATASET_JSON, 'r') as f:
        metadata = json.load(f)
    
    train_entries = metadata.get("train", [])
    
    # 2. Mapeo de rutas y extracción de num_findings (Dimensión F)
    data_dicts = []
    for entry in train_entries:
        # En caso de que un volumen no tenga findings, evitamos que falle el len()
        num_f = len(entry.get("findings", {})) 
        
        data_dicts.append({
            "image": os.path.join(IMG_DIR, entry["name"]), 
            "label": os.path.join(SEG_DIR, entry["name"]),
            "num_findings": num_f
        })

    # 3. Pipeline de transformaciones espaciales
    # LoadImaged con ensure_channel_first=True es crítico:
    # - Para el CT (3D), añade la dimensión de canal (1, H, W, D).
    # - Para la máscara (4D), Nibabel lee (H, W, D, F) y MONAI lo permuta a (F, H, W, D).
    preprocessing_pipeline = Compose([
        # Carga cruda sin reordenamiento automático
        LoadImaged(keys=["image", "label"], reader="NibabelReader"),
        
        # Agrega el canal [1, H, W, D] SOLO a la imagen CT. 
        # La máscara ya tiene su F en el índice 0.
        EnsureChannelFirstd(keys=["image"]), 
        
        Orientationd(keys=["image", "label"], axcodes="RAS"),
        Spacingd(
            keys=["image", "label"], 
            pixdim=(1.5, 1.5, 1.5), 
            mode=["bilinear", "nearest"]
        )
    ])

    dataset = Dataset(data=data_dicts, transform=preprocessing_pipeline)
    dataloader = DataLoader(dataset, batch_size=1, num_workers=4)

    # 4. Transformaciones de guardado (Desacopladas del pipeline en memoria)
    # resample=False asegura que no se intente revertir el Spacingd
    save_img = SaveImage(
        output_dir=OUT_DIR, 
        output_postfix="ct", 
        output_ext=".nii.gz", 
        resample=False,
        separate_folder=False
    )
    save_seg = SaveImage(
        output_dir=OUT_DIR, 
        output_postfix="seg", 
        output_ext=".nii.gz", 
        resample=False,
        separate_folder=False
    )

    print(f"Iniciando preprocesamiento batch de {len(data_dicts)} volúmenes hacia {OUT_DIR}...")
    
    # 5. Ejecución y validación
    for batch in tqdm(dataloader, desc="Preprocesando scans", unit="scan"):
        for data in decollate_batch(batch):
            f_esperado = data["num_findings"]
            f_real = data["label"].shape[0]
            
            # Validación dura de la dimensionalidad (F, H, W, D) para compatibilidad con VoxTell
            assert f_real == f_esperado, (
                f"Error dimensional en {data['image'].meta['filename_or_obj']}: "
                f"dataset.json declara {f_esperado} findings, pero la máscara cargada tiene {f_real} canales."
            )
            
            save_img(data["image"])
            save_seg(data["label"])

if __name__ == "__main__":
    main()