import os
import json
import torch
import numpy as np
import nibabel as nib
from tqdm import tqdm
from dotenv import load_dotenv

# Cargar variables de entorno desde el archivo .env
load_dotenv()

# 1. Aislar estrictamente la GPU 0
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

from voxtell.inference.predictor import VoxTellPredictor
from nnunetv2.imageio.nibabel_reader_writer import NibabelIOWithReorient

def main():
    # Rutas base inyectadas desde .env
    base_dir = os.getenv("BASE_DIR")
    data_preprocessed_dir = os.getenv("DATA_PREP_DIR")
    json_path = os.getenv("DATASET_JSON")
    out_dir = os.getenv("DATA_PRED_DIR")
    
    # Pequeña validación de seguridad por si el .env no carga bien
    if not all([base_dir, data_preprocessed_dir, json_path, out_dir]):
        raise ValueError("Error: Faltan variables de entorno. Verifica tu archivo .env.")

    os.makedirs(out_dir, exist_ok=True)

    # 2. Cargar el dataset.json para extraer los prompts y nombres de archivo
    print(f"Cargando dataset metadata desde: {json_path}")
    with open(json_path, 'r') as f:
        dataset_info = json.load(f)
    
    # Extraemos el split de prueba (el evaluador oficial usa la key "test")
    test_entries = dataset_info.get("test", [])
    print(f"Total de casos a evaluar encontrados: {len(test_entries)}")

    # 3. Inicializar el modelo
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model_dir = os.getenv("MODEL_DIR")
    print("Cargando pesos de VoxTell en VRAM...")
    predictor = VoxTellPredictor(model_dir=model_dir, device=device)
    reader = NibabelIOWithReorient()

    # 4. Bucle de inferencia
    for entry in tqdm(test_entries, desc="Inferencia Zero-Shot"):
        # NOTA: Revisa si la clave de los textos en tu dataset.json es "findings", "prompts" o "texts"
        # y ajústala si es necesario.
        prompts = entry.get("findings", []) 
        
        # Extraemos solo el nombre del archivo para buscarlo en la carpeta preprocesada
        img_filename = os.path.basename(entry["image_path"]) 
        seg_filename = os.path.basename(entry["seg_path"])
        
        img_path = os.path.join(data_preprocessed_dir, img_filename)
        
        if not os.path.exists(img_path):
            print(f"Advertencia: Omitiendo {img_filename}, no encontrado.")
            continue

        # Inferencia
        img, _ = reader.read_images([img_path])
        pred_tensor = predictor.predict_single_image(img, prompts)
        
        # 5. Guardado NIfTI 4D
        # Robamos el 'affine' del CT preprocesado para asegurar perfecta alineación espacial
        original_nifti = nib.load(img_path)
        affine = original_nifti.affine
        
        # Forzamos np.uint8 como exige rexrank_eval.py para ahorrar memoria
        pred_nifti = nib.Nifti1Image(pred_tensor.astype(np.uint8), affine)
        out_path = os.path.join(out_dir, seg_filename) 
        nib.save(pred_nifti, out_path)

    print(f"Proceso finalizado. Predicciones guardadas en: {out_dir}")

if __name__ == "__main__":
    main()