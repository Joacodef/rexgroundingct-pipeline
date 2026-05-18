import os
import torch
import numpy as np
from huggingface_hub import snapshot_download
from dotenv import load_dotenv

# Cargar variables de entorno desde el archivo .env
load_dotenv()

# 1. Aislar estrictamente la GPU 0 (Política del nodo)
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# Importar dependencias de VoxTell después de setear las variables de entorno
from voxtell.inference.predictor import VoxTellPredictor
from nnunetv2.imageio.nibabel_reader_writer import NibabelIOWithReorient

def main():
    # Inyectar rutas desde el archivo .env
    download_dir = os.getenv("MODEL_DIR")
    data_prep_dir = os.getenv("DATA_PREP_DIR")
    output_dir = os.getenv("DATA_PRED_DIR")

    # Validación de seguridad para variables críticas de entorno
    if not all([download_dir, data_prep_dir, output_dir]):
        raise ValueError("Error: Faltan variables de entorno en el script. Verifica tu archivo .env.")

    # 2. Configuración de Dispositivo
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Dispositivo seleccionado: {device}")

    # Descargaremos la versión del paper (v1.0)
    MODEL_NAME = "voxtell_v1.0" 
    os.makedirs(download_dir, exist_ok=True)

    # 3. Obtención de Pesos
    print(f"Validando/Descargando pesos de {MODEL_NAME} desde Hugging Face...")
    model_path = snapshot_download(
        repo_id="mrokuss/VoxTell", 
        allow_patterns=[f"{MODEL_NAME}/*", "*.json"], 
        local_dir=download_dir
    )
    voxtell_weights_dir = os.path.join(download_dir, MODEL_NAME)

    # 4. Inicializar Predictor 
    # (El text encoder Qwen3-Embedding-4B se descargará/cargará automáticamente aquí)
    print("Inicializando VoxTellPredictor...")
    predictor = VoxTellPredictor(
        model_dir=voxtell_weights_dir,
        device=device,
    )

    # 5. Lectura y Reorientación de Datos (Crítico para VoxTell)
    # Construcción dinámica de la ruta usando la variable de entorno
    image_path = os.path.join(data_prep_dir, "train_6_a_2_ct.nii.gz") 
    print(f"Cargando y reorientando volumen a RAS: {image_path}")
    reader = NibabelIOWithReorient()
    img, img_properties = reader.read_images([image_path])

    # 6. Definir los Prompts (Lista F)
    text_prompts = ["liver", "right kidney", "left kidney", "spleen", "pancreas"]
    print(f"Prompts a evaluar: {text_prompts}")

    # 7. Inferencia
    print("Ejecutando inferencia Zero-Shot...")
    # Salida: (num_prompts, x, y, z) -> Equivale a (F, H, W, D)
    voxtell_seg = predictor.predict_single_image(img, text_prompts)
    
    print(f"Éxito. Forma del tensor de salida (F, H, W, D): {voxtell_seg.shape}")

    # 8. Guardado del tensor final
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "pred_0001.npy")
    
    np.save(output_path, voxtell_seg.astype(np.uint8))
    print(f"Tensor guardado listo para evaluación en: {output_path}")

if __name__ == "__main__":
    main()