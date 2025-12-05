import numpy as np
import torch
from dreamsim import dreamsim
from PIL import Image


# Configuración de Hardware
if torch.cuda.is_available():
    device = "cuda"
elif torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"

# Cargar modelo (Idealmente haz esto FUERA del bucle si comparas muchas)
model, preprocess = dreamsim(pretrained=True, device=device)

def evaluate_dreamsim(array_original, array_mosaico):
    """
    Recibe dos imágenes como matrices Numpy (H, W, C) y calcula su similitud.
    """

    # Función auxiliar interna para convertir Numpy -> PIL
    def preparar_imagen(arr):
        # Asegurarnos de que sea una copia para no modificar el original
        arr = np.array(arr)
        
        # CASO IMPORTANTE: Si la imagen viene en formato Float (0.0 a 1.0)
        # Muchas librerías científicas usan floats, pero PIL necesita 0-255
        if arr.dtype != np.uint8:
            # Si los valores son pequeños (tipo 0.5, 0.9), asumimos rango 0-1
            if arr.max() <= 1.5: 
                arr = (arr * 255).astype(np.uint8)
            else:
                arr = arr.astype(np.uint8)
                
        return Image.fromarray(arr).convert('RGB')

    # Procesar las imágenes
    try:
        pil_orig = preparar_imagen(array_original)
        pil_mos = preparar_imagen(array_mosaico)
        
        img_ref = preprocess(pil_orig).to(device)
        img_mos = preprocess(pil_mos).to(device)
        
        # Calcular distancia
        with torch.no_grad():
            distancia = model(img_ref, img_mos)
            
        return distancia.item()

    except Exception as e:
        print(f"Error procesando los arrays: {e}")
        return None