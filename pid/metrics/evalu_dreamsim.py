import sys
import os
import numpy as np
import torch
from dreamsim import dreamsim
from PIL import Image

ruta_del_archivo = os.path.abspath(__file__)
carpeta_donde_estoy = os.path.dirname(ruta_del_archivo)
carpeta_del_proyecto = os.path.dirname(carpeta_donde_estoy)
sys.path.append(carpeta_del_proyecto)

print(f"Buscando módulos en: {carpeta_del_proyecto}")
try:
    from example.basic import image, mos
    print("¡Importación correcta!")
except ImportError as e:
    print(f"Error: Python sigue sin verlo. {e}")

    print("Carpetas donde Python está buscando:", sys.path)

def evaluar_arrays(array_original, array_mosaico):
    """
    Recibe dos imágenes como matrices Numpy (H, W, C) y calcula su similitud.
    """
    # 1. Configuración de Hardware
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # 2. Cargar modelo (Idealmente haz esto FUERA del bucle si comparas muchas)
    model, preprocess = dreamsim(pretrained=True, device=device)

    # 3. Función auxiliar interna para convertir Numpy -> PIL
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

    # 4. Procesar las imágenes
    try:
        pil_orig = preparar_imagen(array_original)
        pil_mos = preparar_imagen(array_mosaico)
        
        img_ref = preprocess(pil_orig).to(device)
        img_mos = preprocess(pil_mos).to(device)
        
        # 5. Calcular distancia
        with torch.no_grad():
            distancia = model(img_ref, img_mos)
            
        return distancia.item()

    except Exception as e:
        print(f"Error procesando los arrays: {e}")
        return None

# --- EJEMPLO DE USO (Simulando tu programa) ---
if __name__ == "__main__":
    # Imaginemos que 'photomosaic' te ha dado estos dos arrays:
    # (Creamos ruido aleatorio solo para probar que el código no falla)
    print("Generando arrays de prueba...")
    
    # Imagen Original
    imagen_numpy_1 = image
    
    # Fotomosaico
    imagen_numpy_2 = mos

    # LLAMADA A LA FUNCIÓN
    score = evaluar_arrays(imagen_numpy_1, imagen_numpy_2)
    
    print(f"Distancia calculada directamente desde Numpy: {score:.4f}")