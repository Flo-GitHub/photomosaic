import cv2
import os
import numpy as np


# Carpeta donde están las imágenes descargadas
INPUT_DIR = "teselas"

# Carpeta donde se guardarán las imágenes procesadas y cuadradas
OUTPUT_DIR = "teselas_procesadas_cv"

# Tamaño final de la tesela 
TILE_SIZE = 30 

# Crear la carpeta de ssalida
os.makedirs(OUTPUT_DIR, exist_ok=True)


def process_images_opencv(input_dir, output_dir, size):
    """
    Procesa las imágenes en input_dir.
    """
    processed_count = 0
    
    # Extensiones de archivo que Open CV puede leer
    valid_extensions = ('.jpg','.jpeg')
    
    # Iterar sobre todos los archivos
    for filename in os.listdir(input_dir):
        if filename.lower().endswith(valid_extensions):
            input_path = os.path.join(input_dir, filename)
            output_path = os.path.join(output_dir, filename)

            try:
                # 1. Leer la imagen (OpenCV lee en formato BGR)
                img = cv2.imread(input_path)
                
                # Comprobar si la imagen se cargó correctamente
                if img is None:
                    print(f"Error al cargar la imagen: {filename}")
                    continue

                height, width, _ = img.shape
                
                # 2. Determinar la dimensión más pequeña para el recorte cuadrado
                min_dim = min(width, height)
                
                # 3. Calcular las coordenadas para el recorte central
                start_x = (width - min_dim) // 2
                start_y = (height - min_dim) // 2
                end_x = start_x + min_dim
                end_y = start_y + min_dim
                
                # 4. Recortar la imagen al área cuadrada central
                img_cropped = img[start_y:end_y, start_x:end_x]
                
                # 5. Redimensionar al tamaño final de la tesela
                img_resized = cv2.resize(img_cropped, (size, size), interpolation=cv2.INTER_AREA)
                
                # 6. Guardar la imagen procesada
                cv2.imwrite(output_path, img_resized)
                processed_count += 1
                
            except Exception as e:
                print(f"Error procesando {filename}: {e}")

    print(f"\nProcesamiento completado con OpenCV. {processed_count} imágenes guardadas en '{output_dir}'.")
    print(f"El tamaño de las teselas es: {size}x{size} píxeles.")


if __name__ == "__main__":
    process_images_opencv(INPUT_DIR, OUTPUT_DIR, TILE_SIZE)
    