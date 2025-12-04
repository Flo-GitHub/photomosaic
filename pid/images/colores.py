import os
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from skimage.io import imread
import sys


CARPETA_TESELAS = 'teselas_procesadas_cv' 

def analizar_distribucion_color(carpeta):
    colores = []
    
    print(f"Analizando imágenes en: {carpeta} ...")
    archivos = [f for f in os.listdir(carpeta) if f.lower().endswith(('.jpg', '.jpeg'))]
    
    if not archivos:
        print(" No encontré imágenes en esa carpeta.")
        return

    # 1. Calcular el color promedio de cada imagen
    for archivo in archivos:
        try:
            ruta = os.path.join(carpeta, archivo)
            img = imread(ruta)
            
            # Si la imagen tiene 4 canales (RGBA), quitamos el Alpha
            if img.shape[-1] == 4:
                img = img[:, :, :3]
            
            # Si es blanco y negro, lo saltamos o convertimos
            if len(img.shape) == 2:
                continue

            # Calculamos la media de R, G y B
            avg_color = img.mean(axis=(0, 1)) # Promedio de todos los píxeles
            colores.append(avg_color)
            
        except Exception as e:
            pass # Si una imagen falla, la saltamos

    colores = np.array(colores)
    print(f" Analizadas {len(colores)} imágenes.")

    # 2. Generar el Gráfico 3D
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Normalizamos los colores de 0-255 a 0-1 para que matplotlib los entienda
    colores_norm = colores / 255.0

    # Pintamos cada punto:
    # X = Rojo, Y = Verde, Z = Azul
    # c = El color real del punto
    ax.scatter(colores[:, 0], colores[:, 1], colores[:, 2], c=colores_norm, s=20)

    ax.set_xlabel('Rojo (Red)')
    ax.set_ylabel('Verde (Green)')
    ax.set_zlabel('Azul (Blue)')
    ax.set_title('Distribución de Colores de tu Pool')
    
    # Límites del cubo RGB (0 a 255)
    ax.set_xlim(0, 255)
    ax.set_ylim(0, 255)
    ax.set_zlim(0, 255)

    plt.show()

if __name__ == "__main__":
    # NOTA: Si estás usando el pool de ejemplo 'example.basic',
    # las imágenes a veces se descargan en una carpeta temporal o en local.
    # Intenta buscar una carpeta llamada 'pool' en tu proyecto.
    
    # Si no tienes carpeta y solo quieres probar, cambia esto por tu ruta real:
    ruta_real = 'teselas_procesadas_cv' 
    
    if os.path.exists(ruta_real):
        analizar_distribucion_color(ruta_real)
    else:
        print("Por favor, edita la variable CARPETA_TESELAS con la ruta correcta.")