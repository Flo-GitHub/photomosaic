import os
import fiftyone as fo
import fiftyone.zoo as foz
import shutil

# Configuracións
DEST = "teselas"
LABELS = ["Flower", "Car", "Balloon", "Toy", "Traffic sign","Fireworks", "Flag", "Butterfly","Fruit","Dinosaur"]
LIMIT = 200 # Imágenes por categoría

# Crear carpeta de destino si no existe
os.makedirs(DEST, exist_ok=True)

print(f"Iniciando descarga de imágenes para: {LABELS}")

# Descargar imágenes usando el 'Zoo' de FiftyOne
# Usamos el dataset 'open-images-v7y'
dataset = foz.load_zoo_dataset(
    "open-images-v7",
    split="validation",       
    label_types=["detections"],
    classes=LABELS,
    max_samples=LIMIT * len(LABELS), 
    shuffle=True,
    seed=42
)

# Mover las imágenes descargadas a tu carpeta 'teselas'
# FiftyOne las guarda en una carpeta oculta por defecto
print("Moviendo imágenes a la carpeta de teselas...")

count = 0
for sample in dataset:
    # Ruta original de la imagen descargada
    source_path = sample.filepath
    
    # Crear un nombre nuevo para evitar duplicados
    filename = os.path.basename(source_path)
    dest_path = os.path.join(DEST, filename)
    
    shutil.copy(source_path, dest_path)
    count += 1

print(f"¡Listo! Se han guardado {count} imágenes en la carpeta '{DEST}'.")