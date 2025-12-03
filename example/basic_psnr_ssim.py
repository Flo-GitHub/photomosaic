import photomosaic as pm
import matplotlib.pyplot as plt
from skimage import data
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
import cv2
import numpy as np

# 1. Cargar imagen original (Esta imagen suele ser uint8, 0-255)
image = data.coffee()

# 2. Generar el mosaico
pool = pm.make_pool("teselas_procesadas_cv/*.jpg")
mos = pm.basic_mosaic(image, pool, (50, 75))

# 3. Redimensionar
if image.shape != mos.shape:
    mos_resized = cv2.resize(mos, (image.shape[1], image.shape[0]))
else:
    mos_resized = mos

# --- CORRECCIÓN CRÍTICA ---
# Verificamos si la imagen está normalizada (0 a 1) o estándar (0 a 255)
print(f"Rango de valores del mosaico antes de procesar: Min={mos_resized.min()}, Max={mos_resized.max()}")

if mos_resized.dtype != 'uint8':
    if mos_resized.max() <= 1.0:
        print("Detectado rango 0-1. Escalando a 0-255...")
        # Multiplicamos por 255 y luego convertimos a entero
        mos_resized = (mos_resized * 255).astype('uint8')
    else:
        print("Detectado rango 0-255 (float). Convirtiendo a entero...")
        # Ya está en 0-255, solo convertimos el tipo de dato
        mos_resized = mos_resized.astype('uint8')
# --------------------------

# 4. Cálculo de Métricas (Ahora ambas imágenes son uint8 y 0-255)
val_psnr = psnr(image, mos_resized, data_range=255)
val_ssim = ssim(image, mos_resized, channel_axis=-1, win_size=3, data_range=255)

print("-" * 30)
print(f"Resultados de Calidad:")
print(f"PSNR: {val_psnr:.4f} dB")
print(f"SSIM: {val_ssim:.4f}")
print("-" * 30)

# 5. Visualización
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

ax1.imshow(image)
ax1.set_title("Original")
ax1.axis('off')

# Ahora sí debería verse correctamente
ax2.imshow(mos_resized) 
ax2.set_title(f"Mosaico Recuperado\nPSNR: {val_psnr:.2f} | SSIM: {val_ssim:.3f}")
ax2.axis('off')

plt.tight_layout()
plt.show()