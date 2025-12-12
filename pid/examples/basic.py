import photomosaic as pm
import matplotlib.pyplot as plt

# Analyze the collection (the "pool") of images.
from skimage import data
image = data.chelsea() # ahora es una imagen de gato

print(image.shape)

pool = pm.make_pool("pid/pruebas/teselas_menos_variedad/*.jpg")
mos = pm.basic_mosaic(image, pool, (30, 45))

plt.imshow(mos)
plt.show()