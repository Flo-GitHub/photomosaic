
import os
import glob
from functools import partial

import numpy as np
from skimage import img_as_float
from skimage.io import imread, imsave
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

import photomosaic as pm
from pid.metrics.evalu_dreamsim import evaluate_dreamsim
from pid.metrics.evalu_lpips import evaluate_lpips

# ============================
#   CONSTANTES DE CONFIG
# ============================

# Dónde están las imágenes objetivo sobre las que hacer todas las pruebas
CARPETA_IMAGENES_PRUEBA = "pid/pruebas/imagenes_prueba"
PATRON_IMAGENES = "*.jpg"

# Patrón de teselas dentro de cada carpeta_teselas que se pase a la función
PATRON_TESELAS = "*.jpg"

# Resultados base
CARPETA_RESULTADOS_BASE = "pid/pruebas/resultados"
FORMATO_SALIDA = "png"

# Nombre del CSV donde se guardan TODAS las métricas detalladas
# (una fila por combinación prueba + imagen)
NOMBRE_CSV_DETALLE = "metricas_detalle.csv"

# Pool
SAMPLE_SIZE = None  # None = usar todos los píxeles (sin aleatoriedad)

# Rendimiento
USAR_CACHE_REDIMENSIONADOS = True

# ============================
#   FUNCIÓN PRINCIPAL
# ============================

def realizar_prueba(
    nombre_prueba: str,
    carpeta_teselas: str = "pid/pruebas/teselas_mas_variedad",
    grid_dims: tuple[int, int] = (60, 40),
    usar_adaptacion_pool: bool = False,
    usar_espacio_perceptual: bool = True,
    depth: int = 0,
    analyzer_pool = partial(np.mean, axis=0),
    tile_color_aggregator = partial(np.mean, axis=0),
):
    """
    Ejecuta una prueba completa de fotomosaicos.

    - Usa todas las imágenes de CARPETA_IMAGENES_PRUEBA.
    - Usa las teselas de `carpeta_teselas` (con PATRON_TESELAS).
    - Genera mosaicos con los parámetros dados.
    - Guarda mosaicos en:
        resultados/<tipo_experimento>/<nombre_prueba>/
      donde <tipo_experimento> se extrae de nombre_prueba si sigue
      la convención 'prueba_<tipo>_<config>...'.
    - Calcula métricas (PSNR, SSIM, LPIPS, DreamSim).
    - Añade una fila por imagen al CSV:
        resultados/<tipo_experimento>/metricas_detalle.csv
    """

    # 0) Determinar tipo de experimento a partir del nombre
    tipo_experimento = _extraer_tipo_desde_nombre(nombre_prueba)

    if tipo_experimento is not None:
        carpeta_experimento_base = os.path.join(
            CARPETA_RESULTADOS_BASE, tipo_experimento
        )
    else:
        # Si no sigue la convención, usar directamente la carpeta base
        carpeta_experimento_base = CARPETA_RESULTADOS_BASE

    # 1) Carpeta de salida para las imágenes de esta prueba concreta
    carpeta_salida = os.path.join(carpeta_experimento_base, nombre_prueba)
    os.makedirs(carpeta_salida, exist_ok=True)

    # 2) Construir pool de teselas una sola vez
    patron_teselas_completo = os.path.join(carpeta_teselas, PATRON_TESELAS)
    pool = pm.make_pool(
        patron_teselas_completo,
        analyzer=analyzer_pool,
        sample_size=SAMPLE_SIZE,
    )

    # 3) Listar imágenes de prueba
    patron_imagenes_completo = os.path.join(CARPETA_IMAGENES_PRUEBA, PATRON_IMAGENES)
    rutas_imagenes = sorted(glob.glob(patron_imagenes_completo))

    if not rutas_imagenes:
        print(f"[AVISO] No se encontraron imágenes en {patron_imagenes_completo}")
        return

    # 4) Preparar matcher y cache global para esta prueba
    matcher = pm.simple_matcher(pool)
    cache_redimensionados = {} if USAR_CACHE_REDIMENSIONADOS else None

    resultados_metricas = []

    # 5) Bucle principal sobre todas las imágenes de prueba
    for ruta_img in rutas_imagenes:
        nombre_base = os.path.splitext(os.path.basename(ruta_img))[0]
        print(f"[INFO] Procesando imagen: {nombre_base}")

        # (a) Leer y preparar imagen
        image = imread(ruta_img)
        image = img_as_float(image)

        img_trabajo = image
        if usar_espacio_perceptual:
            img_trabajo = pm.perceptual(img_trabajo)

        if usar_adaptacion_pool:
            img_trabajo = pm.adapt_to_pool(img_trabajo, pool)

        # (b) Particionar
        scaled_img = pm.rescale_commensurate(
            img_trabajo, grid_dims=grid_dims, depth=depth
        )
        tiles = pm.partition(scaled_img, grid_dims=grid_dims, depth=depth)

        # (c) Calcular color de cada tesela
        tile_colors = [
            tile_color_aggregator(scaled_img[tile].reshape(-1, 3))
            for tile in tiles
        ]

        # (d) Matching
        matches = [matcher(tc) for tc in tile_colors]

        # (e) Canvas (blanco fijo)
        canvas = np.ones_like(scaled_img)

        # (f) Dibujar mosaico
        draw_kwargs = {}
        if cache_redimensionados is not None:
            draw_kwargs["resized_copy_cache"] = cache_redimensionados

        mos = pm.draw_mosaic(canvas, tiles, matches, **draw_kwargs)

        # (g) Guardar mosaico como imagen (uint8 en [0, 255])
        mos_rgb = mos
        mos_rgb = np.clip(mos_rgb, 0.0, 1.0)
        mos_uint8 = (mos_rgb * 255).round().astype(np.uint8)

        ruta_salida_img = os.path.join(
            carpeta_salida,
            f"{nombre_base}.{FORMATO_SALIDA}",
        )
        imsave(ruta_salida_img, mos_uint8)

        # (h) Calcular métricas por imagen
        fila_metricas = {
            "prueba": nombre_prueba,
            "tipo_experimento": tipo_experimento,
            "imagen": nombre_base,
            "carpeta_teselas": carpeta_teselas,
            "grid_rows": grid_dims[0],
            "grid_cols": grid_dims[1],
            "usar_adaptacion_pool": usar_adaptacion_pool,
            "usar_espacio_perceptual": usar_espacio_perceptual,
            "depth": depth,
        }

        # Referencia con misma resolución (RGB)
        if usar_espacio_perceptual:
            # scaled_img está en espacio perceptual -> convertir a RGB
            ref_rgb = pm.rgb(scaled_img)
        else:
            # scaled_img ya está en RGB -> usar directamente
            ref_rgb = scaled_img

        ref_rgb = np.clip(ref_rgb, 0.0, 1.0).astype(np.float32)
        mos_rgb_f = np.clip(mos_rgb, 0.0, 1.0).astype(np.float32)

        #CALCULAR PSNR:
        psnr_val = peak_signal_noise_ratio(ref_rgb, mos_rgb_f, data_range=1.0)
        fila_metricas["psnr"] = float(psnr_val)

        #CALCULAR SSIM:
        ssim_val = structural_similarity(
            ref_rgb,
            mos_rgb_f,
            channel_axis=-1,
            data_range=1.0,
        )
        fila_metricas["ssim"] = float(ssim_val)

        #CALCULAR LPIPS:
        lpips_val = evaluate_lpips(ref_rgb, mos_rgb_f)
        fila_metricas["lpips"] = float(lpips_val)

        #CALCULAR DREAMSIM:
        dreamsim_val = evaluate_dreamsim(ref_rgb, mos_rgb_f)
        fila_metricas["dreamsim"] = float(dreamsim_val)

        resultados_metricas.append(fila_metricas)

    # 6) Guardar todas las filas de métricas en el CSV de detalle
    _guardar_metricas_detalle(
        carpeta_experimento_base=carpeta_experimento_base,
        resultados_metricas=resultados_metricas,
    )


# ============================
#   HELPERS
# ============================

def _extraer_tipo_desde_nombre(nombre_prueba: str):
    """
    Convención: nombre_prueba = 'prueba_<tipo>_<config>...'

    Ejemplos:
      'prueba_grid_dims_45x30'              -> tipo = 'grid_dims'
      'prueba_usar_adaptacion_pool_True'    -> tipo = 'usar_adaptacion_pool'
      'prueba_usar_espacio_perceptual_False'-> tipo = 'usar_espacio_perceptual'

    Implementación:
      - Spliteamos por '_'
      - Verificamos que empiece por 'prueba' y tenga al menos 3 partes.
      - Consideramos que la ÚLTIMA parte es la configuración (p.ej. 'True', '45x30')
      - El tipo es todo lo que hay entre medias unido con '_'.
    """
    partes = nombre_prueba.split("_")
    if len(partes) >= 3 and partes[0] == "prueba":
        tipo = "_".join(partes[1:-1])
        return tipo
    return None


def _guardar_metricas_detalle(
    carpeta_experimento_base: str,
    resultados_metricas: list[dict],
):
    """
    Escribe/actualiza el CSV de detalle:
    - una fila por imagen y prueba
    - todas las pruebas del mismo tipo de experimento comparten el mismo CSV
    """
    if not resultados_metricas:
        return

    os.makedirs(carpeta_experimento_base, exist_ok=True)

    import csv

    ruta_csv_detalle = os.path.join(carpeta_experimento_base, NOMBRE_CSV_DETALLE)

    # Determinar cabeceras a partir de las claves presentes
    claves = sorted({k for fila in resultados_metricas for k in fila.keys()})
    escribir_cabecera = not os.path.exists(ruta_csv_detalle)

    with open(ruta_csv_detalle, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=claves)
        if escribir_cabecera:
            writer.writeheader()
        for fila in resultados_metricas:
            writer.writerow(fila)

    print(f"[INFO] Métricas detalladas añadidas a {ruta_csv_detalle}")