
import os
import glob
from functools import partial

import numpy as np
from skimage import img_as_float
from skimage.io import imread, imsave
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

import photomosaic as pm
from pid.metrics.evalu_lpips import evaluate_lpips
# from pid_metrics.dreamsim import evaluate_dreamsim  # TODO: ajustar import si lo tenéis



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

# Nombres de ficheros CSV (se crean dentro de la carpeta del tipo de experimento)
NOMBRE_CSV_DETALLE = "metricas_detalle.csv"          # una fila por (prueba, imagen)
NOMBRE_CSV_RESUMEN = "metricas_resumen_prueba.csv"  # una fila por prueba/config
NOMBRE_CSV_RESUMEN_IMAGEN = "metricas_resumen_imagen.csv"  # una fila por imagen

# Pool
SAMPLE_SIZE = None  # None = usar todos los píxeles (sin aleatoriedad)

# Rendimiento
USAR_CACHE_REDIMENSIONADOS = True

# Métricas a calcular
CALCULAR_PSNR = True
CALCULAR_SSIM = True
CALCULAR_LPIPS = True
CALCULAR_DREAMSIM = False  # poned True cuando tengáis la función lista


# ============================
#   FUNCIÓN PRINCIPAL
# ============================

def realizar_prueba(
    nombre_prueba: str,
    carpeta_teselas: str = "pid/pruebas/teselas_menos_variedad",
    grid_dims: tuple[int, int] = (45, 30),
    usar_adaptacion_pool: bool = True,
    usar_espacio_perceptual: bool = True,
    depth: int = 1,
    analyzer_pool = partial(np.mean, axis=0),
    tile_color_aggregator = partial(np.mean, axis=0),
):
    """
    Ejecuta una prueba completa de fotomosaicos:

    - Usa todas las imágenes de CARPETA_IMAGENES_PRUEBA.
    - Usa las teselas de `carpeta_teselas` (con PATRON_TESELAS).
    - Genera mosaicos con los parámetros dados.
    - Guarda mosaicos en resultados/<tipo>/<nombre_prueba>/.
    - Actualiza:
        * metricas_detalle.csv (fila por imagen)
        * metricas_resumen_prueba.csv (fila por configuración)
        * metricas_resumen_imagen.csv (fila por imagen agregando sobre pruebas)
    """

    # 0) Determinar tipo de experimento (si sigue la convención de nombre)
    tipo_experimento = _extraer_tipo_desde_nombre(nombre_prueba)

    if tipo_experimento is not None:
        carpeta_experimento_base = os.path.join(
            CARPETA_RESULTADOS_BASE, tipo_experimento
        )
    else:
        # fallback: comportamiento antiguo
        carpeta_experimento_base = CARPETA_RESULTADOS_BASE

    # 1) Carpeta de salida para esta prueba
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
        scaled_img = pm.rescale_commensurate(img_trabajo, grid_dims=grid_dims, depth=depth)
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

        # (g) Guardar mosaico
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

        # referencia con misma resolución (RGB)
        ref_rgb = pm.rgb(scaled_img)
        ref_rgb = np.clip(ref_rgb, 0.0, 1.0).astype(np.float32)
        mos_rgb_f = np.clip(mos_rgb, 0.0, 1.0).astype(np.float32)

        if CALCULAR_PSNR:
            psnr_val = peak_signal_noise_ratio(ref_rgb, mos_rgb_f, data_range=1.0)
            fila_metricas["psnr"] = float(psnr_val)

        if CALCULAR_SSIM:
            ssim_val = structural_similarity(
                ref_rgb, mos_rgb_f,
                channel_axis=-1,
                data_range=1.0,
            )
            fila_metricas["ssim"] = float(ssim_val)

        if CALCULAR_LPIPS:
            lpips_val = evaluate_lpips(ref_rgb, mos_rgb_f)
            fila_metricas["lpips"] = float(lpips_val)

        if CALCULAR_DREAMSIM:
            # dreamsim_val = evaluate_dreamsim(ref_rgb, mos_rgb_f)
            # fila_metricas["dreamsim"] = float(dreamsim_val)
            pass

        resultados_metricas.append(fila_metricas)

    # 6) Guardar métricas por prueba (detalle + resumen por prueba)
    _guardar_metricas_detalle_y_resumen_prueba(
        nombre_prueba=nombre_prueba,
        tipo_experimento=tipo_experimento,
        carpeta_teselas=carpeta_teselas,
        grid_dims=grid_dims,
        usar_adaptacion_pool=usar_adaptacion_pool,
        usar_espacio_perceptual=usar_espacio_perceptual,
        depth=depth,
        resultados_metricas=resultados_metricas,
        carpeta_experimento_base=carpeta_experimento_base,
    )

    # 7) Recalcular resumen por imagen a partir del CSV de detalle
    _recalcular_resumen_por_imagen(carpeta_experimento_base)


# ============================
#   HELPERS
# ============================

def _extraer_tipo_desde_nombre(nombre_prueba: str):
    """
    Convención: nombre_prueba = 'prueba_<tipo>_<config>...'
    Ejemplo:    'prueba_grid_dims_45x30' -> tipo = 'grid_dims'
    """
    partes = nombre_prueba.split("_")
    if len(partes) >= 3 and partes[0] == "prueba":
        return partes[1]
    return None


def _guardar_metricas_detalle_y_resumen_prueba(
    nombre_prueba: str,
    tipo_experimento: str | None,
    carpeta_teselas: str,
    grid_dims: tuple[int, int],
    usar_adaptacion_pool: bool,
    usar_espacio_perceptual: bool,
    depth: int,
    resultados_metricas: list[dict],
    carpeta_experimento_base: str,
):
    """
    Escribe/actualiza:
      - CSV de detalle (una fila por imagen)
      - CSV de resumen por prueba (una fila por configuración)
    """
    if not resultados_metricas:
        return

    os.makedirs(carpeta_experimento_base, exist_ok=True)

    import csv

    # --- CSV DETALLE ---
    ruta_csv_detalle = os.path.join(carpeta_experimento_base, NOMBRE_CSV_DETALLE)

    claves_detalle = sorted({k for fila in resultados_metricas for k in fila.keys()})
    escribir_cabecera_detalle = not os.path.exists(ruta_csv_detalle)

    with open(ruta_csv_detalle, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=claves_detalle)
        if escribir_cabecera_detalle:
            writer.writeheader()
        for fila in resultados_metricas:
            writer.writerow(fila)

    # --- CSV RESUMEN POR PRUEBA ---
    posibles_metricas = ["psnr", "ssim", "lpips", "dreamsim"]
    metricas_presentes = [m for m in posibles_metricas if m in resultados_metricas[0]]

    resumen = {
        "prueba": nombre_prueba,
        "tipo_experimento": tipo_experimento,
        "carpeta_teselas": carpeta_teselas,
        "grid_rows": grid_dims[0],
        "grid_cols": grid_dims[1],
        "usar_adaptacion_pool": usar_adaptacion_pool,
        "usar_espacio_perceptual": usar_espacio_perceptual,
        "depth": depth,
        "num_imagenes": len(resultados_metricas),
    }

    for m in metricas_presentes:
        valores = np.array([fila[m] for fila in resultados_metricas], dtype=float)
        resumen[f"{m}_mean"] = float(np.mean(valores))
        resumen[f"{m}_std"] = float(np.std(valores))
        resumen[f"{m}_min"] = float(np.min(valores))
        resumen[f"{m}_max"] = float(np.max(valores))

    ruta_csv_resumen = os.path.join(carpeta_experimento_base, NOMBRE_CSV_RESUMEN)
    claves_resumen = sorted(resumen.keys())
    escribir_cabecera_resumen = not os.path.exists(ruta_csv_resumen)

    with open(ruta_csv_resumen, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=claves_resumen)
        if escribir_cabecera_resumen:
            writer.writeheader()
        writer.writerow(resumen)

    print(f"[INFO] Métricas de detalle -> {ruta_csv_detalle}")
    print(f"[INFO] Métricas resumen por prueba -> {ruta_csv_resumen}")


def _recalcular_resumen_por_imagen(carpeta_experimento_base: str):
    """
    Lee metricas_detalle.csv en carpeta_experimento_base y genera
    metricas_resumen_imagen.csv con estadísticas agregadas por imagen
    (a través de todas las configuraciones / pruebas de ese tipo).
    """
    import csv

    ruta_csv_detalle = os.path.join(carpeta_experimento_base, NOMBRE_CSV_DETALLE)
    if not os.path.exists(ruta_csv_detalle):
        # Todavía no hay detalle, nada que resumir
        return

    # Leer todo el detalle
    with open(ruta_csv_detalle, "r", newline="") as f:
        reader = csv.DictReader(f)
        filas = list(reader)

    if not filas:
        return

    # Agrupar por imagen
    por_imagen: dict[str, list[dict]] = {}
    for fila in filas:
        img = fila.get("imagen", "UNKNOWN")
        por_imagen.setdefault(img, []).append(fila)

    posibles_metricas = ["psnr", "ssim", "lpips", "dreamsim"]

    resumen_imagenes = []
    for imagen, lista_filas in por_imagen.items():
        # Tomamos algunos metadatos de la primera fila (tipo_experimento, carpeta_teselas, etc.)
        base = lista_filas[0]
        fila_resumen = {
            "imagen": imagen,
            "tipo_experimento": base.get("tipo_experimento"),
            "carpeta_teselas": base.get("carpeta_teselas"),
            "num_pruebas": len(lista_filas),
        }

        # Extraer métricas presentes
        metricas_presentes = [m for m in posibles_metricas if m in base and base[m] != ""]
        for m in metricas_presentes:
            vals = []
            for f in lista_filas:
                v = f.get(m, "")
                if v != "":
                    try:
                        vals.append(float(v))
                    except ValueError:
                        pass
            if not vals:
                continue
            vals = np.array(vals, dtype=float)
            fila_resumen[f"{m}_mean"] = float(np.mean(vals))
            fila_resumen[f"{m}_std"] = float(np.std(vals))
            fila_resumen[f"{m}_min"] = float(np.min(vals))
            fila_resumen[f"{m}_max"] = float(np.max(vals))

        resumen_imagenes.append(fila_resumen)

    # Escribir/reescribir CSV de resumen por imagen (lo regeneramos entero cada vez)
    ruta_csv_resumen_imagen = os.path.join(carpeta_experimento_base, NOMBRE_CSV_RESUMEN_IMAGEN)
    if not resumen_imagenes:
        return

    # Determinar cabeceras
    claves = sorted({k for fila in resumen_imagenes for k in fila.keys()})

    with open(ruta_csv_resumen_imagen, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=claves)
        writer.writeheader()
        for fila in resumen_imagenes:
            writer.writerow(fila)

    print(f"[INFO] Métricas resumen por imagen -> {ruta_csv_resumen_imagen}")