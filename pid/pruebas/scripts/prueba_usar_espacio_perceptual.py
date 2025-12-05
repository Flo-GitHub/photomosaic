from pid.pruebas.scripts.prueba_base import realizar_prueba

def prueba_usar_espacio_perceptual():
    usar_espacio_perceptual_pool = [
        True, False
    ]

    for usar_espacio_perceptual in usar_espacio_perceptual_pool:
        realizar_prueba(
            nombre_prueba=f"prueba_usar_espacio_perceptual_{usar_espacio_perceptual}",
            carpeta_teselas="pid/pruebas/teselas_menos_variedad",
            usar_espacio_perceptual=usar_espacio_perceptual,
        )

prueba_usar_espacio_perceptual()