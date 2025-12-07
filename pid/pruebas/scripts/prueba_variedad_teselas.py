from pid.pruebas.scripts.prueba_base import realizar_prueba

def prueba_variedad_teselas():
    carpetas = [
        "pid/pruebas/teselas_menos_variedad",
        "pid/pruebas/teselas_mas_variedad"
    ]

    for carpeta_teselas in carpetas:
        realizar_prueba(
            nombre_prueba=f"prueba_variedad_{carpeta_teselas.split('/')[-1].split('_')[-2]}",
            carpeta_teselas=carpeta_teselas,
        )

prueba_variedad_teselas()