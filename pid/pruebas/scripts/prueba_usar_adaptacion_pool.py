from pid.pruebas.scripts.prueba_base import realizar_prueba

def prueba_usar_adaptacion_pool():
    usar_adaptacion_pool = [
        True, False
    ]

    for usar_adaptacion in usar_adaptacion_pool:
        realizar_prueba(
            nombre_prueba=f"prueba_usar_adaptacion_pool_{usar_adaptacion}",
            usar_adaptacion_pool=usar_adaptacion,
        )

prueba_usar_adaptacion_pool()