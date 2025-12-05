from pid.pruebas.scripts.prueba_base import realizar_prueba

def prueba_grid_dims():
    dims = [
        (15, 10),
        (30, 20),
        (45, 30),
        (60, 40),
        (80, 50),
        (100, 70),
        (120, 80),
        (150, 100)
    ]

    for dim in dims:
        realizar_prueba(
            nombre_prueba=f"prueba_grid_dims_{dim[0]}x{dim[1]}",
            carpeta_teselas="pid/pruebas/teselas_menos_variedad",
            grid_dims=dim,
        )

prueba_grid_dims()