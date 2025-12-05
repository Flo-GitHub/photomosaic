from pid.pruebas.scripts.prueba_base import realizar_prueba

def prueba_depth():
    depths = [
        0, 1, 2, 3
    ]

    for depth in depths:
        # también ajustar grid_dims para que no haya demasiadas teselas
        realizar_prueba(
            nombre_prueba=f"prueba_depth_{depth}",
            carpeta_teselas="pid/pruebas/teselas_menos_variedad",
            grid_dims=(45, 30),
            depth=depth,
        )

prueba_depth()