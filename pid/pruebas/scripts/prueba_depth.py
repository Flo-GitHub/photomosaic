from pid.pruebas.scripts.prueba_base import realizar_prueba

def prueba_depth():
    depths = [
        0, 1, 2, 3
    ]

    for depth in depths:
        realizar_prueba(
            nombre_prueba=f"prueba_depth_{depth}",
            depth=depth,
        )

prueba_depth()