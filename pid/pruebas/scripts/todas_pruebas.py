from pid.pruebas.scripts import prueba_depth, prueba_grid_dims, prueba_usar_adaptacion_pool, prueba_usar_espacio_perceptual

# Realizar todas las pruebas
def todas_pruebas():
    prueba_grid_dims()
    prueba_usar_adaptacion_pool()
    prueba_usar_espacio_perceptual()
    prueba_depth()

todas_pruebas()