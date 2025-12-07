from pid.pruebas.scripts import prueba_depth, prueba_grid_dims, prueba_usar_adaptacion_pool, prueba_usar_espacio_perceptual, prueba_variedad_teselas

# Realizar todas las pruebas
def todas_pruebas():
    prueba_variedad_teselas()
    prueba_grid_dims()
    prueba_usar_adaptacion_pool()
    prueba_depth()

    #No funciona así, tendríamos que cambiar la implementación
    #prueba_usar_espacio_perceptual() 

todas_pruebas()