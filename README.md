# 🖼️ Generación y Evaluación de Fotomosaicos Digitales (PID)

Este repositorio contiene el código fuente, la documentación y los resultados experimentales del proyecto enfocado en la **generación automatizada de fotomosaicos** y la **validación objetiva de su calidad visual**.

El proyecto parte del *fork* del repositorio `photomosaic` de Daniel Ballan y lo expande para integrar un completo *pipeline* de evaluación de la calidad de imagen.

## 🚀 Repositorio

Este trabajo se construyó sobre una adaptación del siguiente proyecto:

* **[Repositorio Original (Daniel Ballan)](https://github.com/danielballan/photomosaic)**

* **[Nuestro fork](https://github.com/Flo-GitHub/photomosaic)**

## 🔍 Características Principales

El objetivo central del proyecto fue ir más allá de la mera composición, enfocándose en la **evaluación rigurosa** de los resultados.

### Evaluación de Calidad
Se implementó un conjunto de métricas de similitud para contrastar enfoques tradicionales con métodos avanzados:

| Categoría | Métrica | Enfoque |
| :--- | :--- | :--- |
| **Clásicas** | PSNR, SSIM | Precisión Matemática, Fidelidad Estructural |
| **Perceptuales** | LPIPS, DreamSim | Juicio Humano, Coherencia Semántica (Deep Learning) |

## 💡 Conclusiones Clave

* **Coherencia de Dataset:** Se demostró experimentalmente que la **calidad** y la **coherencia cromática natural** del banco de teselas son más críticas que su tamaño, debido a la sensibilidad del algoritmo de emparejamiento.
* **Insuficiencia de Métricas Clásicas:** PSNR y SSIM resultaron ser insuficientes para evaluar la calidad visual de los fotomosaicos, ya que penalizan la sustitución de textura inherente a la técnica.
* **Superioridad Perceptual:** Las métricas **LPIPS** y **DreamSim** mostraron una mejor correlación con la percepción humana, validando su utilidad para la evaluación objetiva de composiciones artísticas basadas en Deep Learning. 

## 🛠️ Manual de Usuario

Una vez clonado el repositorio se deberá ejecutar en la carpeta del proyecto el comando "pip install -e ."
Este instalará todos los paquetes necesarios y las dependencias. Una vez instalados, se puede proceder a ejecutar cualquiera de los scripts en la carpeta 
pid/pruebas/scripts. La primera vez se descargará un modelo preentrenado DreamSim que ocupa alrededor de 1 Gb, por lo que tardará un poco más. Los resultados de estos scripts de pruebas se generarán en pid/pruebas/resultados, sobreescribiendo los anteriores si los hubiera. Para el análisis de los resultados se pueden usar los scripts de la carpeta pid/pruebas/analisis en R Studio.

Todas las carpetas mencionadas anteriormente ya contienen los resultados de nuestra experimentación, por lo que no es necesario ejecutar ningún script si el objetivo es analizar los resultados que se exponen en la documentación del proyecto.

## 📚 Documentación

La memoria completa del proyecto, incluyendo metodología, resultados detallados y discusión de las métricas, se encuentra disponible en la entrega del proyecto.
