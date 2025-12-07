#install.packages("tidyverse")
library(tidyverse)

# Set working directory to the script's folder
setwd(dirname(rstudioapi::getActiveDocumentContext()$path))

# Load CSV relative to this folder
ruta_csv <- "../resultados/grid_dims/metricas_detalle.csv"
datos <- read.csv(ruta_csv, stringsAsFactors = FALSE)

# Filtrar solo pruebas de tipo grid_dims
datos_grid <- datos %>%
  filter(tipo_experimento == "grid_dims") %>%
  mutate(
    grid_label   = paste0(grid_rows, "x", grid_cols),
    grid_n_tiles = grid_rows * grid_cols
  ) %>%
  arrange(grid_n_tiles) %>%
  mutate(
    grid_label = factor(grid_label, levels = unique(grid_label))
  )

# Agregar por grid_dims (mediana sobre las 10 imágenes)

resumen_grid <- datos_grid %>%
  group_by(grid_rows, grid_cols, grid_label) %>%
  summarise(
    n_imagenes    = n(),
    psnr_med      = median(psnr,     na.rm = TRUE),
    ssim_med      = median(ssim,     na.rm = TRUE),
    lpips_med     = median(lpips,    na.rm = TRUE),
    dreamsim_med  = median(dreamsim, na.rm = TRUE),
    .groups = "drop"
  )

# Normalización teórica 0–1 (1 = imágenes idénticas)
# SSIM: ya está en [0,1], mayor = mejor
# DreamSim: 0 = idéntico, mayor = peor -> invertimos
# LPIPS: 0 = idéntico, mayor = peor -> invertimos
# PSNR: [0,∞); tomamos 60 dB ≈ "idéntico" y reescalamos

PSNR_MAX_THEO <- 60  # puedes ajustar si quieres otro techo

resumen_norm <- resumen_grid %>%
  mutate(
    ssim_norm     = ssim_med,
    dreamsim_norm = 1 - dreamsim_med,
    lpips_norm    = 1 - lpips_med,
    psnr_norm     = pmin(psnr_med / PSNR_MAX_THEO, 1)
  )

# Pasar a formato largo para plot conjunto
plot_df <- resumen_norm %>%
  select(grid_label,
         psnr_norm, ssim_norm, lpips_norm, dreamsim_norm) %>%
  pivot_longer(
    cols      = c(psnr_norm, ssim_norm, lpips_norm, dreamsim_norm),
    names_to  = "metric",
    values_to = "score"
  ) %>%
  mutate(
    metric = factor(
      metric,
      levels = c("psnr_norm", "ssim_norm", "lpips_norm", "dreamsim_norm"),
      labels = c("PSNR", "SSIM", "LPIPS (inv.)", "DreamSim (inv.)")
    )
  )

# Plot: las 4 métricas normalizadas en un solo gráfico

ggplot(plot_df, aes(x = grid_label, y = score, color = metric, group = metric)) +
  geom_line() +
  geom_point() +
  scale_y_continuous(limits = c(0, 1)) +
  labs(
    title = "Métricas normalizadas (0 = no similitud, 1 = imagen idéntica)",
    x = "grid_rows x grid_cols",
    y = "Score normalizado",
    color = "Métrica"
  ) +
  theme_minimal() +
  theme(
    axis.text.x = element_text(angle = 45, hjust = 1)
  )