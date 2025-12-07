# ============================================================
# ANÁLISIS DE DEPTH (normalización teórica 0–1)
# ============================================================

library(tidyverse)

# 1) Leer CSV ---------------------------------------------------------------

ruta_csv <- "../resultados/depth/metricas_detalle.csv"
datos <- read.csv(ruta_csv, stringsAsFactors = FALSE)

# Filtrar solo pruebas de tipo depth
datos_depth <- datos %>%
  filter(tipo_experimento == "depth") %>%
  mutate(
    depth = as.factor(depth)   # convertir a factor para graficar
  )

# 2) Normalización teórica --------------------------------------------------
# SSIM:      [0,1] → ya está
# DreamSim:  0=idéntico → invertimos
# LPIPS:     0=idéntico → invertimos
# PSNR:      [0,∞)  → escalamos PSNR/60 (capped)

PSNR_MAX_THEO <- 60  # techo razonable para similaridad ~ idéntica

datos_norm <- datos_depth %>%
  mutate(
    ssim_norm     = ssim,
    dreamsim_norm = 1 - dreamsim,
    lpips_norm    = 1 - lpips,
    psnr_norm     = pmin(psnr / PSNR_MAX_THEO, 1)
  )

# 3) Resumen estadístico por depth -----------------------------------------

resumen_depth <- datos_norm %>%
  group_by(depth) %>%
  summarise(
    psnr_med_norm     = median(psnr_norm),
    ssim_med_norm     = median(ssim_norm),
    lpips_med_norm    = median(lpips_norm),
    dreamsim_med_norm = median(dreamsim_norm),
    n_imagenes        = n(),
    .groups = "drop"
  )

print(resumen_depth)

# 4) Pasar a formato largo --------------------------------------------------

plot_df <- datos_norm %>%
  select(depth, imagen,
         psnr_norm, ssim_norm, lpips_norm, dreamsim_norm) %>%
  pivot_longer(
    cols = c(psnr_norm, ssim_norm, lpips_norm, dreamsim_norm),
    names_to = "metric",
    values_to = "score"
  ) %>%
  mutate(
    metric = factor(
      metric,
      levels = c("psnr_norm", "ssim_norm", "lpips_norm", "dreamsim_norm"),
      labels = c("PSNR", "SSIM", "LPIPS (inv.)", "DreamSim (inv.)")
    )
  )

# 5) Gráfico: Todas las métricas normalizadas por depth ---------------------

ggplot(plot_df, aes(x = depth, y = score, color = metric)) +
  geom_jitter(width = 0.1, alpha = 0.35) +             # valores individuales
  stat_summary(fun = median, geom = "point", size = 5, shape = 18) +
  stat_summary(fun = median, geom = "line", aes(group = metric)) +
  scale_y_continuous(limits = c(0, 1)) +
  labs(
    title = "Métricas normalizadas por depth\n(0–1, donde 1 = imagen idéntica)",
    x = "Depth",
    y = "Score normalizado",
    color = "Métrica"
  ) +
  theme_minimal()
