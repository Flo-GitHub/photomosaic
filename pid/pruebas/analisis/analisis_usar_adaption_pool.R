# ============================================================
# ANÁLISIS DE usar_adaptacion_pool (0–1, 1 = id. teórica)
# ============================================================

library(tidyverse)

# 1) Leer CSV ---------------------------------------------------------------

ruta_csv <- "../resultados/usar_adaptacion_pool/metricas_detalle.csv"
datos <- read.csv(ruta_csv, stringsAsFactors = FALSE)

# Filtrar solo las pruebas correctas
datos_adapt <- datos %>%
  filter(tipo_experimento == "usar_adaptacion_pool") %>%
  mutate(
    adapt = ifelse(usar_adaptacion_pool, "Adaptación activada", "Adaptación desactivada")
  )

# 2) Normalización teórica --------------------------------------------------
# SSIM:      ya en [0,1]
# DreamSim:  ya en [0,1]
# LPIPS:     0=idéntico → invertimos
# PSNR:      escala 0–1 usando 60 dB ≈ idéntico

PSNR_MAX_THEO <- 60

datos_norm <- datos_adapt %>%
  mutate(
    ssim_norm     = ssim,
    dreamsim_norm = dreamsim,
    lpips_norm    = 1 - lpips,
    psnr_norm     = pmin(psnr / PSNR_MAX_THEO, 1)
  )

# 3) Resumen por condición (medianas) ---------------------------------------

resumen_adapt <- datos_norm %>%
  group_by(adapt) %>%
  summarise(
    psnr_med_norm     = median(psnr_norm),
    ssim_med_norm     = median(ssim_norm),
    lpips_med_norm    = median(lpips_norm),
    dreamsim_med_norm = median(dreamsim_norm),
    n_imagenes        = n(),
    .groups = "drop"
  )

print(resumen_adapt)

# 4) Formato largo para graficar -------------------------------------------

plot_df <- datos_norm %>%
  select(adapt, imagen,
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
      labels = c("PSNR", "SSIM", "LPIPS (inv.)", "DreamSim")
    )
  )

# 5) Plot: comparación completa --------------------------------------------

ggplot(plot_df, aes(x = adapt, y = score, color = metric)) +
  geom_jitter(width = 0.15, alpha = 0.35) +     # puntos por imagen (ruido leve)
  stat_summary(fun = median, geom = "point", size = 5, shape = 18) +
  stat_summary(fun = median, geom = "line", aes(group = metric)) +
  scale_y_continuous(limits = c(0, 1)) +
  labs(
    title = "Efecto de usar_adaptacion_pool en métricas normalizadas\n(0–1, 1 = imagen idéntica)",
    x = "Condición",
    y = "Score normalizado",
    color = "Métrica"
  ) +
  theme_minimal() +
  theme(axis.text.x = element_text(angle = 25, hjust = 1))
