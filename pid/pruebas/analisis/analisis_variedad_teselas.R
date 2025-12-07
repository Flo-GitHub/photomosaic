# ============================================================
# ANÁLISIS DE VARIEDAD DE TESELAS (normalización teórica 0–1)
# ============================================================

library(tidyverse)

# 1) Leer CSV ---------------------------------------------------------------

ruta_csv <- "../resultados/variedad/metricas_detalle.csv"
datos <- read.csv(ruta_csv, stringsAsFactors = FALSE)

# Filtrar solo las pruebas de variedad
datos_var <- datos %>%
  filter(tipo_experimento == "variedad") %>%
  mutate(
    variedad = ifelse(grepl("mas", carpeta_teselas), "Más variedad", "Menos variedad")
  )

# 2) Normalización teórica --------------------------------------------------
# SSIM:      [0,1]     → ya normalizado
# DreamSim:  [0,1]     → ya normalizado
# LPIPS:     0=idéntico → invertimos
# PSNR:      [0,∞)     → escalamos con 60 dB ≈ idéntico (paper normalization logic)

PSNR_MAX_THEO <- 60  # techo razonable para "idéntico"

datos_norm <- datos_var %>%
  mutate(
    ssim_norm     = ssim,
    dreamsim_norm = dreamsim,
    lpips_norm    = 1 - lpips,
    psnr_norm     = pmin(psnr / PSNR_MAX_THEO, 1)
  )

# 3) Resumen por variedad (medianas por condición) --------------------------

resumen_var <- datos_norm %>%
  group_by(variedad) %>%
  summarise(
    psnr_med_norm     = median(psnr_norm),
    ssim_med_norm     = median(ssim_norm),
    lpips_med_norm    = median(lpips_norm),
    dreamsim_med_norm = median(dreamsim_norm),
    .groups = "drop"
  )

print(resumen_var)

# 4) Pasar a formato largo para graficar -----------------------------------

plot_df <- datos_norm %>%
  select(variedad, imagen,
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

# 5) Gráfico: Comparación de métricas por variedad --------------------------

ggplot(plot_df, aes(x = variedad, y = score, color = metric)) +
  geom_jitter(width = 0.1, alpha = 0.4) +  # puntos por imagen
  stat_summary(fun = median, geom = "point", size = 5, shape = 18) +
  stat_summary(fun = median, geom = "line", aes(group = metric)) +
  scale_y_continuous(limits = c(0, 1)) +
  labs(
    title = "Comparación de Métricas Normalizadas por Variedad de Teselas",
    x = "Condición (Variedad de teselas)",
    y = "Score normalizado (0–1, 1 = idéntico)",
    color = "Métrica"
  ) +
  theme_minimal()
