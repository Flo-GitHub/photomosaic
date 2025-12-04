import numpy as np
import torch
import lpips

# Para usar GPU si está disponible (mucho más rápido)
if torch.cuda.is_available():
    device = "cuda"
elif torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"

# Crear modelo LPIPS (AlexNet por defecto)
lpips_model = lpips.LPIPS(net='alex').to(device).eval()

def evaluate_lpips(img_ref_np: np.ndarray,
                     img_mosaic_np: np.ndarray,
                     model=lpips_model,
                     device=device) -> float:
    """
    Calcula LPIPS entre dos imágenes dadas como numpy arrays.

    img_ref_np: np.ndarray, shape [H,W] o [H,W,3]
    img_mosaic_np: np.ndarray, mismo tamaño que img_ref_np
    Devuelve: float (0 = muy similares, más alto = más diferentes)
    """

    def to_tensor(img_np: np.ndarray) -> torch.Tensor:
        # Asegurarnos de que es float32
        t = torch.from_numpy(img_np)

        # Si viene como uint8 0..255, lo pasamos a float 0..1
        if t.dtype == torch.uint8:
            t = t.float() / 255.0
        else:
            t = t.float()

        # Casos de shape:
        # [H,W] (grayscale) -> [1,H,W] -> repetir 3 canales
        if t.ndim == 2:
            t = t.unsqueeze(0)           # [1,H,W]
            t = t.repeat(3, 1, 1)        # [3,H,W]

        # [H,W,C] -> [C,H,W]
        elif t.ndim == 3:
            if t.shape[2] == 3 or t.shape[2] == 1:  # asumimos [H,W,C]
                t = t.permute(2, 0, 1)  # [C,H,W]
            elif t.shape[0] == 3 or t.shape[0] == 1:
                # ya está [C,H,W]
                pass
            else:
                raise ValueError(f"Forma no reconocida para imagen: {t.shape}")

        else:
            raise ValueError(f"Imagen debe ser 2D o 3D, pero es {t.ndim}D")

        # Si está en [0,1], normalizamos a [-1,1]
        if t.min() >= 0.0 and t.max() <= 1.0:
            t = (t - 0.5) / 0.5   # [0,1] -> [-1,1]

        # Añadir dimensión batch: [1,C,H,W]
        t = t.unsqueeze(0)

        return t.to(device)

    # Comprobar tamaños iguales (H,W)
    if img_ref_np.shape[:2] != img_mosaic_np.shape[:2]:
        raise ValueError(f"Las imágenes deben tener mismo tamaño espacial, "
                         f"pero recibí {img_ref_np.shape[:2]} y {img_mosaic_np.shape[:2]}")

    x = to_tensor(img_ref_np)
    y = to_tensor(img_mosaic_np)

    # with torch.no_grad() deshabilita el cálculo de gradientes (más rápido y menos memoria)
    with torch.no_grad():
        d = model(x, y)

    return float(d)