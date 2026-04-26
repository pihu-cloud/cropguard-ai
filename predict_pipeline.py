"""
predict_pipeline.py — CropGuard AI · ResNet50V2 Inference Engine
=================================================================

Key improvements over original:
  - Confidence RESCALING: raw softmax is spread across N classes so even
    the correct class often scores only 10-20%. We rescale to a 0-100%
    user-facing score using temperature sharpening + gap-relative scaling,
    so a clear correct answer reads as 70-95%, not 10%.
  - ImageNet mean/std normalisation (fixes the root cause of low confidence).
  - Aspect-ratio-preserving resize (correct ImageNet eval protocol).
  - Expanded TTA: 8 views instead of 5.
  - Thresholds calibrated for real photos, not studio images.
"""

from __future__ import annotations

import logging
from typing import Any

import cv2
import numpy as np
from PIL import Image

log = logging.getLogger("cropguard.pipeline")

# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

RESIZE_TO: int   = 256
CROP_SIZE: int   = 224

CLAHE_CLIP: float  = 3.0
CLAHE_TILE: tuple  = (8, 8)

BLUR_KERNEL: tuple = (3, 3)
BLUR_SIGMA: float  = 0.8

SEG_MARGIN: int    = 20
SEG_ITER: int      = 5
BG_FILL: float     = 0.5

# ResNet50V2 ImageNet normalisation
# The model was pretrained expecting (pixel/255 - mean) / std
# Simply dividing by 255 gives the wrong input distribution and tanks confidence
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)

# Confidence display thresholds (applied to RESCALED score, not raw softmax)
THRESH_HIGH:   float = 70.0
THRESH_MEDIUM: float = 45.0
THRESH_LOW:    float = 20.0

GAP_WARN:      float = 8.0
COVERAGE_WARN: float = 0.08

# Temperature for sharpening before rescaling (lower = sharper = higher display score)
TEMPERATURE: float = 0.5


# ─────────────────────────────────────────────────────────────────────────────
# 1. PREPROCESSING
# ─────────────────────────────────────────────────────────────────────────────

def _clahe_enhance(img_rgb: np.ndarray) -> np.ndarray:
    lab     = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    clahe   = cv2.createCLAHE(clipLimit=CLAHE_CLIP, tileGridSize=CLAHE_TILE)
    l_eq    = clahe.apply(l)
    return cv2.cvtColor(cv2.merge([l_eq, a, b]), cv2.COLOR_LAB2RGB)


def _gaussian_denoise(img_rgb: np.ndarray) -> np.ndarray:
    return cv2.GaussianBlur(img_rgb, BLUR_KERNEL, sigmaX=BLUR_SIGMA)


def _resize_keep_aspect(pil_img: Image.Image, target: int) -> Image.Image:
    """
    Resize so the SHORT edge equals target, preserving aspect ratio.
    This matches the ImageNet eval protocol used during ResNet pretraining.
    Squashing to a square first distorts leaf shape and hurts accuracy.
    """
    w, h = pil_img.size
    if w < h:
        new_w, new_h = target, int(h * target / w)
    else:
        new_w, new_h = int(w * target / h), target
    return pil_img.resize((new_w, new_h), Image.LANCZOS)


def _center_crop(img_rgb: np.ndarray, size: int = CROP_SIZE) -> np.ndarray:
    h, w = img_rgb.shape[:2]
    y0   = (h - size) // 2
    x0   = (w - size) // 2
    return img_rgb[y0 : y0 + size, x0 : x0 + size]


def _grabcut_segment(img_rgb: np.ndarray) -> tuple[np.ndarray, float]:
    h, w  = img_rgb.shape[:2]
    bgr   = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    mask  = np.zeros((h, w), np.uint8)
    bgd   = np.zeros((1, 65), np.float64)
    fgd   = np.zeros((1, 65), np.float64)
    rect  = (SEG_MARGIN, SEG_MARGIN,
             w - 2 * SEG_MARGIN, h - 2 * SEG_MARGIN)
    try:
        cv2.grabCut(bgr, mask, rect, bgd, fgd, SEG_ITER, cv2.GC_INIT_WITH_RECT)
        fg = np.where(
            (mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD), 1.0, 0.0
        ).astype(np.float32)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        fg     = cv2.morphologyEx(fg, cv2.MORPH_CLOSE, kernel)
        fg     = cv2.GaussianBlur(fg, (7, 7), 0)
        return fg[..., np.newaxis], float(fg.mean())
    except Exception as exc:
        log.debug("GrabCut failed (%s) -- full-image mask", exc)
        return np.ones((h, w, 1), np.float32), 1.0


def _imagenet_normalise(arr_f32: np.ndarray) -> np.ndarray:
    """
    Apply ResNet50V2 ImageNet normalisation: (x - mean) / std per channel.

    This is the single biggest fix for the low confidence problem.
    The model was trained with this normalisation. Without it, every pixel
    value is in the wrong range for the pretrained weights, and the model
    spreads its softmax probability nearly evenly across all classes.

    Example for a green leaf pixel (R=60, G=120, B=40):
        /255 only    -> [0.235, 0.471, 0.157]   wrong distribution
        ImageNet norm-> [-1.09, 0.067, -1.08]   correct distribution
    """
    return (arr_f32 - IMAGENET_MEAN) / IMAGENET_STD


def preprocess_image(
    image_path: str,
    use_segmentation: bool = True,
) -> tuple[np.ndarray, dict]:
    """
    Full preprocessing pipeline for real-world leaf images.

    Steps
    -----
    1. Load as RGB
    2. Aspect-ratio-preserving resize (short edge to 256)
    3. CLAHE enhancement
    4. Gaussian denoising
    5. Centre crop to 224x224
    6. Optional GrabCut segmentation
    7. /255 normalise to [0,1]
    8. ImageNet channel normalisation
    """
    pil  = Image.open(image_path).convert('RGB')
    orig = pil.size

    pil_r  = _resize_keep_aspect(pil, RESIZE_TO)
    arr_u8 = np.array(pil_r, dtype=np.uint8)

    arr_clahe = _clahe_enhance(arr_u8)
    arr_blur  = _gaussian_denoise(arr_clahe)
    arr_crop  = _center_crop(arr_blur, CROP_SIZE)

    leaf_coverage = 1.0
    low_coverage  = False
    if use_segmentation:
        mask, leaf_coverage = _grabcut_segment(arr_crop)
        low_coverage        = leaf_coverage < COVERAGE_WARN
        arr_f = arr_crop.astype(np.float32) / 255.0
        arr_f = arr_f * mask + BG_FILL * (1.0 - mask)
    else:
        arr_f = arr_crop.astype(np.float32) / 255.0

    arr_norm = _imagenet_normalise(arr_f)

    debug_info: dict[str, Any] = {
        "original_size": list(orig),
        "leaf_coverage": round(leaf_coverage * 100.0, 1),
        "segmentation":  use_segmentation,
        "low_coverage":  low_coverage,
        "clahe_clip":    CLAHE_CLIP,
        "blur_sigma":    BLUR_SIGMA,
    }
    return arr_norm, debug_info


# ─────────────────────────────────────────────────────────────────────────────
# 2. TEST-TIME AUGMENTATION (8 views)
# ─────────────────────────────────────────────────────────────────────────────

def _tta_views(arr: np.ndarray) -> list[np.ndarray]:
    """
    8 augmented views of the normalised (224,224,3) array.

    0  original
    1  horizontal flip
    2  brighter (+0.15 in normalised space)
    3  darker  (-0.15 in normalised space)
    4  90 deg CW
    5  90 deg CCW
    6  sharpened  (enhances disease lesion texture)
    7  slightly blurred (handles soft-focus real photos)
    """
    sharp_kernel = np.array([[0, -0.5, 0],
                              [-0.5, 3.0, -0.5],
                              [0, -0.5, 0]], dtype=np.float32)

    sharpened = cv2.filter2D(arr, -1, sharp_kernel).astype(np.float32)
    blurred   = cv2.GaussianBlur(arr, (5, 5), sigmaX=1.0).astype(np.float32)

    return [
        arr,
        arr[:, ::-1, :].copy(),
        np.clip(arr + 0.15, -3.0, 3.0).astype(np.float32),
        np.clip(arr - 0.15, -3.0, 3.0).astype(np.float32),
        np.rot90(arr, k=1).copy(),
        np.rot90(arr, k=3).copy(),
        sharpened,
        blurred,
    ]


# ─────────────────────────────────────────────────────────────────────────────
# 3. INFERENCE
# ─────────────────────────────────────────────────────────────────────────────

def _run_model(model, batch: np.ndarray) -> np.ndarray:
    return model.predict(batch, verbose=0)[0]


def _tta_predict(model, arr: np.ndarray) -> np.ndarray:
    """Run 8-view TTA and return averaged raw probability vector."""
    view_probs = []
    for view in _tta_views(arr):
        batch = np.expand_dims(view, axis=0)
        view_probs.append(_run_model(model, batch))
    return np.mean(view_probs, axis=0)


# ─────────────────────────────────────────────────────────────────────────────
# 4. CONFIDENCE RESCALING
# ─────────────────────────────────────────────────────────────────────────────

def _rescale_confidence(probs: np.ndarray) -> tuple[float, np.ndarray]:
    """
    Convert raw softmax probabilities into a user-facing 0-100% score.

    WHY raw softmax is misleading as a display number:
        With 10 classes and a real-world image, softmax spreads probability
        across classes. The correct class might only score 15% raw — which
        reads as "the model is 15% sure" but actually means "this is the
        most likely class by a clear margin over all others at ~2% each".

    HOW rescaling works:

    Step 1 — Temperature sharpening:
        Raise each probability to power (1/T) and renormalise.
        With T=0.5 this squeezes the distribution so the winner gets
        a higher share. This is mathematically equivalent to applying
        softmax with temperature T on the original logits.

    Step 2 — Gap-relative display score:
        gap_ratio = (p1 - p2) / p1
        display   = 20 + 75 * gap_ratio^0.6

        Anchors:
            gap_ratio = 0.0  (completely tied)    -> display = 20%
            gap_ratio = 0.5  (moderate lead)      -> display ~62%
            gap_ratio = 0.9  (very clear winner)  -> display ~91%

    This is a display heuristic, not statistical calibration.
    It makes the number intuitive without claiming false precision.

    Returns (display_conf 0-100, sharpened_probs)
    """
    p_exp = np.power(np.clip(probs, 1e-10, 1.0), 1.0 / TEMPERATURE)
    sharp = p_exp / p_exp.sum()

    sorted_s     = np.sort(sharp)[::-1]
    p1, p2       = float(sorted_s[0]), float(sorted_s[1])
    gap_ratio    = (p1 - p2) / (p1 + 1e-10)
    display_conf = 20.0 + 75.0 * (gap_ratio ** 0.6)
    display_conf = float(np.clip(display_conf, 5.0, 99.0))

    return round(display_conf, 1), sharp


# ─────────────────────────────────────────────────────────────────────────────
# 5. SCORING
# ─────────────────────────────────────────────────────────────────────────────

def _confidence_level(conf: float) -> str:
    if conf >= THRESH_HIGH:   return 'high'
    if conf >= THRESH_MEDIUM: return 'med'
    return 'low'


def _score_prediction(probs: np.ndarray) -> dict:
    display_conf, sharp_probs = _rescale_confidence(probs)

    sorted_sharp = np.sort(sharp_probs)[::-1]
    top_idx      = int(np.argmax(probs))
    conf_gap     = float(sorted_sharp[0] - sorted_sharp[1]) * 100.0
    uncertain    = display_conf < THRESH_LOW
    ambiguous    = (not uncertain) and (conf_gap < GAP_WARN)

    return {
        "top_idx":     top_idx,
        "confidence":  display_conf,
        "raw_conf":    round(float(probs[top_idx]) * 100.0, 2),
        "conf_level":  _confidence_level(display_conf),
        "conf_gap":    round(conf_gap, 2),
        "uncertain":   uncertain,
        "ambiguous":   ambiguous,
        "sharp_probs": sharp_probs,
    }


# ─────────────────────────────────────────────────────────────────────────────
# 6. TOP-3 BUILDER
# ─────────────────────────────────────────────────────────────────────────────

def _build_top3(
    probs: np.ndarray,
    sharp_probs: np.ndarray,
    class_names: list[str],
    disease_info: dict,
    make_fallback_fn,
) -> list[dict]:
    top3_idx    = np.argsort(probs)[-3:][::-1]
    top1_disp, _ = _rescale_confidence(probs)
    top1_raw    = float(probs[top3_idx[0]])
    result      = []

    for rank, i in enumerate(top3_idx):
        label = class_names[int(i)]
        info  = disease_info.get(label) or make_fallback_fn(label)
        if rank == 0:
            disp_i = top1_disp
        else:
            ratio  = (float(probs[int(i)]) / (top1_raw + 1e-10)) ** 0.7
            disp_i = round(top1_disp * ratio, 1)

        result.append({
            "label":      label,
            "name":       info['name'],
            "crop":       info['crop'],
            "confidence": disp_i,
        })
    return result


# ─────────────────────────────────────────────────────────────────────────────
# 7. USER-FACING MESSAGE
# ─────────────────────────────────────────────────────────────────────────────

def _build_message(
    uncertain: bool,
    ambiguous: bool,
    confidence: float,
    conf_gap: float,
    top3: list[dict],
    low_coverage: bool,
) -> str | None:
    if uncertain:
        return (
            "Confidence is low. Please upload a clearer, close-up image "
            "of the affected leaf in good natural lighting."
        )
    if low_coverage:
        return (
            "Very little leaf tissue detected. "
            "Try reframing so the leaf fills most of the frame."
        )
    if ambiguous:
        return (
            f"The model is closely split between '{top3[0]['name']}' "
            f"({top3[0]['confidence']:.0f}%) and "
            f"'{top3[1]['name']}' ({top3[1]['confidence']:.0f}%). "
            "A second image from a different angle may help."
        )
    if confidence < THRESH_MEDIUM:
        return (
            "Moderate confidence. The diagnosis is likely correct "
            "but a second image would increase certainty."
        )
    return None


# ─────────────────────────────────────────────────────────────────────────────
# 8. DEMO FALLBACK
# ─────────────────────────────────────────────────────────────────────────────

def _demo_probs(image_path: str, num_classes: int) -> np.ndarray:
    img_small = Image.open(image_path).convert('RGB').resize((32, 32))
    pixel_sum = int(np.array(img_small, dtype=np.uint32).sum())
    base      = pixel_sum % num_classes
    remainder = max(num_classes - 3, 1)
    probs     = np.full(num_classes, 0.06 / remainder, dtype=np.float32)
    probs[base]                     = 0.68
    probs[(base + 1) % num_classes] = 0.18
    probs[(base + 2) % num_classes] = 0.08
    return probs


# ─────────────────────────────────────────────────────────────────────────────
# 9. PUBLIC API
# ─────────────────────────────────────────────────────────────────────────────

def predict_disease(
    image_path: str,
    models: dict,
    class_names: list[str],
    disease_info: dict,
    make_fallback_fn,
    use_segmentation: bool = True,
) -> dict:
    """Full prediction pipeline. Returns display-ready 0-100% confidence."""
    num_classes = len(class_names)

    arr, debug_info = preprocess_image(image_path, use_segmentation)
    low_coverage    = debug_info.get("low_coverage", False)

    if models:
        model      = list(models.values())[0]
        probs      = _tta_predict(model, arr)
        model_used = "ResNet50V2 (TTA x8)"
        demo_mode  = False
    else:
        probs      = _demo_probs(image_path, num_classes)
        model_used = "Demo Mode -- place resnet.h5 next to app.py"
        demo_mode  = True

    scores      = _score_prediction(probs)
    top_idx     = scores["top_idx"]
    confidence  = scores["confidence"]
    conf_gap    = scores["conf_gap"]
    uncertain   = scores["uncertain"]
    ambiguous   = scores["ambiguous"]
    sharp_probs = scores["sharp_probs"]

    top3 = _build_top3(probs, sharp_probs, class_names, disease_info, make_fallback_fn)

    label = class_names[min(max(top_idx, 0), num_classes - 1)]
    info  = disease_info.get(label) or make_fallback_fn(label)

    symptoms = info.get('symptoms', [])
    if isinstance(symptoms, str):
        symptoms = [symptoms]

    message = _build_message(
        uncertain, ambiguous, confidence, conf_gap, top3, low_coverage
    )

    return {
        "label":       label,
        "disease":     info['name'],
        "crop":        info['crop'],

        "confidence":  confidence,
        "raw_conf":    scores["raw_conf"],
        "conf_level":  scores["conf_level"],
        "conf_gap":    conf_gap,

        "uncertain":   uncertain,
        "ambiguous":   ambiguous,
        "message":     message,

        "description": info['description'],
        "symptoms":    symptoms,
        "treatment":   info.get('treatment', []),
        # Cap severity based on confidence:
        # - Below THRESH_LOW (20%)  → Uncertain
        # - Below THRESH_MEDIUM (45%) → max Medium (don't show High/Critical on low conf)
        # - Below THRESH_HIGH (70%) → max High (don't show Critical on medium conf)
        # - Above THRESH_HIGH       → show true severity
        "severity":    (
            'Uncertain' if uncertain else
            'Medium'    if confidence < THRESH_MEDIUM and info['severity'] in ('High', 'Critical') else
            'High'      if confidence < THRESH_HIGH   and info['severity'] == 'Critical' else
            info['severity']
        ),
        "is_healthy":  'healthy' in label.lower(),

        "top3":        top3,

        "model_used":    model_used,
        "demo_mode":     demo_mode,
        "num_classes":   num_classes,
        "preprocessing": debug_info,
    }
