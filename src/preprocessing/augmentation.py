import cv2
import numpy as np
import albumentations as A


def get_train_transform() -> A.Compose:
    """
    Training augmentation pipeline.
    Operates on uint8 HxW single-channel arrays.
    """
    return A.Compose([
        # Geometric
        A.HorizontalFlip(p=0.5),
        A.Rotate(limit=5, border_mode=cv2.BORDER_REFLECT, p=0.3),
        A.ShiftScaleRotate(
            shift_limit=0.02, scale_limit=0.05,
            rotate_limit=3, border_mode=cv2.BORDER_REFLECT, p=0.2
        ),
        # Photometric
        A.RandomBrightnessContrast(
            brightness_limit=0.15, contrast_limit=0.15, p=0.5
        ),
        A.GaussianBlur(blur_limit=(3, 5), p=0.2),
        A.GaussNoise(var_limit=(2.0, 12.0), p=0.3),
        # Occlusion simulation
        A.CoarseDropout(
            max_holes=4, max_height=8, max_width=32,
            min_holes=1, fill_value=0, p=0.2
        ),
    ])


def get_val_transform() -> A.Compose:
    """No augmentation — identity pipeline."""
    return A.Compose([])


def augment_strip(strip_float32: np.ndarray,
                  transform: A.Compose) -> np.ndarray:
    """
    Apply an albumentations transform to a float32 strip [0,1].
    Returns augmented float32 strip [0,1].
    """
    uint8 = (strip_float32 * 255).clip(0, 255).astype(np.uint8)
    out   = transform(image=uint8)["image"]
    return out.astype(np.float32) / 255.0