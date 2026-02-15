# src/data/transforms.py
from __future__ import annotations


from dataclasses import dataclass
from typing import List, Sequence, Optional
import random

from torchvision import transforms

# =========================
# Image size constants
# =========================
IMAGE_HEIGHT: int = 224
IMAGE_WIDTH: int = 224
RESIZE_SHORT_SIDE: int = 256

# =========================
# Image normalization constants (ImageNet)
# =========================
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD  = (0.229, 0.224, 0.225)

# =========================
# Image augmentation hyperparameters
# =========================
RRC_SCALE = (0.75, 1.0)
RRC_RATIO = (0.90, 1.10)

HFLIP_P: float = 0.5
ROTATE_DEGREES: int = 10

CJ_BRIGHTNESS: float = 0.20
CJ_CONTRAST: float = 0.20
CJ_SATURATION: float = 0.15
CJ_HUE: float = 0.02


def build_train_image_transform(normalize: bool = True) -> transforms.Compose:
    ops = [
        transforms.RandomResizedCrop(
            size=(IMAGE_HEIGHT, IMAGE_WIDTH),
            scale=RRC_SCALE,
            ratio=RRC_RATIO,
        ),
        transforms.RandomHorizontalFlip(p=HFLIP_P),
        transforms.RandomRotation(degrees=ROTATE_DEGREES),
        transforms.ColorJitter(
            brightness=CJ_BRIGHTNESS,
            contrast=CJ_CONTRAST,
            saturation=CJ_SATURATION,
            hue=CJ_HUE,
        ),
        transforms.ToTensor(),
    ]
    if normalize:
        ops.append(transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD))
    return transforms.Compose(ops)


def build_val_image_transform(normalize: bool = True) -> transforms.Compose:
    ops = [
        transforms.Resize(RESIZE_SHORT_SIDE),
        transforms.CenterCrop((IMAGE_HEIGHT, IMAGE_WIDTH)),
        transforms.ToTensor(),
    ]
    if normalize:
        ops.append(transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD))
    return transforms.Compose(ops)


# ======================================================================================
# Text (ingredients list) augmentations
# ======================================================================================

@dataclass
class IngredientShuffle:
    """
    Shuffle the order of ingredients (list of tokens).
    Useful if your text encoder is order-sensitive (RNN/Transformer).
    Makes no sense for multi-hot, but doesn't hurt if applied before vectorization.

    p: probability of applying shuffle
    """
    p: float = 0.5
    seed: Optional[int] = None

    def __post_init__(self):
        self._rng = random.Random(self.seed)

    def __call__(self, tokens: Sequence[str]) -> List[str]:
        tokens = list(tokens)
        if len(tokens) <= 1:
            return tokens
        if self._rng.random() > self.p:
            return tokens
        self._rng.shuffle(tokens)
        return tokens


@dataclass
class IngredientDropout:
    """
    Ingredient dropout: randomly drop some of the ingredients.
    This makes the model more robust to incomplete/noisy ingredient lists.

    p_drop: probability to drop each token
    min_left: minimum number of tokens to keep
    """
    p_drop: float = 0.2
    min_left: int = 1
    seed: Optional[int] = None

    def __post_init__(self):
        self._rng = random.Random(self.seed)

    def __call__(self, tokens: Sequence[str]) -> List[str]:
        tokens = list(tokens)
        if not tokens:
            return tokens
        kept = [t for t in tokens if self._rng.random() > self.p_drop]
        if len(kept) < self.min_left:
            kept = tokens[: self.min_left]
        return kept


class ComposeText:
    """
    Analog of transforms.Compose, but for list of tokens.
    """
    def __init__(self, ops):
        self.ops = list(ops)

    def __call__(self, tokens: Sequence[str]) -> List[str]:
        out = list(tokens)
        for op in self.ops:
            out = op(out)
        return out


# =========================
# Text augmentation hyperparameters
# =========================
TEXT_SHUFFLE_P: float = 0.5
TEXT_DROPOUT_P: float = 0.2
TEXT_MIN_LEFT: int = 1


def build_train_text_transform(
    shuffle_p: float = TEXT_SHUFFLE_P,
    dropout_p: float = TEXT_DROPOUT_P,
    min_left: int = TEXT_MIN_LEFT,
    seed: Optional[int] = None,
) -> ComposeText:
    """
    Train-time augmentation for ingredients:
      1) Dropout (drop some tokens)
      2) Shuffle (randomize order)

    The order: first downsample, then shuffle leftovers.
    """
    return ComposeText([
        IngredientDropout(p_drop=dropout_p, min_left=min_left, seed=seed),
        IngredientShuffle(p=shuffle_p, seed=seed),
    ])


def build_val_text_transform() -> ComposeText:
    """
    For test, do not apply any augmentation (return tokens as is).
    """
    return ComposeText([])
