import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
from PIL import Image

from src.transform import (
    build_train_image_transform,
    build_val_image_transform,
    build_train_text_transform,
    build_val_text_transform,
    IngredientShuffle,
    IngredientDropout,
)

def create_dummy_image():
    # Вернет dummy Image 256x256
    arr = np.random.randint(0, 256, (256, 256, 3), dtype=np.uint8)
    return Image.fromarray(arr)

def test_train_image_transform_shape():
    img = create_dummy_image()
    transform = build_train_image_transform()
    tensor = transform(img)
    # Ожидается тензор 3x224x224
    assert tensor.shape == (3, 224, 224)

def test_val_image_transform_shape():
    img = create_dummy_image()
    transform = build_val_image_transform()
    tensor = transform(img)
    # Ожидается тензор 3x224x224
    assert tensor.shape == (3, 224, 224)

def test_train_text_transform_shuffle_and_dropout():
    tokens = ["salt", "milk", "sugar", "flour", "oil"]
    transform = build_train_text_transform(shuffle_p=1.0, dropout_p=0.4, min_left=2, seed=42)
    out = transform(tokens)
    assert 2 <= len(out) <= 5
    assert set(out).issubset(set(tokens))

def test_val_text_transform_noop():
    tokens = ["a", "b", "c"]
    transform = build_val_text_transform()
    out = transform(tokens)
    assert out == tokens

def test_ingredient_shuffle_works():
    tokens = ["a", "b", "c"]
    shuffler = IngredientShuffle(p=1.0, seed=123)
    out = shuffler(tokens)
    assert out != tokens
    assert sorted(out) == sorted(tokens)

def test_ingredient_dropout_min_left():
    tokens = ["x", "y", "z"]
    dropout = IngredientDropout(p_drop=1.0, min_left=2, seed=2)
    out = dropout(tokens)
    assert len(out) == 2
    assert set(out).issubset(set(tokens))

def test_ingredient_dropout_empty():
    dropout = IngredientDropout()
    out = dropout([])
    assert out == []

def test_ingredient_dropout_only_one():
    dropout = IngredientDropout()
    out = dropout(['1'])
    assert out == ['1']