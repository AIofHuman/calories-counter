from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional, Dict, Any, List, Union

import pandas as pd
import torch
from torch.utils.data import Dataset
from PIL import Image
import re
import random


_INGR_RE = re.compile(r"ingr_(\d+)")

class DishDataset(Dataset):
    def __init__(
            self, dish_df: pd.DataFrame, images_root: str, ingr_df: pd.DataFrame,
            train = True,
            image_transform = None, 
            text_transform = None
        ):
        # Filter by required split
        split_value = 'train' if train else 'test'
        filtered_df = dish_df[dish_df['split'] == split_value].reset_index(drop=True)
        images_root = Path(images_root)
        
        # Filter rows by the presence of the corresponding image
        def img_exists(row):
            dish_id = row["dish_id"]
            img_path = images_root / str(dish_id) / "rgb.png"
            return img_path.exists()
        
        filtered_df = filtered_df[filtered_df.apply(img_exists, axis=1)].reset_index(drop=True)
        self.dish_df = filtered_df

        self.images_root = Path(images_root)
        self.ingr_map = dict(zip(ingr_df['id'], ingr_df['ingr']))
        self.image_transform = image_transform
        self.text_transform = text_transform
        self.train = train

    def __len__(self):
        return len(self.dish_df)

    def _parse_ingredient_ids(self, ingredients_str: str) -> List[str]:
        """
        "ingr_0000000122;ingr_0000000026" -> ['nuts', 'carrot', 'meet']
        """
        if ingredients_str is None or not isinstance(ingredients_str, str) or ingredients_str.strip() == "":
            return []

        ingr_ids = []
        for match in _INGR_RE.finditer(ingredients_str):
            try:
                ingr_id = int(match.group(1))
                ingr_ids.append(ingr_id)
            except (ValueError, IndexError) as e:
                # log errors
                continue
        ingr_names = [self.ingr_map.get(i, f"<UNK_{i}>") for i in ingr_ids]

        return ingr_names 

    def __getitem__(self, idx: int):
        row = self.dish_df.iloc[idx]
        dish_id = row["dish_id"]
        img_path = self.images_root / str(dish_id) / "rgb.png"
        image = Image.open(img_path).convert("RGB")

        if self.image_transform:
            image = self.image_transform(image)

        ingr_ids = row["ingredients"]
        text = self._parse_ingredient_ids(ingr_ids)

        if self.text_transform:
            text = self.text_transform(text)

        y = torch.tensor(row["total_calories"], dtype=torch.float32)
        return {"image": image, "text": text, "y": y, "dish_id": dish_id}


if __name__ == "__main__":
    # Self-test:
    DISH_DF_PATH = "./data/dish.csv"
    INGR_DF_PATH = "./data/ingredients.csv"
    IMAGES_ROOT = "./data/images"

    # Load data
    dish_df = pd.read_csv(DISH_DF_PATH)
    ingr_df = pd.read_csv(INGR_DF_PATH)

    # Create test dataset
    test_dataset = DishDataset(
        dish_df=dish_df,
        images_root=IMAGES_ROOT,
        ingr_df=ingr_df,
        train=False
    )
    print(f"There are {len(test_dataset)} dishes in the test set")
        
    # Select two random dishes
    num_samples = 2
    if len(test_dataset) < num_samples:
        print(f"There are less than {num_samples} dishes in the test set")
        num_samples = len(test_dataset)

    random.seed(42)  # For reproducibility
    indices = random.sample(range(len(test_dataset)), num_samples)

    for i in indices:
        sample = test_dataset[i]
        print(f"\n--- Sample {i} ---")
        print(f"Dish ID: {test_dataset.dish_df.iloc[i]['dish_id']}")
        print(f"Ingredients: {sample['text']}")
        print(f"Calories: {sample['y'].item()}")
        print(f"Idx: {sample['dish_id']}")
        print("Open image...")
        sample['image'].show() 
    
