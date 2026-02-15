from pathlib import Path
import torch
import pandas as pd
from functools import partial
from torch.utils.data import DataLoader
from src.dataset import DishDataset
from src.transform import build_val_image_transform, build_val_text_transform
from src.multimodal import MultiModalRegressor
from src.train import build_vocab_from_ingredients_csv, mae, collate

def load_model(checkpoint_path, device='cpu'):
    ckpt = torch.load(checkpoint_path, map_location=device)
    model = MultiModalRegressor(
        n_ingr=ckpt["n_ingr"],
        image_model=ckpt["cfg"]["model"]["image_backbone"],
        image_pretrained=ckpt["cfg"]["model"]["image_pretrained"],
        img_dim=ckpt["cfg"]["model"]["img_dim"],
        txt_hidden=ckpt["cfg"]["model"]["txt_hidden"],
        txt_dim=ckpt["cfg"]["model"]["txt_dim"],
        head_hidden=ckpt["cfg"]["model"]["head_hidden"],
        head_dropout=ckpt["cfg"]["model"]["dropout"],
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model, ckpt["cfg"]

@torch.no_grad()
def predict_on_test(checkpoint_path, data_dir="./data", batch_size=128, device='cpu'):
    
    model, cfg = load_model(checkpoint_path, device)
    data_dir = Path(data_dir)
    dish_path = data_dir / cfg["data"]["dish_csv"]
    ingr_path = data_dir / cfg["data"]["ingredients_csv"]
    images_root = data_dir / cfg["data"]["images_dir"]
    dish_df = pd.read_csv(dish_path)
    ingr_df = pd.read_csv(ingr_path)
    name_to_idx = build_vocab_from_ingredients_csv(ingr_df)
    n_ingr = len(name_to_idx)

    val_img_tfm = build_val_image_transform(normalize=cfg["data"]["normalize"])
    val_txt_tfm = build_val_text_transform()
    test_ds = DishDataset(
        dish_df=dish_df,
        images_root=str(images_root),
        ingr_df=ingr_df,
        train=False,
        image_transform=val_img_tfm,
        text_transform=val_txt_tfm,
    )
    
    collate_fn = partial(collate, name_to_idx=name_to_idx, n_ingr=n_ingr)
    loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    all_preds, all_targets, all_idxs = [], [], []

    for batch in loader:
        image = batch["image"].to(device)
        ingr = batch["ingr"].to(device)
        y = batch["y"].to(device)
        pred = model(image, ingr)
        all_preds.append(pred.cpu())
        all_targets.append(y.cpu())
        all_idxs.extend(batch["dish_id"])
    
    all_preds = torch.cat(all_preds).numpy()
    all_targets = torch.cat(all_targets).numpy()

    # Test MAE
    test_mae = mae(torch.tensor(all_preds), torch.tensor(all_targets))
    return all_preds, all_targets, all_idxs, test_mae, test_ds, dish_df
