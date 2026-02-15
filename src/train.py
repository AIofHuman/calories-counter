# src/training/train_mm.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Any, Optional

import pandas as pd
from functools import partial
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
import yaml


from src.seed import seed_everything
from src.multimodal import MultiModalRegressor
from src.transform import build_train_image_transform, build_val_image_transform, build_train_text_transform, build_val_text_transform
from src.dataset import DishDataset

# from seed import seed_everything
# from multimodal import MultiModalRegressor
# from transform import build_train_image_transform, build_val_image_transform, build_train_text_transform, build_val_text_transform
# from dataset import DishDataset


def mae(pred: torch.Tensor, target: torch.Tensor) -> float:
    return torch.mean(torch.abs(pred - target)).item()


def build_vocab_from_ingredients_csv(ingr_df: pd.DataFrame) -> Dict[str, int]:
    # name -> index
    names = ingr_df["ingr"].astype(str).tolist()
    return {name: i for i, name in enumerate(names)}

def collate(batch: List[Dict[str, Any]], name_to_idx: Dict[str, int], n_ingr: int) -> Dict[str, Any]:
    images = torch.stack([b["image"] for b in batch], dim=0)
    x = torch.zeros((len(batch), n_ingr), dtype=torch.float32)
    for i, b in enumerate(batch):
        tokens = b["text"]  # list[str]
        for t in tokens:
            j = name_to_idx.get(t)
            if j is not None:
                x[i, j] = 1.0

    y = torch.stack([b["y"] for b in batch], dim=0).float().view(-1)
    dish_id = [b["dish_id"] for b in batch]
    return {"image": images, "ingr": x, "y": y, "dish_id": dish_id}

@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> float:
    model.eval()
    total_abs = 0.0
    total_n = 0
    for batch in loader:
        image = batch["image"].to(device, non_blocking=True)
        ingr = batch["ingr"].to(device, non_blocking=True)
        y = batch["y"].to(device, non_blocking=True)

        pred = model(image, ingr)
        total_abs += torch.sum(torch.abs(pred - y)).item()
        total_n += y.numel()
    return total_abs / max(1, total_n)


def train(cfg: dict) -> dict:
    """
    Запускать из ноутбука:
      from src.training.train_mm import train
      results = train(cfg)

    cfg: dict из yaml
    Возвращает dict с итоговыми метриками и путями к артефактам.
    """
    # --- seed
    seed_everything(cfg["seed"], deterministic=cfg["deterministic"])

    # --- device
    device = torch.device("cuda" if torch.cuda.is_available() and cfg["device"] == "cuda" else "cpu")

    # --- paths
    data_dir = Path(cfg["data"]["dir"])
    dish_path = data_dir / cfg["data"]["dish_csv"]
    ingr_path = data_dir / cfg["data"]["ingredients_csv"]
    images_root = data_dir / cfg["data"]["images_dir"]

    out_dir = Path(cfg["output"]["dir"])
    ckpt_dir = out_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # --- read data
    dish_df = pd.read_csv(dish_path)
    ingr_df = pd.read_csv(ingr_path)

    # vocab
    name_to_idx = build_vocab_from_ingredients_csv(ingr_df)
    n_ingr = len(name_to_idx)

    
    train_img_tfm = build_train_image_transform(normalize=cfg["data"]["normalize"])
    val_img_tfm = build_val_image_transform(normalize=cfg["data"]["normalize"])

    # текстовые аугментации применяем только на train
    train_txt_tfm = build_train_text_transform(
        shuffle_p=cfg["text_aug"]["shuffle_p"],
        dropout_p=cfg["text_aug"]["dropout_p"],
        min_left=cfg["text_aug"]["min_left"],
        seed=cfg["seed"],
    )
    val_txt_tfm = build_val_text_transform()

    train_ds_full = DishDataset(
        dish_df=dish_df,
        images_root=str(images_root),
        ingr_df=ingr_df,
        train=True,
        image_transform=train_img_tfm,
        text_transform=train_txt_tfm,
    )

    # внутренний val split из train 0.2
    val_frac = cfg["data"]["val_frac"]
    n_total = len(train_ds_full)
    n_val = int(n_total * val_frac)
    n_train = n_total - n_val
    gen = torch.Generator().manual_seed(cfg["seed"])
    train_ds, val_ds = torch.utils.data.random_split(train_ds_full, [n_train, n_val], generator=gen)

    # отдельный test_ds из split=test (без аугментаций)
    test_ds = DishDataset(
        dish_df=dish_df,
        images_root=str(images_root),
        ingr_df=ingr_df,
        train=False,
        image_transform=val_img_tfm,
        text_transform=val_txt_tfm,
    )

    # --- loaders
    collate_fn = partial(collate, name_to_idx=name_to_idx, n_ingr=n_ingr)

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg["train"]["batch_size"],
        shuffle=True,
        num_workers=cfg["train"]["num_workers"],
        pin_memory=True,
        collate_fn=collate_fn,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg["train"]["batch_size"],
        shuffle=False,
        num_workers=cfg["train"]["num_workers"],
        pin_memory=True,
        collate_fn=collate_fn,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=cfg["train"]["batch_size"],
        shuffle=False,
        num_workers=cfg["train"]["num_workers"],
        pin_memory=True,
        collate_fn=collate_fn,
    )

    # --- model
    model = MultiModalRegressor(
        n_ingr=n_ingr,
        image_model=cfg["model"]["image_backbone"],
        image_pretrained=cfg["model"]["image_pretrained"],
        img_dim=cfg["model"]["img_dim"],
        txt_hidden=cfg["model"]["txt_hidden"],
        txt_dim=cfg["model"]["txt_dim"],
        head_hidden=cfg["model"]["head_hidden"],
        head_dropout=cfg["model"]["dropout"],
    ).to(device)

    # --- loss / optim
    if cfg["train"]["loss"] == "l1":
        criterion = nn.L1Loss()
    else:
        criterion = nn.SmoothL1Loss(beta=cfg["train"]["huber_beta"])

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg["train"]["lr"],
        weight_decay=cfg["train"]["weight_decay"],
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=cfg["train"]["epochs"]
    )

    # --- train loop
    best_val = float("inf")
    best_path = None
    patience = cfg["train"]["early_stopping_patience"]
    patience_left = patience

    history = {"train_mae": [], "val_mae": [], "lr": []}

    for epoch in range(1, cfg["train"]["epochs"] + 1):
        model.train()
        running_abs = 0.0
        running_n = 0
        running_loss = 0.0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{cfg['train']['epochs']}", leave=False)
        for batch in pbar:
            image = batch["image"].to(device, non_blocking=True)
            ingr = batch["ingr"].to(device, non_blocking=True)
            y = batch["y"].to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            pred = model(image, ingr)
            loss = criterion(pred, y)
            loss.backward()

            if cfg["train"]["grad_clip"] > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["train"]["grad_clip"])

            optimizer.step()

            running_loss += loss.item() * y.numel()
            running_abs += torch.sum(torch.abs(pred.detach() - y)).item()
            running_n += y.numel()

            pbar.set_postfix(
                loss=running_loss / max(1, running_n),
                mae=running_abs / max(1, running_n),
                lr=optimizer.param_groups[0]["lr"],
            )

        train_mae = running_abs / max(1, running_n)
        val_mae = evaluate(model, val_loader, device)

        scheduler.step()
        lr_now = optimizer.param_groups[0]["lr"]

        history["train_mae"].append(train_mae)
        history["val_mae"].append(val_mae)
        history["lr"].append(lr_now)

        print(f"Epoch {epoch:03d} | train MAE: {train_mae:.4f} | val MAE: {val_mae:.4f} | lr: {lr_now:.2e}")

        # early stopping + save best
        if val_mae < best_val - cfg["train"]["min_delta"]:
            best_val = val_mae
            patience_left = patience
            best_path = ckpt_dir / f"best_epoch_{epoch:03d}_valMAE_{val_mae:.4f}.pt"
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "cfg": cfg,
                    "epoch": epoch,
                    "best_val_mae": best_val,
                    "n_ingr": n_ingr,
                },
                best_path,
            )
        else:
            patience_left -= 1
            if patience_left <= 0:
                print(f"Early stopping: no improvement for {patience} epochs.")
                break

    # load best and evaluate test
    if best_path is not None and best_path.exists():
        ckpt = torch.load(best_path, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])

    test_mae = evaluate(model, test_loader, device)
    print(f"TEST MAE: {test_mae:.4f}")

    # save history
    out_metrics_path = out_dir / "metrics.yaml"
    with open(out_metrics_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(
            {
                "best_val_mae": float(best_val),
                "test_mae": float(test_mae),
                "history": history,
                "best_checkpoint": str(best_path) if best_path else None,
            },
            f,
            allow_unicode=True,
            sort_keys=False,
        )

    return {
        "best_val_mae": float(best_val),
        "test_mae": float(test_mae),
        "best_checkpoint": str(best_path) if best_path else None,
        "metrics_path": str(out_metrics_path),
    }


def load_cfg(path: str | Path) -> dict:
    path = Path(path)
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


if __name__ == "__main__":
    
    config_path = "./config/effinet_multihot.yaml"
    print(f"Загружаем config: {config_path}")
    cfg = load_cfg(config_path)

    # Можно переопределить директорию вывода на временную или отдельную для тестов
    # cfg["output"]["dir"] = "test_outputs"

    results = train(cfg)
    print("Результаты обучения:")
    print(results)
