import json
import os
from pathlib import Path

import torch
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2


# -------------------------  shared helpers ------------------------- #
def _ensure_chw_tensor(arr_or_tensor):
    """
    Accepts either a NumPy array (H×W×C or H×W) or a torch.Tensor that is
    already CHW/HCW. Returns a float32 tensor in CHW order with values ∈ [0, 1].
    """
    if isinstance(arr_or_tensor, torch.Tensor):
        tensor = arr_or_tensor
    elif isinstance(arr_or_tensor, np.ndarray):
        tensor = torch.from_numpy(arr_or_tensor)
    else:
        raise TypeError(f"Unsupported type {type(arr_or_tensor)}")

    # Convert HWC → CHW if needed
    if tensor.ndim == 3 and tensor.shape[0] != 1 and tensor.shape[0] != 3:
        tensor = tensor.permute(2, 0, 1)          # HWC → CHW
    elif tensor.ndim == 2:                        # (H, W) → (1, H, W)
        tensor = tensor.unsqueeze(0)

    return tensor.float() / 255.0
# ------------------------------------------------------------------- #


class PolygonColorDataset(Dataset):
    """Dataset for the real polygon-colorization task."""

    def __init__(
        self,
        data_dir: str | Path,
        split: str = "training",
        transform: A.BasicTransform | None = None,
        image_size: int = 128,
    ):
        self.data_dir = Path(data_dir)
        self.split = split
        self.image_size = image_size

        # ─── Load data mapping ────────────────────────────────────────
        json_path = self.data_dir / split / "data.json"
        with open(json_path, "r") as f:
            self.data = json.load(f)

        # ─── Color mapping ────────────────────────────────────────────
        self.colors = {
            "red": 0,
            "blue": 1,
            "green": 2,
            "yellow": 3,
            "orange": 4,
            "purple": 5,
            "pink": 6,
            "brown": 7,
            "black": 8,
            "white": 9,
        }
        self.num_colors = len(self.colors)

        # ─── Default transforms ───────────────────────────────────────
        if transform is None:
            if split == "training":
                transform = A.Compose(
                    [
                        A.Resize(image_size, image_size),
                        A.HorizontalFlip(p=0.5),
                        A.Rotate(limit=30, p=0.5),
                        A.RandomBrightnessContrast(p=0.2),
                        A.Normalize(
                            mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225],
                        ),
                        ToTensorV2(),
                    ]
                )
            else:
                transform = A.Compose(
                    [
                        A.Resize(image_size, image_size),
                        A.Normalize(
                            mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225],
                        ),
                        ToTensorV2(),
                    ]
                )
        self.transform = transform

    # ------------------------------ Dataset API ------------------------------ #
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]

        # ---- Load images (PIL → ndarray) ------------------------------------ #
        inp_path = self.data_dir / self.split / "inputs" / item["input"]
        out_path = self.data_dir / self.split / "outputs" / item["output"]

        inp_img = np.array(Image.open(inp_path).convert("RGB"))
        out_img = np.array(Image.open(out_path).convert("RGB"))

        # ---- Color label ---------------------------------------------------- #
        color_name = item["color"].lower()
        color_idx = self.colors.get(color_name, 0)  # default “red” if unknown

        # ---- Transforms ----------------------------------------------------- #
        if self.transform:
            transformed = self.transform(image=inp_img, mask=out_img)
            inp_tensor = _ensure_chw_tensor(transformed["image"])
            out_tensor = _ensure_chw_tensor(transformed["mask"])
        else:
            inp_tensor = _ensure_chw_tensor(inp_img)
            out_tensor = _ensure_chw_tensor(out_img)

        return {
            "input": inp_tensor,
            "output": out_tensor,
            "color_idx": torch.tensor(color_idx, dtype=torch.long),
            "color_name": color_name,
        }


class SyntheticPolygonDataset(Dataset):
    """Procedurally generates polygon–color pairs on the fly."""

    def __init__(
        self,
        num_samples: int = 1000,
        image_size: int = 128,
        transform: A.BasicTransform | None = None,
    ):
        self.num_samples = num_samples
        self.image_size = image_size

        # ---- Color definitions ------------------------------------------------ #
        self.colors_rgb = {
            "red": [255, 0, 0],
            "blue": [0, 0, 255],
            "green": [0, 255, 0],
            "yellow": [255, 255, 0],
            "orange": [255, 165, 0],
            "purple": [128, 0, 128],
            "pink": [255, 192, 203],
            "brown": [165, 42, 42],
            "black": [0, 0, 0],
            "white": [255, 255, 255],
        }
        self.color_names = list(self.colors_rgb.keys())
        self.num_colors = len(self.color_names)

        # ---- Default transforms ---------------------------------------------- #
        if transform is None:
            transform = A.Compose(
                [
                    A.HorizontalFlip(p=0.5),
                    A.Rotate(limit=30, p=0.5),
                    A.RandomBrightnessContrast(p=0.2),
                    A.Normalize(
                        mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225],
                    ),
                    ToTensorV2(),
                ]
            )
        self.transform = transform

    # ------------------------------ Utils ------------------------------ #
    def _generate_polygon(self, n_sides=3):
        """Random regular polygon (with jittered angle)."""
        center = np.array([self.image_size // 2, self.image_size // 2])
        radius = np.random.uniform(20, 40)
        angles = np.linspace(0, 2 * np.pi, n_sides, endpoint=False)
        angles += np.random.uniform(0, 2 * np.pi / n_sides)

        verts = np.stack(
            [center[0] + radius * np.cos(angles), center[1] + radius * np.sin(angles)],
            axis=1,
        )
        return verts.astype(np.int32)

    @staticmethod
    def _draw_polygon(vertices, color_rgb, image_size, fill=False):
        """Return an empty image with the polygon drawn."""
        import cv2

        img = np.zeros((image_size, image_size, 3), dtype=np.uint8)
        if fill:
            cv2.fillPoly(img, [vertices], color_rgb)
        else:
            cv2.polylines(img, [vertices], True, color_rgb, thickness=2)
        return img

    # --------------------------- Dataset API --------------------------- #
    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        # Random polygon
        n_sides = np.random.randint(3, 9)
        verts = self._generate_polygon(n_sides)

        # Outline image (white edges)
        outline_rgb = [255, 255, 255]
        inp_img = self._draw_polygon(verts, outline_rgb, self.image_size, fill=False)

        # Pick a random fill colour
        color_name = np.random.choice(self.color_names)
        color_rgb = self.colors_rgb[color_name]
        out_img = self._draw_polygon(verts, color_rgb, self.image_size, fill=True)

        color_idx = self.color_names.index(color_name)

        # Apply transforms
        if self.transform:
            transformed = self.transform(image=inp_img, mask=out_img)
            inp_tensor = _ensure_chw_tensor(transformed["image"])
            out_tensor = _ensure_chw_tensor(transformed["mask"])
        else:
            inp_tensor = _ensure_chw_tensor(inp_img)
            out_tensor = _ensure_chw_tensor(out_img)

        return {
            "input": inp_tensor,
            "output": out_tensor,
            "color_idx": torch.tensor(color_idx, dtype=torch.long),
            "color_name": color_name,
        }


# --------------------------------------------------------------------------- #
def create_dataloaders(
    data_dir: str | Path,
    batch_size: int = 16,
    num_workers: int = 4,
    image_size: int = 128,
):
    """
    Returns (train_loader, val_loader, n_colors).
    Falls back to synthetic data if <data_dir>/<split>/data.json is missing.
    """
    data_dir = Path(data_dir)

    json_train = data_dir / "training" / "data.json"
    if json_train.exists():
        train_ds = PolygonColorDataset(data_dir, "training", image_size=image_size)
        val_ds = PolygonColorDataset(data_dir, "validation", image_size=image_size)
    else:
        print("Real dataset not found → using synthetic data")
        train_ds = SyntheticPolygonDataset(5000, image_size=image_size)
        val_ds = SyntheticPolygonDataset(1000, image_size=image_size)

    dl_kwargs = dict(batch_size=batch_size, num_workers=num_workers, pin_memory=True)
    train_loader = torch.utils.data.DataLoader(train_ds, shuffle=True, **dl_kwargs)
    val_loader = torch.utils.data.DataLoader(val_ds, shuffle=False, **dl_kwargs)

    return train_loader, val_loader, train_ds.num_colors
