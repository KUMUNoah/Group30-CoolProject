import os
import re
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import transforms


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Image folders relative to data_root
IMAGE_DIRS = ["imgs_part_1", "imgs_part_2", "imgs_part_3"]

# Bowen's Disease (BOD) is SCC in situ — merged into SCC per the paper
DIAGNOSTIC_MERGE = {"BOD": "SCC"}

# All six final diagnostic classes (after merging BOD → SCC)
CLASSES = ["BCC", "SCC", "ACK", "SEK", "MEL", "NEV"]

# Cancer vs non-cancer grouping (useful for binary tasks)
CANCER_CLASSES   = {"BCC", "MEL", "SCC"}
DISEASE_CLASSES  = {"ACK", "NEV", "SEK"}

# Categorical columns and their expected unique values (for encoding / docs)
CATEGORICAL_COLS = [
    "smoke", "drink", "background_father", "background_mother",
    "pesticide", "gender", "skin_cancer_history", "cancer_history",
    "has_piped_water", "has_sewage_system", "fitspatrick", "region",
    "itch", "grew", "hurt", "changed", "bleed", "elevation", "biopsed",
]

NUMERICAL_COLS = ["age", "diameter_1", "diameter_2"]


# ---------------------------------------------------------------------------
# Helper: locate images across part folders
# ---------------------------------------------------------------------------

def _build_img_index(data_root: str) -> Dict[str, str]:
    """Scan all part folders and return a dict mapping img_id → full path."""
    index: Dict[str, str] = {}
    root = Path(data_root)
    for folder in IMAGE_DIRS:
        folder_path = root / folder
        if not folder_path.exists():
            continue
        for fpath in folder_path.glob("**/*.png"):
            index[fpath.stem] = str(fpath)   # key = filename without extension
    return index


# ---------------------------------------------------------------------------
# Dataset class
# ---------------------------------------------------------------------------

class PADDataset(Dataset):
    """
    PyTorch Dataset for PAD-UFES-20.

    Parameters
    ----------
    data_root : str
        Root directory containing imgs_part_1/, imgs_part_2/, imgs_part_3/,
        and metadata.csv.
    split_df : pd.DataFrame
        A pre-filtered slice of the metadata DataFrame for this split.
    img_index : dict
        Mapping from img_id stem → full image path (built by _build_img_index).
    transform : callable, optional
        Torchvision transforms applied to the PIL image.
    use_metadata : bool
        Whether to return encoded metadata features alongside the image.
        Useful for multi-modal models.
    label_encoder : LabelEncoder, optional
        Fitted sklearn LabelEncoder for the 'diagnostic' column.
        If None a new one is fitted on split_df (not recommended for val/test).
    meta_means : pd.Series, optional
        Column means used to impute missing numerical values.
        Pass training set means to val/test to avoid leakage.
    meta_modes : pd.Series, optional
        Column modes used to impute missing categorical values.
        Pass training set modes to val/test to avoid leakage.
    """

    def __init__(
        self,
        data_root: str,
        split_df: pd.DataFrame,
        img_index: Dict[str, str],
        transform: Optional[Callable] = None,
        use_metadata: bool = False,
        label_encoder: Optional[LabelEncoder] = None,
        meta_means: Optional[pd.Series] = None,
        meta_modes: Optional[pd.Series] = None,
    ):
        self.data_root    = data_root
        self.img_index    = img_index
        self.transform    = transform
        self.use_metadata = use_metadata

        # -- Work on a copy so we don't mutate the caller's DataFrame --
        df = split_df.reset_index(drop=True).copy()

        # Merge BOD → SCC
        df["diagnostic"] = df["diagnostic"].replace(DIAGNOSTIC_MERGE)

        # Drop rows whose image is not found on disk
        found_mask = df["img_id"].apply(
            lambda x: _strip_ext(str(x)) in img_index
        )
        missing = (~found_mask).sum()
        if missing:
            print(f"[PADDataset] Warning: {missing} image(s) not found on disk — skipping.")
        df = df[found_mask].reset_index(drop=True)

        self.df = df

        # -- Label encoding --
        if label_encoder is None:
            self.le = LabelEncoder()
            self.le.fit(CLASSES)          # fit on all classes for consistency
        else:
            self.le = label_encoder

        self.labels: List[int] = self.le.transform(df["diagnostic"].values).tolist()

        # -- Metadata encoding (optional) --
        if use_metadata:
            self._meta_means = meta_means
            self._meta_modes = meta_modes
            self.meta_tensor = self._encode_metadata(df)

    # ------------------------------------------------------------------ #
    #  Metadata helpers
    # ------------------------------------------------------------------ #

    def _encode_metadata(self, df: pd.DataFrame) -> torch.Tensor:
        """Return a (N, F) float tensor of encoded metadata features."""
        feature_frames = []

        # Numerical columns (impute with mean, then z-score normalise)
        for col in NUMERICAL_COLS:
            series = pd.to_numeric(df[col], errors="coerce")
            fill   = self._meta_means[col] if (self._meta_means is not None and col in self._meta_means) else series.mean()
            series = series.fillna(fill)
            # Normalise using training stats when available
            if self._meta_means is not None and col in self._meta_means:
                mean = self._meta_means[col]
                std  = self._meta_means.get(col + "_std", series.std())
            else:
                mean, std = series.mean(), series.std()
            std = std if std > 0 else 1.0
            feature_frames.append(((series - mean) / std).values.reshape(-1, 1))

        # Categorical columns (label encode → int, impute with mode)
        for col in CATEGORICAL_COLS:
            series = df[col].astype(str)
            fill   = (self._meta_modes[col]
                      if (self._meta_modes is not None and col in self._meta_modes)
                      else series.mode().iloc[0])
            series = series.replace("nan", fill).fillna(fill)
            # Simple ordinal encode
            cats   = sorted(series.unique())
            cat2id = {c: i for i, c in enumerate(cats)}
            encoded = series.map(cat2id).fillna(0).astype(float).values.reshape(-1, 1)
            feature_frames.append(encoded)

        meta_array = np.concatenate(feature_frames, axis=1).astype(np.float32)
        return torch.tensor(meta_array)

    # ------------------------------------------------------------------ #
    #  Core Dataset interface
    # ------------------------------------------------------------------ #

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Dict:
        row     = self.df.iloc[idx]
        img_key = _strip_ext(str(row["img_id"]))
        img_path = self.img_index[img_key]

        # Load image
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)

        sample = {
            "image":      image,
            "label":      torch.tensor(self.labels[idx], dtype=torch.long),
            "patient_id": str(row["patient_id"]),
            "lesion_id":  str(row["lesion_id"]),
            "img_id":     str(row["img_id"]),
            "diagnostic": str(row["diagnostic"]),
        }

        if self.use_metadata:
            sample["metadata"] = self.meta_tensor[idx]

        return sample

    # ------------------------------------------------------------------ #
    #  Utility
    # ------------------------------------------------------------------ #

    @property
    def class_names(self) -> List[str]:
        return list(self.le.classes_)

    @property
    def num_classes(self) -> int:
        return len(self.le.classes_)

    def class_weights(self) -> torch.Tensor:
        """Inverse-frequency weights per class (useful for loss weighting)."""
        counts = np.bincount(self.labels, minlength=self.num_classes).astype(float)
        counts = np.where(counts == 0, 1, counts)          # avoid /0
        weights = 1.0 / counts
        weights /= weights.sum()
        return torch.tensor(weights, dtype=torch.float32)

    def sample_weights(self) -> List[float]:
        """Per-sample weight for WeightedRandomSampler."""
        cw = self.class_weights().numpy()
        return [float(cw[l]) for l in self.labels]

    def label_distribution(self) -> pd.Series:
        return pd.Series(
            [self.class_names[l] for l in self.labels]
        ).value_counts().sort_index()


# ---------------------------------------------------------------------------
# Default transforms
# ---------------------------------------------------------------------------

def get_transforms(
    img_size: int = 224,
    augment: bool = True,
) -> Tuple[transforms.Compose, transforms.Compose]:
    """
    Returns (train_transform, eval_transform).

    Images are variable-size smartphone photos — we resize to a square.
    Augmentation is only applied to the training transform.
    """
    # ImageNet stats — reasonable starting point for transfer learning
    mean = [0.485, 0.456, 0.406]
    std  = [0.229, 0.224, 0.225]

    eval_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])

    if not augment:
        return eval_transform, eval_transform

    train_transform = transforms.Compose([
        transforms.Resize((int(img_size * 1.15), int(img_size * 1.15))),
        transforms.RandomCrop(img_size),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05),
        transforms.RandomRotation(15),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])

    return train_transform, eval_transform


# ---------------------------------------------------------------------------
# High-level factory function
# ---------------------------------------------------------------------------

def get_dataloaders(
    data_root: str,
    batch_size: int = 32,
    img_size: int = 224,
    val_size: float = 0.15,
    test_size: float = 0.15,
    use_metadata: bool = True,
    use_weighted_sampler: bool = True,
    num_workers: int = 4,
    seed: int = 42,
    augment: bool = True,
    metadata_filename: str = "metadata.csv",
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Build train / val / test DataLoaders for PAD-UFES-20.

    The split is done at **patient level** to prevent data leakage
    (different images of the same patient never span train/val/test).

    Parameters
    ----------
    data_root : str
        Root folder containing imgs_part_1/, imgs_part_2/, imgs_part_3/,
        and metadata.csv.
    batch_size : int
        Mini-batch size.
    img_size : int
        Square resolution to resize images to (default 224 for ImageNet models).
    val_size : float
        Fraction of patients in the validation split.
    test_size : float
        Fraction of patients in the test split.
    use_metadata : bool
        Return metadata tensors alongside images.
    use_weighted_sampler : bool
        Balance class distribution in training via WeightedRandomSampler.
    num_workers : int
        DataLoader workers.
    seed : int
        Random seed for reproducibility.
    augment : bool
        Apply data augmentation to the training set.
    metadata_filename : str
        Name of the CSV file inside data_root.

    Returns
    -------
    (train_loader, val_loader, test_loader)
    """
    data_root = str(data_root)

    # -- Load metadata --
    csv_path = os.path.join(data_root, metadata_filename)
    df = pd.read_csv(csv_path)
    print(f"[get_dataloaders] Loaded metadata: {len(df)} rows, {df.shape[1]} columns.")

    # Merge BOD → SCC
    df["diagnostic"] = df["diagnostic"].replace(DIAGNOSTIC_MERGE)

    # -- Build image index across all part folders --
    img_index = _build_img_index(data_root)
    print(f"[get_dataloaders] Found {len(img_index)} PNG images on disk.")

    # -- Patient-level split to prevent leakage --
    patients  = df["patient_id"].unique()
    train_pids, temp_pids = train_test_split(
        patients,
        test_size=val_size + test_size,
        random_state=seed,
    )
    relative_val = val_size / (val_size + test_size)
    val_pids, test_pids = train_test_split(
        temp_pids,
        test_size=1.0 - relative_val,
        random_state=seed,
    )

    train_df = df[df["patient_id"].isin(train_pids)].reset_index(drop=True)
    val_df   = df[df["patient_id"].isin(val_pids)].reset_index(drop=True)
    test_df  = df[df["patient_id"].isin(test_pids)].reset_index(drop=True)

    print(
        f"[get_dataloaders] Patient split → "
        f"train: {len(train_pids)} pts / {len(train_df)} imgs  |  "
        f"val: {len(val_pids)} pts / {len(val_df)} imgs  |  "
        f"test: {len(test_pids)} pts / {len(test_df)} imgs"
    )

    # -- Compute training set imputation stats (avoid leakage) --
    meta_means, meta_modes = None, None
    if use_metadata:
        num_data    = train_df[NUMERICAL_COLS].apply(pd.to_numeric, errors="coerce")
        meta_means  = num_data.mean()
        # also store per-column std for normalisation
        for col in NUMERICAL_COLS:
            meta_means[col + "_std"] = num_data[col].std()
        meta_modes = train_df[CATEGORICAL_COLS].astype(str).mode().iloc[0]

    # -- Shared label encoder fitted on all classes --
    le = LabelEncoder()
    le.fit(CLASSES)

    # -- Transforms --
    train_transform, eval_transform = get_transforms(img_size=img_size, augment=augment)

    # -- Datasets --
    train_dataset = PADDataset(
        data_root, train_df, img_index,
        transform=train_transform,
        use_metadata=use_metadata,
        label_encoder=le,
        meta_means=meta_means,
        meta_modes=meta_modes,
    )
    val_dataset = PADDataset(
        data_root, val_df, img_index,
        transform=eval_transform,
        use_metadata=use_metadata,
        label_encoder=le,
        meta_means=meta_means,
        meta_modes=meta_modes,
    )
    test_dataset = PADDataset(
        data_root, test_df, img_index,
        transform=eval_transform,
        use_metadata=use_metadata,
        label_encoder=le,
        meta_means=meta_means,
        meta_modes=meta_modes,
    )

    # -- Samplers --
    train_sampler = None
    if use_weighted_sampler:
        sw = train_dataset.sample_weights()
        train_sampler = WeightedRandomSampler(
            weights=sw,
            num_samples=len(sw),
            replacement=True,
        )

    # -- DataLoaders --
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=train_sampler,
        shuffle=(train_sampler is None),
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    # -- Print class distribution --
    print("\n[get_dataloaders] Training class distribution:")
    print(train_dataset.label_distribution().to_string())
    print(f"\nClass → index mapping: { {c: i for i, c in enumerate(le.classes_)} }")
    print(f"Metadata features returned: {use_metadata}")
    if use_metadata:
        print(f"  Numerical ({len(NUMERICAL_COLS)}): {NUMERICAL_COLS}")
        print(f"  Categorical ({len(CATEGORICAL_COLS)}): {CATEGORICAL_COLS}")

    return train_loader, val_loader, test_loader
    

# ---------------------------------------------------------------------------
# Utility: strip file extension from img_id (handles "PAT_1_1_1.png" or
# "PAT_1_1_1" in the CSV)
# ---------------------------------------------------------------------------

def _strip_ext(img_id: str) -> str:
    return Path(img_id).stem