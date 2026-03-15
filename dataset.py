"""
SSL Dataset with Plant Disease Augmentation
For each image, returns TWO independently augmented views.
This is the "positive pair" that SimCLR trains on.

"""

import os

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

# ---------------------------------------------------------------------------
# 1.  Augmentation Pipeline  (plant disease specific)
# ---------------------------------------------------------------------------

def get_ssl_augmentation(image_size: int = 224) -> transforms.Compose:
    """
    Two random augmentations applied independently to the same image.
    Each call produces a different view.

    """
    return transforms.Compose([
        # Crop: tighter scale keeps lesions/spots in frame
        transforms.RandomResizedCrop(image_size, scale=(0.4, 1.0)),

        # Spatial flips — leaves have no canonical orientation
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),

        # Moderate rotation — field photos vary widely in leaf angle
        transforms.RandomRotation(degrees=30),

        # Subtle colour jitter — preserve diagnostic disease colours
        # (browning, yellowing, dark spots are classification signals)
        transforms.ColorJitter(
            brightness=0.2,   # was 0.4 — gentler
            contrast=0.2,     # was 0.4 — gentler
            saturation=0.2,   # was 0.4 — gentler
            hue=0.05,         # was 0.1 — very small: hue shift alters disease look
        ),

        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],   # ImageNet stats (good default for leaf photos)
            std=[0.229, 0.224, 0.225]
        ),
    ])


# ---------------------------------------------------------------------------
# 2.  Contrastive Dataset
# ---------------------------------------------------------------------------

class PlantDiseaseSSLDataset(Dataset):
    """
    Loads unlabelled plant disease images from a directory tree.

    For each image, __getitem__ returns:
        (view_A, view_B)  — two differently augmented versions
    No labels are returned — SSL is fully unsupervised.
    """

    VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

    def __init__(self, data_dir: str, image_size: int = 224):
        """
        Args:
            data_dir   : path to the root data folder (e.g. "data/")
            image_size : images resized to (image_size × image_size)
                         224 recommended for ResNet-18 backbone
        """
        self.augment   = get_ssl_augmentation(image_size)
        self.filepaths = self._collect_images(data_dir)

        if len(self.filepaths) == 0:
            raise ValueError(f"No images found in '{data_dir}'. "
                             "Check the path and file extensions.")

        print(f"[Dataset] Found {len(self.filepaths):,} images in '{data_dir}'")

    # ------------------------------------------------------------------
    def _collect_images(self, root: str) -> list:
        paths = []
        for dirpath, _, filenames in os.walk(root):
            for fname in filenames:
                if os.path.splitext(fname)[1].lower() in self.VALID_EXTENSIONS:
                    paths.append(os.path.join(dirpath, fname))
        return sorted(paths)

    # ------------------------------------------------------------------
    def __len__(self) -> int:
        return len(self.filepaths)

    def __getitem__(self, idx: int):
        img    = Image.open(self.filepaths[idx]).convert("RGB")
        view_a = self.augment(img)   # first  random augmentation
        view_b = self.augment(img)   # second random augmentation (different!)
        return view_a, view_b


# ---------------------------------------------------------------------------
# 3.  DataLoader factory
# ---------------------------------------------------------------------------

def get_ssl_dataloader(
    data_dir:    str,
    batch_size:  int  = 64,
    image_size:  int  = 224,
    num_workers: int  = 2,
    shuffle:     bool = True,
) -> DataLoader:
    dataset = PlantDiseaseSSLDataset(data_dir=data_dir, image_size=image_size)
    loader  = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,    # NT-Xent needs consistent batch sizes
    )
    return loader


# ---------------------------------------------------------------------------
# Quick sanity check
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    data_dir = sys.argv[1] if len(sys.argv) > 1 else "data"

    loader = get_ssl_dataloader(data_dir, batch_size=4, image_size=224)
    view_a, view_b = next(iter(loader))
    print(f"view_A batch shape : {view_a.shape}")   # (4, 3, 224, 224)
    print(f"view_B batch shape : {view_b.shape}")   # (4, 3, 224, 224)
