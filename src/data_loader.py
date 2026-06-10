"""
Data loading utilities for EuroSAT only
"""

import torch
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import transforms
from PIL import Image
import csv
import os
from typing import Tuple


class DatasetFactory:
    """Factory class for creating datasets"""
    
    @staticmethod
    def get_dataset(name: str, data_path: str, split: str = 'train', 
                   img_size: int = 224, augment: bool = True):
        """
        Get dataset by name.
        
        Args:
            name: Dataset name ('EuroSAT')
            data_path: Path to data directory
            split: One of 'train', 'val', 'test'
            img_size: Target image size
            augment: Whether to apply data augmentation
        """
        if name != 'EuroSAT':
            raise ValueError(f"Only EuroSAT is supported in this codebase.")

        return DatasetFactory._get_eurosat(data_path, split, img_size, augment)

    @staticmethod
    def _get_eurosat(data_path: str, split: str, img_size: int, augment: bool):
        """Load EuroSAT dataset from CSV split files."""
        if split not in {'train', 'val', 'test'}:
            raise ValueError("split must be one of 'train', 'val', or 'test'.")

        if split == 'train' and augment:
            transform = transforms.Compose([
                transforms.Resize((img_size, img_size)),
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
                transforms.RandomRotation(15),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
            ])
        else:
            transform = transforms.Compose([
                transforms.Resize((img_size, img_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
            ])

        split_csv = f"{split}.csv"
        return EuroSATCSVDataset(data_path, split_csv, transform)


class EuroSATCSVDataset(Dataset):
    """EuroSAT dataset backed by split CSV files."""

    def __init__(self, root_dir: str, split_csv: str, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.samples = []

        csv_path = os.path.join(root_dir, split_csv)
        with open(csv_path, 'r', newline='') as csv_file:
            reader = csv.DictReader(csv_file)
            for row in reader:
                filename = row.get('Filename')
                label = row.get('Label')
                if filename is None or label is None:
                    continue

                image_path = os.path.join(root_dir, filename)
                self.samples.append((image_path, int(label)))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        image_path, label = self.samples[idx]
        image = Image.open(image_path).convert('RGB')

        if self.transform is not None:
            image = self.transform(image)

        return image, label


def get_dataloaders(config: dict) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create train, validation and test dataloaders
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    dataset_name = config['dataset']['name']
    data_path = config['dataset']['data_path']
    batch_size = config['dataset']['batch_size']
    num_workers = config['dataset']['num_workers']
    img_size = config['dataset']['image_size']
    val_fraction = config['dataset'].get('val_fraction', 0.1)
    seed = config['dataset'].get('seed', 42)

    if dataset_name != 'EuroSAT':
        raise ValueError("Only EuroSAT is supported in this codebase.")

    train_dataset = DatasetFactory.get_dataset(
        dataset_name, data_path, split='train', img_size=img_size, augment=True
    )

    val_csv_path = os.path.join(data_path, 'val.csv')
    if os.path.exists(val_csv_path):
        val_dataset = DatasetFactory.get_dataset(
            dataset_name, data_path, split='val', img_size=img_size, augment=False
        )
    else:
        # Fallback to an internal random split from train.csv when val.csv is absent
        val_dataset_full = DatasetFactory.get_dataset(
            dataset_name, data_path, split='train', img_size=img_size, augment=False
        )

        total_samples = len(train_dataset)
        val_size = int(total_samples * val_fraction)
        train_size = total_samples - val_size
        if val_size <= 0 or train_size <= 0:
            raise ValueError("Validation fraction must be between 0 and 1 and produce non-empty splits.")

        generator = torch.Generator().manual_seed(seed)
        indices = torch.randperm(total_samples, generator=generator).tolist()
        val_indices = indices[:val_size]
        train_indices = indices[val_size:]

        train_dataset = Subset(train_dataset, train_indices)
        val_dataset = Subset(val_dataset_full, val_indices)

    test_dataset = DatasetFactory.get_dataset(
        dataset_name, data_path, split='test', img_size=img_size, augment=False
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    config['model']['num_classes'] = 10
    
    return train_loader, val_loader, test_loader


class IJEPADataAugmentation:
    """
    Data augmentation for I-JEPA self-supervised pre-training
    Generates masked views for context and target encoders
    """
    def __init__(self, img_size=224, patch_size=16, mask_scale=(0.15, 0.2),
                 aspect_ratio=(0.75, 1.5), num_masks=4):
        self.img_size = img_size
        self.patch_size = patch_size
        self.mask_scale = mask_scale
        self.aspect_ratio = aspect_ratio
        self.num_masks = num_masks
        self.num_patches = (img_size // patch_size) ** 2
        self.grid_size = img_size // patch_size
        
    def generate_masks(self):
        """Generate random masks for context and target"""
        masks = []
        for _ in range(self.num_masks):
            # Random scale and aspect ratio
            scale = np.random.uniform(*self.mask_scale)
            ratio = np.random.uniform(*self.aspect_ratio)
            
            # Calculate mask dimensions
            mask_area = int(self.num_patches * scale)
            mask_h = int(np.sqrt(mask_area / ratio))
            mask_w = int(mask_h * ratio)
            
            # Clip to grid size
            mask_h = min(mask_h, self.grid_size)
            mask_w = min(mask_w, self.grid_size)
            
            # Random position
            top = np.random.randint(0, self.grid_size - mask_h + 1)
            left = np.random.randint(0, self.grid_size - mask_w + 1)
            
            masks.append((top, left, mask_h, mask_w))
        
        return masks
