"""
Data loading utilities for EuroSAT and torchvision benchmark datasets
"""

import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
import numpy as np
from typing import Tuple
from PIL import Image
import csv
import os


class GrayscaleToRGB:
    """Convert single-channel grayscale images to 3-channel RGB by repeating channels"""
    def __call__(self, x):
        """Convert grayscale tensor to RGB"""
        if x.shape[0] == 1:
            return x.repeat(3, 1, 1)
        return x


class DatasetFactory:
    """Factory class for creating datasets"""
    
    @staticmethod
    def get_dataset(name: str, data_path: str, train: bool = True, 
                   img_size: int = 224, augment: bool = True):
        """
        Get dataset by name
        
        Args:
            name: Dataset name ('MNIST', 'CIFAR10', 'CIFAR100')
            data_path: Path to data directory
            train: Whether to load training or test set
            img_size: Target image size
            augment: Whether to apply data augmentation
        """
        
        if name == 'MNIST':
            return DatasetFactory._get_mnist(data_path, train, img_size, augment)
        elif name == 'CIFAR10':
            return DatasetFactory._get_cifar10(data_path, train, img_size, augment)
        elif name == 'CIFAR100':
            return DatasetFactory._get_cifar100(data_path, train, img_size, augment)
        elif name == 'EuroSAT':
            return DatasetFactory._get_eurosat(data_path, train, img_size, augment)
        else:
            raise ValueError(f"Unknown dataset: {name}")

    @staticmethod
    def _get_eurosat(data_path: str, train: bool, img_size: int, augment: bool):
        """Load EuroSAT dataset from CSV splits."""
        if train and augment:
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

        split_csv = 'train.csv' if train else 'test.csv'
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
    
    @staticmethod
    def _get_mnist(data_path: str, train: bool, img_size: int, augment: bool):
        """Load MNIST dataset"""
        
        if train and augment:
            transform = transforms.Compose([
                transforms.Resize(img_size),
                transforms.RandomRotation(10),
                transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
                transforms.ToTensor(),
                # Convert grayscale to RGB by repeating channels
                GrayscaleToRGB(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                   std=[0.229, 0.224, 0.225])
            ])
        else:
            transform = transforms.Compose([
                transforms.Resize(img_size),
                transforms.ToTensor(),
                GrayscaleToRGB(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                   std=[0.229, 0.224, 0.225])
            ])
        
        dataset = datasets.MNIST(
            root=data_path,
            train=train,
            download=True,
            transform=transform
        )
        
        return dataset
    
    @staticmethod
    def _get_cifar10(data_path: str, train: bool, img_size: int, augment: bool):
        """Load CIFAR-10 dataset"""
        
        if train and augment:
            transform = transforms.Compose([
                transforms.Resize(img_size),
                transforms.RandomCrop(img_size, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                   std=[0.229, 0.224, 0.225])
            ])
        else:
            transform = transforms.Compose([
                transforms.Resize(img_size),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                   std=[0.229, 0.224, 0.225])
            ])
        
        dataset = datasets.CIFAR10(
            root=data_path,
            train=train,
            download=True,
            transform=transform
        )
        
        return dataset
    
    @staticmethod
    def _get_cifar100(data_path: str, train: bool, img_size: int, augment: bool):
        """Load CIFAR-100 dataset"""
        
        if train and augment:
            transform = transforms.Compose([
                transforms.Resize(img_size),
                transforms.RandomCrop(img_size, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
                transforms.RandomRotation(15),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                   std=[0.229, 0.224, 0.225])
            ])
        else:
            transform = transforms.Compose([
                transforms.Resize(img_size),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                   std=[0.229, 0.224, 0.225])
            ])
        
        dataset = datasets.CIFAR100(
            root=data_path,
            train=train,
            download=True,
            transform=transform
        )
        
        return dataset


def get_dataloaders(config: dict) -> Tuple[DataLoader, DataLoader]:
    """
    Create train and test dataloaders
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Tuple of (train_loader, test_loader)
    """
    dataset_name = config['dataset']['name']
    data_path = config['dataset']['data_path']
    batch_size = config['dataset']['batch_size']
    num_workers = config['dataset']['num_workers']
    img_size = config['dataset']['image_size']
    
    # Determine number of classes
    num_classes_map = {
        'MNIST': 10,
        'CIFAR10': 10,
        'CIFAR100': 100,
        'EuroSAT': 10
    }
    num_classes = num_classes_map.get(dataset_name, 10)
    
    # Update config with num_classes
    config['model']['num_classes'] = num_classes
    
    # Create datasets
    train_dataset = DatasetFactory.get_dataset(
        dataset_name, data_path, train=True, img_size=img_size, augment=True
    )
    
    test_dataset = DatasetFactory.get_dataset(
        dataset_name, data_path, train=False, img_size=img_size, augment=False
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    print(f"Dataset: {dataset_name}")
    print(f"Train samples: {len(train_dataset)}")
    print(f"Test samples: {len(test_dataset)}")
    print(f"Number of classes: {num_classes}")
    
    return train_loader, test_loader


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
