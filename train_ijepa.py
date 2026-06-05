"""
Quick start script for training with I-JEPA pre-trained models
"""

import argparse
import yaml
import torch
from src.train import Trainer


# Configuration for I-JEPA model
IJEPA_MODEL = 'ijepa-vith14-1k'  # facebook/ijepa_vith14_1k


def load_ijepa_config(dataset: str):
    """Load I-JEPA config for a given dataset"""
    
    # Determine number of classes
    num_classes = {
        'MNIST': 10,
        'CIFAR10': 10,
        'CIFAR100': 100
    }.get(dataset, 10)
    
    # Batch size for I-JEPA ViT-H/14 (large model)
    batch_size = 64
    
    # Create config
    config = {
        'dataset': {
            'name': dataset,
            'data_path': './data',
            'batch_size': batch_size,
            'num_workers': 4,
            'image_size': 224
        },
        'model': {
            'use_pretrained': True,
            'pretrained_source': 'huggingface',
            'pretrained_name': IJEPA_MODEL,
            'num_classes': num_classes,
            'dropout': 0.1
        },
        'training': {
            'epochs': 50,
            'warmup_epochs': 5,
            'base_lr': 2e-4,
            'weight_decay': 0.05,
            'save_frequency': 10,
            'eval_frequency': 5,
            'checkpoint_path': f'./checkpoints/ijepa_{dataset.lower()}',
            'log_dir': f'./logs/ijepa_{dataset.lower()}'
        },
        'finetuning': {
            'freeze_encoder': False,
            'unfreeze_after_epochs': None
        }
    }
    
    return config


def main():
    parser = argparse.ArgumentParser(description='Fine-tune I-JEPA on various datasets')
    
    parser.add_argument('--dataset', type=str, default='MNIST',
                       choices=['MNIST', 'CIFAR10', 'CIFAR100'],
                       help='Dataset to train on')
    
    parser.add_argument('--mode', type=str, default='full',
                       choices=['full', 'linear', 'progressive'],
                       help='Training mode: full fine-tuning, linear probe, or progressive')
    
    parser.add_argument('--epochs', type=int, default=None,
                       help='Number of epochs (overrides default)')
    
    parser.add_argument('--batch-size', type=int, default=None,
                       help='Batch size (overrides default)')
    
    parser.add_argument('--lr', type=float, default=None,
                       help='Learning rate (overrides default)')
    
    parser.add_argument('--config', type=str, default=None,
                       help='Path to custom config file (overrides preset)')
    
    args = parser.parse_args()
    
    # Load configuration
    if args.config:
        # Use custom config file
        with open(args.config, 'r') as f:
            config = yaml.safe_load(f)
    else:
        # Use I-JEPA config
        config = load_ijepa_config(args.dataset)
    
    # Apply command-line overrides
    if args.epochs:
        config['training']['epochs'] = args.epochs
    
    if args.batch_size:
        config['dataset']['batch_size'] = args.batch_size
    
    if args.lr:
        config['training']['base_lr'] = args.lr
    
    # Set training mode
    if args.mode == 'linear':
        config['finetuning']['freeze_encoder'] = True
        config['finetuning']['unfreeze_after_epochs'] = None
        config['training']['base_lr'] = 1e-3  # Higher LR for linear probe
        config['training']['epochs'] = min(20, config['training']['epochs'])
        print("🔒 LINEAR PROBING MODE: Only training classifier head")
    
    elif args.mode == 'progressive':
        config['finetuning']['freeze_encoder'] = True
        config['finetuning']['unfreeze_after_epochs'] = 10
        print("🔄 PROGRESSIVE MODE: Starting frozen, will unfreeze after 10 epochs")
    
    else:  # full
        config['finetuning']['freeze_encoder'] = False
        config['finetuning']['unfreeze_after_epochs'] = None
        print("🚀 FULL FINE-TUNING MODE: Training entire model")
    
    # Print configuration
    print("\n" + "="*70)
    print("I-JEPA Fine-tuning Configuration")
    print("="*70)
    print(f"Model:          I-JEPA ViT-H/14 (facebook/ijepa_vith14_1k)")
    print(f"Dataset:        {config['dataset']['name']}")
    print(f"Mode:           {args.mode.upper()}")
    print(f"Batch size:     {config['dataset']['batch_size']}")
    print(f"Epochs:         {config['training']['epochs']}")
    print(f"Learning rate:  {config['training']['base_lr']}")
    print(f"Freeze encoder: {config['finetuning']['freeze_encoder']}")
    if config['finetuning']['unfreeze_after_epochs']:
        print(f"Unfreeze after: {config['finetuning']['unfreeze_after_epochs']} epochs")
    print(f"Device:         {'CUDA' if torch.cuda.is_available() else 'CPU'}")
    if torch.cuda.is_available():
        print(f"GPU:            {torch.cuda.get_device_name(0)}")
    print("="*70 + "\n")
    
    # Train
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    trainer = Trainer(config, device)
    trainer.train()
    
    print("\n" + "="*70)
    print("✅ Training completed!")
    print(f"Best accuracy: {trainer.best_acc:.2f}%")
    print(f"Checkpoints saved in: {config['training']['checkpoint_path']}")
    print(f"Logs saved in: {config['training']['log_dir']}")
    print("="*70 + "\n")


if __name__ == '__main__':
    main()
