"""
Quick start script for linear probing or kNN evaluation with I-JEPA
"""

import argparse
import yaml
import torch
from src.train import Trainer


# Configuration for I-JEPA model
IJEPA_MODEL = 'ijepa-vith14-1k'  # facebook/ijepa_vith14_1k


def load_ijepa_config(dataset: str):
    """Load I-JEPA config for a given dataset."""

    # EuroSAT has 10 classes.
    num_classes = 10
    
    # Batch size for I-JEPA ViT-H/14 (large model)
    batch_size = 64
    
    # Create config
    config = {
        'dataset': {
            'name': dataset,
            'data_path': '/kaggle/input/datasets/apollo2506/eurosat-dataset/EuroSAT',
            'batch_size': batch_size,
            'num_workers': 4,
            'image_size': 224,
            'val_fraction': 0.1,
            'seed': 42
        },
        'model': {
            'use_pretrained': True,
            'pretrained_source': 'huggingface',
            'pretrained_name': IJEPA_MODEL,
            'num_classes': num_classes,
            'dropout': 0.1
        },
        'training': {
            'mode': 'linear_probe',
            'knn_k': 20,
            'epochs': 20,
            'warmup_epochs': 5,
            'base_lr': 1e-3,
            'weight_decay': 0.05,
            'save_frequency': 10,
            'eval_frequency': 5,
            'checkpoint_path': f'./checkpoints/ijepa_{dataset.lower()}',
            'log_dir': f'./logs/ijepa_{dataset.lower()}',
            'early_stopping': {
                'enabled': False,
                'patience': 5,
                'min_delta': 0.0
            }
        },
        'finetuning': {
            'freeze_encoder': True,
            'unfreeze_after_epochs': None
        }
    }
    
    return config


def main():
    parser = argparse.ArgumentParser(description='Run linear probing or kNN with I-JEPA')
    
    parser.add_argument('--dataset', type=str, default='EuroSAT',
                       choices=['EuroSAT'],
                       help='Dataset to train on')
    
    parser.add_argument('--mode', type=str, default='linear_probe',
                       choices=['linear_probe', 'knn'],
                       help='Execution mode: linear probing or kNN evaluation')
    
    parser.add_argument('--epochs', type=int, default=None,
                       help='Number of epochs (overrides default)')
    
    parser.add_argument('--batch-size', type=int, default=None,
                       help='Batch size (overrides default)')
    
    parser.add_argument('--lr', type=float, default=None,
                       help='Learning rate (overrides default)')

    parser.add_argument('--knn-k', type=int, default=None,
                       help='Number of neighbors for kNN mode')
    
    parser.add_argument('--early-stopping', action='store_true',
                       help='Enable early stopping based on validation accuracy')
    parser.add_argument('--patience', type=int, default=None,
                       help='Early stopping patience in epochs')
    parser.add_argument('--min-delta', type=float, default=None,
                       help='Minimum validation improvement to reset early stopping counter')
    
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

    if args.knn_k:
        config['training']['knn_k'] = args.knn_k

    if args.early_stopping:
        config['training'].setdefault('early_stopping', {})
        config['training']['early_stopping']['enabled'] = True
    if args.patience is not None:
        config['training'].setdefault('early_stopping', {})
        config['training']['early_stopping']['patience'] = args.patience
    if args.min_delta is not None:
        config['training'].setdefault('early_stopping', {})
        config['training']['early_stopping']['min_delta'] = args.min_delta
    
    # Set execution mode
    config['training']['mode'] = args.mode
    if args.mode == 'linear_probe':
        config['finetuning']['freeze_encoder'] = True
        config['finetuning']['unfreeze_after_epochs'] = None
        config['training']['base_lr'] = 1e-3  # Higher LR for linear probe
        config['training']['epochs'] = min(20, config['training']['epochs'])
        print("🔒 LINEAR PROBING MODE: Training only the classification head")
    
    elif args.mode == 'knn':
        config['finetuning']['freeze_encoder'] = True
        config['finetuning']['unfreeze_after_epochs'] = None
        config['training']['epochs'] = 0
        print("📌 KNN MODE: No gradient training, encoder features + kNN classifier")
    
    # Print configuration
    print("\n" + "="*70)
    print("I-JEPA Linear Probe / kNN Configuration")
    print("="*70)
    print("Model:          I-JEPA ViT-H/14 (facebook/ijepa_vith14_1k)")
    print(f"Dataset:        {config['dataset']['name']}")
    print(f"Mode:           {args.mode}")
    print(f"Batch size:     {config['dataset']['batch_size']}")
    print(f"Epochs:         {config['training']['epochs']}")
    print(f"Learning rate:  {config['training']['base_lr']}")
    print(f"Freeze encoder: {config['finetuning']['freeze_encoder']}")
    if args.mode == 'knn':
        print(f"kNN neighbors:  {config['training']['knn_k']}")
    if config['training'].get('early_stopping', {}).get('enabled', False):
        print(f"Early stopping: enabled (patience={config['training']['early_stopping'].get('patience', 5)}, min_delta={config['training']['early_stopping'].get('min_delta', 0.0)})")
    else:
        print("Early stopping: disabled")
    print(f"Device:         {'CUDA' if torch.cuda.is_available() else 'CPU'}")
    if torch.cuda.is_available():
        print(f"GPU:            {torch.cuda.get_device_name(0)}")
    print("="*70 + "\n")
    
    # Train
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    trainer = Trainer(config, device)
    final_acc = trainer.train()
    
    print("\n" + "="*70)
    print("✅ Run completed!")
    print(f"Final accuracy: {final_acc:.2f}%")
    if args.mode == 'linear_probe':
        print(f"Best accuracy: {trainer.best_acc:.2f}%")
        print(f"Checkpoints saved in: {config['training']['checkpoint_path']}")
    print(f"Logs saved in: {config['training']['log_dir']}")
    print("="*70 + "\n")


if __name__ == '__main__':
    main()
