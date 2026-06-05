"""
Training script for I-JEPA fine-tuning with pre-trained models from Hugging Face
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import os
import yaml
from tqdm import tqdm
import numpy as np
from typing import Dict, Optional

from .pretrained_models import HuggingFaceIJEPAFinetune
from .data_loader import get_dataloaders


class Trainer:
    """Trainer for I-JEPA fine-tuning"""
    
    def __init__(self, config: Dict, device: str = 'cuda'):
        self.config = config
        self.device = device
        
        # Create directories
        os.makedirs(config['training']['checkpoint_path'], exist_ok=True)
        os.makedirs(config['training']['log_dir'], exist_ok=True)
        
        # Initialize tensorboard
        self.writer = SummaryWriter(config['training']['log_dir'])
        
        # Get dataloaders
        self.train_loader, self.test_loader = get_dataloaders(config)
        
        # Build model
        self.model = self._build_model()
        self.model = self.model.to(device)
        
        # Loss function
        self.criterion = nn.CrossEntropyLoss()
        
        # Optimizer
        self.optimizer = self._build_optimizer()
        
        # Scheduler
        self.scheduler = self._build_scheduler()
        
        # Metrics
        self.best_acc = 0.0
        self.current_epoch = 0
        
    def _build_model(self) -> nn.Module:
        """Build I-JEPA model from Hugging Face"""
        model_config = self.config['model']
        finetuning_config = self.config['finetuning']
        
        num_classes = model_config['num_classes']
        freeze_encoder = finetuning_config['freeze_encoder']
        
        # Load pre-trained model
        pretrained_source = model_config.get('pretrained_source', 'huggingface')
        pretrained_name = model_config.get('pretrained_name', 'ijepa-vith14-1k')
        
        if pretrained_source != 'huggingface':
            raise ValueError(f"Only 'huggingface' source is supported for I-JEPA")
        
        model = HuggingFaceIJEPAFinetune(
            model_name=pretrained_name,
            num_classes=num_classes,
            freeze_encoder=freeze_encoder,
            dropout=model_config.get('dropout', 0.1)
        )
        
        return model
    
    def _build_optimizer(self) -> optim.Optimizer:
        """Build optimizer"""
        training_config = self.config['training']
        
        # Separate parameters for weight decay
        param_groups = [
            {'params': [], 'weight_decay': 0.0},  # No decay (biases, norms)
            {'params': [], 'weight_decay': training_config['weight_decay']}  # Decay
        ]
        
        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
            
            # No weight decay for biases and layer norms
            if 'bias' in name or 'norm' in name:
                param_groups[0]['params'].append(param)
            else:
                param_groups[1]['params'].append(param)
        
        optimizer = optim.AdamW(param_groups, lr=training_config['base_lr'])
        
        return optimizer
    
    def _build_scheduler(self):
        """Build learning rate scheduler with warmup"""
        training_config = self.config['training']
        total_steps = len(self.train_loader) * training_config['epochs']
        warmup_steps = len(self.train_loader) * training_config['warmup_epochs']
        
        def lr_lambda(current_step):
            if current_step < warmup_steps:
                return float(current_step) / float(max(1, warmup_steps))
            else:
                progress = float(current_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
                return max(0.0, 0.5 * (1.0 + np.cos(np.pi * progress)))
        
        scheduler = optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda)
        return scheduler
    
    def train_epoch(self) -> float:
        """Train for one epoch"""
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        
        # Progressive fine-tuning: unfreeze encoder after N epochs
        unfreeze_after = self.config['finetuning'].get('unfreeze_after_epochs', None)
        if unfreeze_after and self.current_epoch == unfreeze_after:
            if hasattr(self.model, 'unfreeze_encoder'):
                print(f"\n🔓 Unfreezing encoder at epoch {self.current_epoch}")
                self.model.unfreeze_encoder()
                # Rebuild optimizer to include unfrozen parameters
                self.optimizer = self._build_optimizer()
                self.scheduler = self._build_scheduler()
        
        pbar = tqdm(self.train_loader, desc=f'Epoch {self.current_epoch}')
        
        for batch_idx, (images, targets) in enumerate(pbar):
            images = images.to(self.device)
            targets = targets.to(self.device)
            
            # Forward pass
            outputs = self.model(images)
            loss = self.criterion(outputs, targets)
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping for stability
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            self.scheduler.step()
            
            # Metrics
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            
            # Update progress bar
            acc = 100. * correct / total
            pbar.set_postfix({
                'loss': total_loss / (batch_idx + 1),
                'acc': f'{acc:.2f}%',
                'lr': self.optimizer.param_groups[0]['lr']
            })
        
        avg_loss = total_loss / len(self.train_loader)
        avg_acc = 100. * correct / total
        
        return avg_loss, avg_acc
    
    @torch.no_grad()
    def evaluate(self) -> tuple:
        """Evaluate on test set"""
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        
        for images, targets in tqdm(self.test_loader, desc='Evaluating'):
            images = images.to(self.device)
            targets = targets.to(self.device)
            
            # Forward pass
            outputs = self.model(images)
            loss = self.criterion(outputs, targets)
            
            # Metrics
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
        
        avg_loss = total_loss / len(self.test_loader)
        avg_acc = 100. * correct / total
        
        return avg_loss, avg_acc
    
    def save_checkpoint(self, filename: str):
        """Save checkpoint"""
        checkpoint = {
            'epoch': self.current_epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_acc': self.best_acc,
            'config': self.config
        }
        
        filepath = os.path.join(self.config['training']['checkpoint_path'], filename)
        torch.save(checkpoint, filepath)
        print(f'Saved checkpoint: {filepath}')
    
    def load_checkpoint(self, filepath: str):
        """Load checkpoint"""
        checkpoint = torch.load(filepath, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.current_epoch = checkpoint['epoch']
        self.best_acc = checkpoint['best_acc']
        
        print(f'Loaded checkpoint from epoch {self.current_epoch}')
    
    def train(self):
        """Main training loop"""
        num_epochs = self.config['training']['epochs']
        eval_frequency = self.config['training']['eval_frequency']
        save_frequency = self.config['training']['save_frequency']
        
        print(f"Starting training for {num_epochs} epochs")
        print(f"Device: {self.device}")
        print(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        print(f"Trainable parameters: {sum(p.numel() for p in self.model.parameters() if p.requires_grad):,}")
        
        for epoch in range(self.current_epoch, num_epochs):
            self.current_epoch = epoch
            
            # Train
            train_loss, train_acc = self.train_epoch()
            
            # Log training metrics
            self.writer.add_scalar('Loss/train', train_loss, epoch)
            self.writer.add_scalar('Accuracy/train', train_acc, epoch)
            self.writer.add_scalar('LearningRate', self.optimizer.param_groups[0]['lr'], epoch)
            
            print(f'Epoch {epoch}: Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%')
            
            # Evaluate
            if (epoch + 1) % eval_frequency == 0:
                test_loss, test_acc = self.evaluate()
                
                # Log test metrics
                self.writer.add_scalar('Loss/test', test_loss, epoch)
                self.writer.add_scalar('Accuracy/test', test_acc, epoch)
                
                print(f'Epoch {epoch}: Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.2f}%')
                
                # Save best model
                if test_acc > self.best_acc:
                    self.best_acc = test_acc
                    self.save_checkpoint('best_model.pth')
                    print(f'New best accuracy: {self.best_acc:.2f}%')
            
            # Save checkpoint
            if (epoch + 1) % save_frequency == 0:
                self.save_checkpoint(f'checkpoint_epoch_{epoch}.pth')
        
        # Final evaluation
        test_loss, test_acc = self.evaluate()
        print(f'\nFinal Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.2f}%')
        print(f'Best Test Acc: {self.best_acc:.2f}%')
        
        # Save final model
        self.save_checkpoint('final_model.pth')
        
        self.writer.close()


def train_model(config_path: str):
    """Train model with configuration file"""
    # Load config
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Set device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Using device: {device}')
    
    # Create trainer
    trainer = Trainer(config, device)
    
    # Train
    trainer.train()


if __name__ == '__main__':
    train_model('config.yaml')
