"""
Utilities to load pre-trained I-JEPA models from Hugging Face
"""

import torch
import torch.nn as nn
from transformers import AutoModel


class HuggingFaceIJEPAFinetune(nn.Module):
    """
    Fine-tuning wrapper for Hugging Face I-JEPA (Image-based Joint-Embedding Predictive Architecture)
    """
    AVAILABLE_MODELS = {
        # I-JEPA (Meta/Facebook) - Official pre-trained model
        'ijepa-vith14-1k': 'facebook/ijepa_vith14_1k',  # I-JEPA ViT-H/14
    }
    
    def __init__(self, model_name='ijepa-vith14-1k', num_classes=10, 
                 freeze_encoder=False, dropout=0.1):
        """
        Args:
            model_name: Name of the model (from AVAILABLE_MODELS)
            num_classes: Number of output classes
            freeze_encoder: Whether to freeze the encoder
            dropout: Dropout rate for classification head
        """
        super().__init__()
        
        if model_name in self.AVAILABLE_MODELS:
            model_id = self.AVAILABLE_MODELS[model_name]
        else:
            model_id = model_name  # Allow custom model IDs
        
        print(f"Loading pre-trained model: {model_id}")
        
        # Load pre-trained I-JEPA model
        try:
            # Use AutoModel for I-JEPA (works with Hugging Face)
            self.encoder = AutoModel.from_pretrained(model_id)
            hidden_size = self.encoder.config.hidden_size
        except Exception as e:
            try:
                # Fallback to AutoModel for other architectures (including I-JEPA)
                self.encoder = AutoModel.from_pretrained(model_id)
                hidden_size = self.encoder.config.hidden_size
            except Exception as e2:
                print(f"Error loading model: {e2}")
                print("Trying with trust_remote_code=True...")
                # Some models need trust_remote_code
                self.encoder = AutoModel.from_pretrained(model_id, trust_remote_code=True)
                hidden_size = self.encoder.config.hidden_size
        
        # Freeze encoder if requested
        if freeze_encoder:
            for param in self.encoder.parameters():
                param.requires_grad = False
            print("Encoder frozen (linear probing mode)")
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, num_classes)
        )
        
        print(f"Model loaded with {hidden_size}-dim embeddings for {num_classes} classes")
    
    def forward(self, x):
        """
        Args:
            x: Input images (B, C, H, W)
        Returns:
            logits: Classification logits (B, num_classes)
        """
        # Get encoder outputs
        outputs = self.encoder(pixel_values=x)
        
        # Use [CLS] token representation
        cls_output = outputs.last_hidden_state[:, 0]
        
        # Classification
        logits = self.classifier(cls_output)
        
        return logits
    
    def unfreeze_encoder(self):
        """Unfreeze encoder for full fine-tuning"""
        for param in self.encoder.parameters():
            param.requires_grad = True
        print("Encoder unfrozen")
    
    def freeze_encoder(self):
        """Freeze encoder for linear probing"""
        for param in self.encoder.parameters():
            param.requires_grad = False
        print("Encoder frozen")


if __name__ == '__main__':
    # Test I-JEPA loading
    print("Testing I-JEPA model loading...")
    
    model = HuggingFaceIJEPAFinetune(
        model_name='ijepa-vith14-1k',
        num_classes=10,
        freeze_encoder=False
    )
    
    # Test forward pass
    x = torch.randn(2, 3, 224, 224)
    out = model(x)
    print(f"✓ I-JEPA Output shape: {out.shape}")
