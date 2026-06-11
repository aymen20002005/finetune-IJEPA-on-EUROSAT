"""
Utilities to load pre-trained I-JEPA models from Hugging Face
"""

import torch
import torch.nn as nn
from transformers import AutoModel


class AttentionLinearProbeHead(nn.Module):
    """Attention-enhanced probe head operating on frozen patch tokens."""

    def __init__(
        self,
        hidden_size: int,
        num_classes: int,
        dropout: float = 0.1,
        attention_heads: int = 8,
        attention_dropout: float = 0.1,
        attention_type: str = 'self',
    ):
        super().__init__()

        if attention_type not in {'self', 'cross'}:
            raise ValueError("attention_type must be 'self' or 'cross'")

        self.attention_type = attention_type

        self.norm = nn.LayerNorm(hidden_size)
        self.attn = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=attention_heads,
            dropout=attention_dropout,
            batch_first=True,
        )
        self.fusion = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.classifier = nn.Linear(hidden_size, num_classes)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        tokens = self.norm(tokens)

        if self.attention_type == 'cross':
            cls_query = tokens[:, :1]
            patch_context = tokens[:, 1:] if tokens.size(1) > 1 else tokens
            attended_cls, _ = self.attn(cls_query, patch_context, patch_context, need_weights=False)
            cls_token = attended_cls.squeeze(1)
            patch_summary = patch_context.mean(dim=1)
        else:
            attn_tokens, _ = self.attn(tokens, tokens, tokens, need_weights=False)
            cls_token = attn_tokens[:, 0]
            if attn_tokens.size(1) > 1:
                patch_summary = attn_tokens[:, 1:].mean(dim=1)
            else:
                patch_summary = cls_token

        fused = self.fusion(torch.cat([cls_token, patch_summary], dim=-1))
        return self.classifier(fused)


class HuggingFaceIJEPAFinetune(nn.Module):
    """
    Fine-tuning wrapper for Hugging Face I-JEPA (Image-based Joint-Embedding Predictive Architecture)
    """
    AVAILABLE_MODELS = {
        # I-JEPA (Meta/Facebook) - Official pre-trained model
        'ijepa-vith14-1k': 'facebook/ijepa_vith14_1k',  # I-JEPA ViT-H/14
    }
    
    def __init__(
        self,
        model_name='ijepa-vith14-1k',
        num_classes=10,
        freeze_encoder=False,
        dropout=0.1,
        attention_heads=8,
        attention_dropout=0.1,
        attention_type='self',
    ):
        """
        Args:
            model_name: Name of the model (from AVAILABLE_MODELS)
            num_classes: Number of output classes
            freeze_encoder: Whether to freeze the encoder
            dropout: Dropout rate for classification head
            attention_heads: Number of heads for attention probe
            attention_dropout: Dropout used inside the attention probe
            attention_type: Attention mode for probe ('self' or 'cross')
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

        if hidden_size % attention_heads != 0:
            raise ValueError(
                f"attention_heads ({attention_heads}) must divide hidden_size ({hidden_size})"
            )
        
        # Classification head
        self.classifier = AttentionLinearProbeHead(
            hidden_size=hidden_size,
            num_classes=num_classes,
            dropout=dropout,
            attention_heads=attention_heads,
            attention_dropout=attention_dropout,
            attention_type=attention_type,
        )
        print(
            f"Using {attention_type} attention probe head (heads={attention_heads}, attn_dropout={attention_dropout})"
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

        logits = self.classifier(outputs.last_hidden_state)
        
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
