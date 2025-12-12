import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class PatchEmbedding(nn.Module):
    """
    Splits an image into patches and embeds them.
    """
    def __init__(self, img_size: int = 224, patch_size: int = 16,
                 in_channels: int = 3, embed_dim: int = 768):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.n_patches = (img_size // patch_size) ** 2

        # Convolutional layer to extract patches and embed them
        self.projection = nn.Conv2d(
            in_channels, embed_dim,
            kernel_size=patch_size, stride=patch_size
        )

    def forward(self, x):
        # x shape: (batch_size, channels, height, width)
        x = self.projection(x)
        x = x.flatten(2)  # (batch_size, embed_dim, n_patches)
        x = x.transpose(1, 2)  # (batch_size, n_patches, embed_dim)
        return x


class MultiHeadSelfAttention(nn.Module):
    """
    Multi-Head Self-Attention mechanism.
    """
    def __init__(self, embed_dim: int = 768, num_heads: int = 12,
                 dropout: float = 0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        assert embed_dim % num_heads == 0

        self.qkv = nn.Linear(embed_dim, embed_dim * 3)
        self.projection = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        batch_size, n_tokens, embed_dim = x.shape

        # Generate Q, K, V
        qkv = self.qkv(x)  # (batch_size, n_tokens, embed_dim * 3)
        qkv = qkv.reshape(batch_size, n_tokens, 3, self.num_heads,
                          self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        # Compute attention scores
        scores = (torch.matmul(q, k.transpose(-2, -1)) /
                  math.sqrt(self.head_dim))
        attention = F.softmax(scores, dim=-1)
        attention = self.dropout(attention)

        # Apply attention to values
        out = torch.matmul(attention, v)
        out = out.transpose(1, 2)
        out = out.reshape(batch_size, n_tokens, embed_dim)

        # Final projection
        out = self.projection(out)
        out = self.dropout(out)
        return out


class MLPBlock(nn.Module):
    """
    Multi-Layer Perceptron block with GELU activation.
    """
    def __init__(self, embed_dim: int = 768, mlp_ratio: float = 4.0,
                 dropout: float = 0.1):
        super().__init__()
        hidden_dim = int(embed_dim * mlp_ratio)

        self.fc1 = nn.Linear(embed_dim, hidden_dim)
        self.activation = nn.GELU()
        self.fc2 = nn.Linear(hidden_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.fc1(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x


class TransformerBlock(nn.Module):
    """
    Transformer encoder block with self-attention and MLP.
    """
    def __init__(self, embed_dim: int = 768, num_heads: int = 12,
                 mlp_ratio: float = 4.0, dropout: float = 0.1):
        super().__init__()

        self.norm1 = nn.LayerNorm(embed_dim)
        self.attention = MultiHeadSelfAttention(embed_dim, num_heads, dropout)

        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = MLPBlock(embed_dim, mlp_ratio, dropout)

    def forward(self, x):
        # Self-attention with residual connection
        x = x + self.attention(self.norm1(x))

        # MLP with residual connection
        x = x + self.mlp(self.norm2(x))

        return x


class VisionTransformer(nn.Module):
    """
    Vision Transformer with three-headed classifier for car classification.
    Predicts: brand, model, and year.
    """
    def __init__(
        self,
        img_size: int = 224,
        patch_size: int = 16,
        in_channels: int = 3,
        num_classes_brand: int = 50,
        num_classes_model: int = 200,
        num_classes_year: int = 30,
        embed_dim: int = 768,
        depth: int = 12,
        num_heads: int = 12,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,
        embed_dropout: float = 0.1
    ):
        """
        Args:
            img_size: Input image size (assumes square images)
            patch_size: Size of image patches
            in_channels: Number of input channels (3 for RGB)
            num_classes_brand: Number of car brands to classify
            num_classes_model: Number of car models to classify
            num_classes_year: Number of years to classify
            embed_dim: Embedding dimension
            depth: Number of transformer blocks
            num_heads: Number of attention heads
            mlp_ratio: Ratio of MLP hidden dimension to embedding dimension
            dropout: Dropout rate
            embed_dropout: Dropout rate for embeddings
        """
        super().__init__()

        # Patch embedding
        self.patch_embed = PatchEmbedding(img_size, patch_size, in_channels,
                                          embed_dim)
        n_patches = self.patch_embed.n_patches

        # Class token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))

        # Positional embeddings
        self.pos_embed = nn.Parameter(torch.zeros(1, n_patches + 1, embed_dim))
        self.pos_dropout = nn.Dropout(embed_dropout)

        # Transformer encoder blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, mlp_ratio, dropout)
            for _ in range(depth)
        ])

        # Layer normalization
        self.norm = nn.LayerNorm(embed_dim)

        # Three-headed classifier
        self.head_brand = nn.Linear(embed_dim, num_classes_brand)
        self.head_model = nn.Linear(embed_dim, num_classes_model)
        self.head_year = nn.Linear(embed_dim, num_classes_year)

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        # Initialize positional embeddings
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)

        # Initialize linear layers
        self.apply(self._init_layer_weights)

    def _init_layer_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.trunc_normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
        elif isinstance(module, nn.LayerNorm):
            nn.init.constant_(module.bias, 0)
            nn.init.constant_(module.weight, 1.0)
        elif isinstance(module, nn.Conv2d):
            nn.init.kaiming_normal_(module.weight, mode='fan_out')
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)

    def forward(self, x):
        """
        Args:
            x: Input images of shape (batch_size, in_channels,
               img_size, img_size)
        Returns:
            Tuple of (brand_logits, model_logits, year_logits)
        """
        batch_size = x.shape[0]

        # Patch embedding
        x = self.patch_embed(x)  # (batch_size, n_patches, embed_dim)

        # Add class token
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        # (batch_size, n_patches + 1, embed_dim)
        x = torch.cat([cls_tokens, x], dim=1)

        # Add positional embeddings
        x = x + self.pos_embed
        x = self.pos_dropout(x)

        # Apply transformer blocks
        for block in self.blocks:
            x = block(x)

        # Apply layer normalization
        x = self.norm(x)

        # Extract class token representation
        cls_token_final = x[:, 0]  # (batch_size, embed_dim)

        # Three-headed classification
        brand_logits = self.head_brand(cls_token_final)
        model_logits = self.head_model(cls_token_final)
        year_logits = self.head_year(cls_token_final)

        return brand_logits, model_logits, year_logits

    def get_features(self, x):
        """
        Extract features without classification (useful for visualization
        or transfer learning).
        Args:
            x: Input images of shape (batch_size, in_channels,
               img_size, img_size)

        Returns:
            Feature vector of shape (batch_size, embed_dim)
        """
        batch_size = x.shape[0]

        # Patch embedding
        x = self.patch_embed(x)

        # Add class token
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)

        # Add positional embeddings
        x = x + self.pos_embed
        x = self.pos_dropout(x)

        # Apply transformer blocks
        for block in self.blocks:
            x = block(x)

        # Apply layer normalization
        x = self.norm(x)

        # Extract class token representation
        cls_token_final = x[:, 0]

        return cls_token_final


def create_vit_small(num_classes_brand: int, num_classes_model: int,
                     num_classes_year: int, img_size: int = 224, **kwargs):
    """
    Create a small Vision Transformer model.
    """
    # Extract specific parameters to avoid duplicates
    params = {
        'img_size': img_size,
        'patch_size': 16,
        'embed_dim': kwargs.pop('embed_dim', 384),
        'depth': kwargs.pop('depth', 12),
        'num_heads': kwargs.pop('num_heads', 6),
        'num_classes_brand': num_classes_brand,
        'num_classes_model': num_classes_model,
        'num_classes_year': num_classes_year,
    }
    params.update(kwargs)
    return VisionTransformer(**params)


def create_vit_base(num_classes_brand: int, num_classes_model: int,
                    num_classes_year: int, img_size: int = 224, **kwargs):
    """
    Create a base Vision Transformer model.
    """
    # Extract specific parameters to avoid duplicates
    params = {
        'img_size': img_size,
        'patch_size': 16,
        'embed_dim': kwargs.pop('embed_dim', 768),
        'depth': kwargs.pop('depth', 12),
        'num_heads': kwargs.pop('num_heads', 12),
        'num_classes_brand': num_classes_brand,
        'num_classes_model': num_classes_model,
        'num_classes_year': num_classes_year,
    }
    params.update(kwargs)
    return VisionTransformer(**params)


def create_vit_large(num_classes_brand: int, num_classes_model: int,
                     num_classes_year: int, img_size: int = 224, **kwargs):
    """
    Create a large Vision Transformer model.
    """
    # Extract specific parameters to avoid duplicates
    params = {
        'img_size': img_size,
        'patch_size': 16,
        'embed_dim': kwargs.pop('embed_dim', 1024),
        'depth': kwargs.pop('depth', 24),
        'num_heads': kwargs.pop('num_heads', 16),
        'num_classes_brand': num_classes_brand,
        'num_classes_model': num_classes_model,
        'num_classes_year': num_classes_year,
    }
    params.update(kwargs)
    return VisionTransformer(**params)
