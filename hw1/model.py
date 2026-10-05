import torch
import torch.nn as nn

class ConvBlock(nn.Module):
    def __init__(self, input_channels, output_channels, kernel_size=3, stride=1, pooling=2):
        super().__init__()
        self.conv = nn.Conv2d(input_channels, output_channels, kernel_size, stride, padding=kernel_size//2)
        self.bn = nn.BatchNorm2d(output_channels)
        self.relu = nn.ReLU()
        self.mp = nn.MaxPool2d(pooling)

    def forward(self, input):
        output = self.mp(self.relu(self.bn(self.conv(input))))
        return output


class ShortChunkEncoder(nn.Module):
    """The 7 conv blocks + global max pool: (B, in_channels, n_mels, T) -> (B, n_channels*4)."""

    def __init__(self, n_channels=64, in_channels=1):
        super().__init__()
        self.layer1 = ConvBlock(in_channels, n_channels, pooling=2)
        self.layer2 = ConvBlock(n_channels, n_channels, pooling=2)
        self.layer3 = ConvBlock(n_channels, n_channels*2, pooling=2)
        self.layer4 = ConvBlock(n_channels*2, n_channels*2, pooling=2)
        self.layer5 = ConvBlock(n_channels*2, n_channels*2, pooling=2)
        self.layer6 = ConvBlock(n_channels*2, n_channels*2, pooling=2)
        self.layer7 = ConvBlock(n_channels*2, n_channels*4, pooling=2)
        self.out_dim = n_channels*4

    def forward(self, x):
        # CNN
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = self.layer6(x)
        x = self.layer7(x) # Shape: (B, C, 1, T)
        x = x.squeeze(2)   # Shape: (B, C, T)

        if x.shape[-1] != 1:
            x = nn.MaxPool1d(x.shape[-1])(x) # Shape: (B, C, 1)
        return x.squeeze(2) # Shape: (B, C)


class _DenseHead(nn.Module):
    """Embedding -> logits. Attribute names match the original ShortChunkCNN so old checkpoints load."""

    def _build_head(self, in_dim, hidden_dim, n_class):
        self.dense1 = nn.Linear(in_dim, hidden_dim)
        self.bn = nn.BatchNorm1d(hidden_dim)
        self.dense2 = nn.Linear(hidden_dim, n_class)
        self.dropout = nn.Dropout(0.5)
        self.relu = nn.ReLU()

    def _head(self, x):
        x = self.dense1(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.dense2(x)  # Raw logits; CrossEntropyLoss applies softmax

        return x


class ShortChunkCNN(ShortChunkEncoder, _DenseHead):
    """Early fusion: stems are stacked as input channels and share every conv layer."""

    def __init__(self, n_channels=64, n_class=6, in_channels=1):
        super().__init__(n_channels=n_channels, in_channels=in_channels)
        self._build_head(self.out_dim, n_channels*4, n_class)

    def forward(self, x):
        return self._head(super().forward(x))


class LateFusionCNN(_DenseHead):
    """Late fusion: one encoder per stem (own conv weights and BatchNorm statistics), embeddings
    concatenated, then a shared dense head. Input is the same (B, S, n_mels, T) as early fusion."""

    def __init__(self, n_channels=64, n_class=6, n_stems=2):
        super().__init__()
        self.branches = nn.ModuleList(ShortChunkEncoder(n_channels=n_channels) for _ in range(n_stems))
        self._build_head(n_stems * n_channels*4, n_channels*4, n_class)

    def forward(self, x):
        emb = torch.cat([branch(x[:, i:i + 1]) for i, branch in enumerate(self.branches)], dim=1)
        return self._head(emb)


def build_model(n_class, stems=("mix",), fusion="early", n_channels=64):
    if fusion == "late" and len(stems) > 1:
        return LateFusionCNN(n_channels=n_channels, n_class=n_class, n_stems=len(stems))
    return ShortChunkCNN(n_channels=n_channels, n_class=n_class, in_channels=len(stems))
