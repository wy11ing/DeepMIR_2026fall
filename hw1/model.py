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


class ShortChunkCNN(nn.Module):
    def __init__(
        self,
        n_channels=64, 
        sample_rate=16000,
        n_fft=512,
        f_min=0.0,
        f_max=8000.0,
        n_mels=128,
        n_class=6
    ):
        super().__init__()

        # CNN
        self.layer1 = ConvBlock(1, n_channels, pooling=2)
        self.layer2 = ConvBlock(n_channels, n_channels, pooling=2)
        self.layer3 = ConvBlock(n_channels, n_channels*2, pooling=2)
        self.layer4 = ConvBlock(n_channels*2, n_channels*2, pooling=2)
        self.layer5 = ConvBlock(n_channels*2, n_channels*2, pooling=2)
        self.layer6 = ConvBlock(n_channels*2, n_channels*2, pooling=2)
        self.layer7 = ConvBlock(n_channels*2, n_channels*4, pooling=2)

        # Dense
        self.dense1 = nn.Linear(n_channels*4, n_channels*4)
        self.bn = nn.BatchNorm1d(n_channels*4)
        self.dense2 = nn.Linear(n_channels*4, n_class)
        self.dropout = nn.Dropout(0.5)
        self.relu = nn.ReLU()


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
        x = x.squeeze(2) # Shape: (B, C)

        # Dense
        x = self.dense1(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.dense2(x)  # Raw logits; CrossEntropyLoss applies softmax

        return x