"""CNN-GRU model for frame-level classification from grayscale video.

Architecture
------------
1.  Lightweight CNN encoder maps each (1, H, W) frame to a feature vector.
2.  GRU processes sequences of feature vectors, accumulating temporal context.
3.  Linear head maps each GRU output to class logits.

GRU is chosen over LSTM here because:
- Fewer parameters → faster inference (critical for real-time deployment).
- Single hidden state → simpler state management during live streaming.
- Empirically comparable to LSTM on sequences of this length (~240 steps).
"""

import torch
import torch.nn as nn


class ResBlock(nn.Module):
    """Simple ResNet block."""

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = self.relu = nn.GELU() # nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.downsample = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = self.downsample(x)
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += identity
        return self.relu(out)


class CNNEncoder(nn.Module):
    """ResNet-style conv stack: grayscale frame → flat feature vector."""

    def __init__(self, channels: list[int], dropout: float = 0.3):
        super().__init__()
        layers = []

        # Initial convolution
        in_c = channels[0] if channels else 32
        layers += [
            nn.Conv2d(1, in_c, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(in_c),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        ]

        for out_c in channels:
            layers.append(ResBlock(in_c, out_c, stride=2 if in_c != out_c else 1))
            in_c = out_c

        layers.append(nn.AdaptiveAvgPool2d((2, 2)))
        layers.append(nn.Flatten())
        layers.append(nn.Dropout(dropout))
        self.net = nn.Sequential(*layers)
        self.out_dim = (channels[-1] if channels else 32) * 4

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, 1, H, W) → (B, out_dim)"""
        return self.net(x)


class VideoClassifier(nn.Module):
    """CNN encoder + GRU + classification head."""

    def __init__(self, num_classes: int, cnn_channels: list[int],
                 gru_hidden: int, gru_layers: int, dropout: float):
        super().__init__()
        self.encoder = CNNEncoder(cnn_channels, dropout)
        self.gru = nn.GRU(
            input_size=self.encoder.out_dim,
            hidden_size=gru_hidden,
            num_layers=gru_layers,
            batch_first=True,
            dropout=dropout if gru_layers > 1 else 0.0,
        )
        self.head = nn.Linear(gru_hidden, num_classes)
        self.dropout = nn.Dropout(dropout)  # NOTE: this was added from 8 to 9. Manual dropout.
        self.gru_layers = gru_layers
        self.gru_hidden = gru_hidden

    def forward(self, x: torch.Tensor, h: torch.Tensor | None = None):
        """
        Args:
            x: (B, T, 1, H, W) — batch of frame sequences.
            h: optional GRU hidden state (num_layers, B, gru_hidden).

        Returns:
            logits: (B, T, num_classes)
            h_out:  updated hidden state
        """
        B, T, C, H, W = x.shape

        features = self.encoder(x.reshape(B * T, C, H, W))  # (B*T, feat_dim)
        features = features.reshape(B, T, -1)                # (B, T, feat_dim)

        gru_out, h_out = self.gru(features, h)           # (B, T, gru_hidden)
        gru_out = self.dropout(gru_out)  # NOTE: this was added from 8 to 9. Apply dropout only during training.
        logits = self.head(gru_out)                      # (B, T, num_classes)
        return logits, h_out

    def init_hidden(self, batch_size: int, device: torch.device):
        return torch.zeros(self.gru_layers, batch_size, self.gru_hidden,
                           device=device)
