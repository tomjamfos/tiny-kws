import torch
from torch import nn

from tiny_kws.labels import CLASS_NAMES
from tiny_kws.models.normalisation import PerBinMeanNormalisation

CONV_KERNEL_SIZE = 3
CONV_OUTPUT_CHANNELS = (16, 32, 64, 64)
POOL_KERNEL_SIZE = 2
DROPOUT_RATE = 0.2


class ConvBlock(nn.Sequential):
    """
    Applies convolution, batch normalisation, ReLU and max pooling.

    Divides the height and width by POOL_KERNEL_SIZE, rounding down.
    """

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__(
            nn.Conv2d(
                in_channels,
                out_channels,
                CONV_KERNEL_SIZE,
                padding=CONV_KERNEL_SIZE // 2,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.MaxPool2d(POOL_KERNEL_SIZE),
        )


class Baseline(nn.Module):
    """
    Takes raw log-mel features shaped (N, 32, 98), where the last axis is frames.
    Returns logits shaped (N, len(CLASS_NAMES)) in CLASS_NAMES order.

    Normalises the input with PerBinMeanNormalisation,
    then applies one ConvBlock per entry of CONV_OUTPUT_CHANNELS.

    Global average pooling reduces each resulting feature map to its mean.
    After dropout, a linear layer maps those means to the logits.
    """

    def __init__(self) -> None:
        super().__init__()

        self.normalisation = PerBinMeanNormalisation()

        input_channels = (1, *CONV_OUTPUT_CHANNELS[:-1])
        self.features = nn.Sequential(
            *(
                ConvBlock(in_chans, out_chans)
                for in_chans, out_chans in zip(input_channels, CONV_OUTPUT_CHANNELS)
            )
        )
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(DROPOUT_RATE),
            nn.Linear(CONV_OUTPUT_CHANNELS[-1], len(CLASS_NAMES)),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.normalisation(x)
        x = self.features(x.unsqueeze(1))
        return self.classifier(x)
