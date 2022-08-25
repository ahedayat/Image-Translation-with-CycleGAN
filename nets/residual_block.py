import torch.nn as nn


class ResidualBlock(nn.Module):
    """
        In this module, 'Residual Block' that is mentioned in the paper, was implemented.
        - Paper:
            "
            Rk denotes a residual block that contains two 3 × 3 con-
            volutional layers with the same number of filters on both
            layer.
            "
    """

    def __init__(self, in_channel) -> None:
        super().__init__()

        self.block_list = [
            # Paper: Reflection padding was used to reduce artifacts.
            nn.ReflectionPad2d(1),
            # two 3 × 3 convolutional layers with the same number of filters on both layer.
            nn.Conv2d(in_channel, in_channel, 3),
            nn.InstanceNorm2d(in_channel),
            nn.ReLU(inplace=True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_channel, in_channel, 3),
            nn.InstanceNorm2d(in_channel)
        ]

        self.block = nn.Sequential(*self.block_list)

    def forward(self, x):
        """
            Forward Propagation
        """
        return x + self.block(x)
