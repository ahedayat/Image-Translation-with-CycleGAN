import torch.nn as nn
from .nets_utils import c_k, init_conv_layer_normal


class Discriminator(nn.Module):
    """
        In this class, Discriminator network is implemented.
    """

    def __init__(self, in_size) -> None:
        super().__init__()

        in_channel, in_h, in_w = in_size

        self.model = self.get_model(in_channel)

        self.out_size = (1, in_h//2 ** 4, in_w//2 ** 4)

    def forward(self, x):
        """
            Forward Propagation
        """
        return self.model(x)

    def get_model(self, in_channel):
        """
            In this function, discriminator model is constructed.
            "
                For discriminator networks, we use 70 × 70 PatchGAN. \
                Let Ck denote a 4 × 4 Convolution-InstanceNorm-LeakyReLU \
                layer with k filters and stride 2. After the last layer, \
                we apply a convo- lution to produce a 1-dimensional output. \
                We do not use InstanceNorm for the first C64 layer. \
                We use leaky ReLUs with a slope of 0.2. \
                The discriminator architecture is: \
                C64-C128-C256-C512
            "
        """
        model = list()

        # C64 without Instance Normalization
        model += c_k(in_channel, 64, instance_norm=False)

        # C128
        model += c_k(64, 128, instance_norm=True)

        # C256
        model += c_k(128, 256, instance_norm=True)

        # C512
        model += c_k(256, 512, instance_norm=True)

        # Zero Paddinng
        model += [nn.ZeroPad2d((1, 0, 1, 0))]

        # Paper: 'After the last layer, we apply a convo- lution to produce a 1-dimensional output.'
        model += [nn.Conv2d(512, 1, 4, padding=1)]

        return nn.Sequential(*model)

    def init_layers(self):
        """
            Initialize Layers of Discriminator
        """
        self.model = self.model.apply(init_conv_layer_normal)
