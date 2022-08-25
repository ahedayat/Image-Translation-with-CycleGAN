import torch.nn as nn
from .nets_utils import c7s1_k, d_k, u_k, init_conv_layer_normal
from .residual_block import ResidualBlock


class Generator(nn.Module):
    """
        In this module, Generator Network is implemented
    """

    def __init__(self, in_channel, num_residual_blocks) -> None:
        super().__init__()

        self.in_channel = in_channel
        self.num_residual_blocks = num_residual_blocks

        # Encoder Network
        self.encoder = self.get_encoder(in_channel=self.in_channel)

        # Residual Blocks
        self.residuals = self.get_residuals(
            num_residuals=self.num_residual_blocks)

        # Decoder Network
        self.decoder = self.get_decoder()

        # Output Layer
        self.output_layer = self.get_output_layer(
            model_in_channel=self.in_channel)

        # Model
        self.model = self.encoder + self.residuals + self.decoder + self.output_layer

        self.model = nn.Sequential(*self.model)

    def forward(self, x):
        """
            Forward Propagation
        """
        return self.model(x)

    def get_encoder(self, in_channel):
        """
            This function returns encoder network.
        """
        encoder = list()

        encoder += c7s1_k(in_channel=in_channel, out_channel=64,
                          reflection_pad=True)  # c7s1-64

        encoder += d_k(in_channel=64, out_channel=128)  # d128
        encoder += d_k(in_channel=128, out_channel=256)  # d256

        # return nn.Sequential(*encoder)
        return encoder

    def get_residuals(self, num_residuals):
        """
            This function returns middle residual blocks
        """
        residuals = list()

        # R256, R256, ..., R256
        for _ in range(num_residuals):
            residuals += [ResidualBlock(in_channel=256)]

        # return nn.Sequential(*residuals)
        return residuals

    def get_decoder(self):
        """
            This function returns decoder blocks
        """
        decoder = list()

        # u128
        decoder += u_k(in_channel=256, out_channel=128)
        # u64
        decoder += u_k(in_channel=128, out_channel=64)

        # return nn.Sequential(*decoder)
        return decoder

    def get_output_layer(self, model_in_channel):
        """
            This funciton returns output layer of Generative Network
        """
        # output_layer = nn.Sequential(
        #     nn.ReflectionPad2d(model_in_channel),
        #     nn.Conv2d(64, model_in_channel, 7),
        #     nn.Tanh()
        # )

        output_layer = [
            nn.ReflectionPad2d(model_in_channel),
            nn.Conv2d(64, model_in_channel, 7),
            nn.Tanh()
        ]

        return output_layer

    def init_layers(self):
        """
            Initialize Layers of Generator
        """
        blocks = list()
        for block in self.model:
            block = block.apply(init_conv_layer_normal)
            blocks.append(block)

        self.model = nn.Sequential(*blocks)
