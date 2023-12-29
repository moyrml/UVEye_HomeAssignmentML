import torch
from torch import nn
from torchsummary import summary
import math


class Encoder(nn.Module):
    def __init__(
            self,
            depth=2,
            input_channels=3,
            expand_factor=2,
            kernel_size=3,
            stride=2,
            padding=1,
            latent_dim=32,
            activation_func=nn.GELU,
            input_dim=500
    ):
        super().__init__()

        layers = nn.ModuleList()
        for n in range(depth):
            layers.extend([
                nn.Conv2d(
                    input_channels * expand_factor ** n, input_channels * expand_factor ** (n+1),
                    kernel_size=kernel_size,
                    stride=stride,
                    padding=padding
                ),
                activation_func()
            ])

        layers.extend([
            nn.Flatten(),
            nn.Linear(
                input_channels * expand_factor ** depth * math.ceil(input_dim / stride ** depth) ** 2,
                latent_dim
            )
        ])

        self.depth = depth
        self.input_channels = input_channels
        self.stride = stride
        self.kernel_size = kernel_size
        self.latent_dim = latent_dim
        self.expand_factor = expand_factor
        self.layers = layers
        self.activation_func = activation_func
        self.input_dim = input_dim
        self.padding = padding

    def forward(self, x):
        for i_m, m in enumerate(self.layers):
            x = m(x)
        return x

    def describe(self):
        print(f"{'Encoder Summary':-^90}")
        summary(
            self,
            input_size=(self.input_channels, self.input_dim, self.input_dim),
            batch_size=-1
        )


class Decoder(nn.Module):
    def __init__(self, encoder):
        super().__init__()

        layers = nn.ModuleList()
        layers.extend([
            nn.Linear(
                encoder.latent_dim,
                encoder.input_channels * encoder.expand_factor ** encoder.depth * math.ceil(encoder.input_dim / encoder.stride ** encoder.depth) ** 2,
            ),
            encoder.activation_func(),
            nn.Unflatten(-1, (encoder.input_channels * encoder.expand_factor ** encoder.depth, math.ceil(encoder.input_dim / encoder.stride ** encoder.depth),math.ceil(encoder.input_dim / encoder.stride ** encoder.depth)))
        ])

        for n in range(encoder.depth):
            layers.extend([
                nn.ConvTranspose2d(
                    encoder.input_channels * encoder.expand_factor ** (encoder.depth - n),
                    encoder.input_channels * encoder.expand_factor ** (encoder.depth - n - 1),
                    encoder.kernel_size,
                    stride=encoder.stride,
                    output_padding=encoder.padding,
                    padding=encoder.padding
                ),
                encoder.activation_func()
            ])

        layers[-1] = nn.Tanh()  # Scale output to [-1,1] since we don't want to work in the [0,255] range.

        self.layers = layers
        self.encoder_input_channels = encoder.input_channels
        self.encoder_expand_factor = encoder.expand_factor
        self.encoder_depth = encoder.depth
        self.encoder_kernel_size = encoder.kernel_size
        self.encoder_stride = encoder.stride
        self.encoder_activation_func = encoder.activation_func
        self.encoder_input_dim = encoder.input_dim
        self.encoder_latent_dim = encoder.latent_dim

    def forward(self, x):
        for i_m, m in enumerate(self.layers):
            x = m(x)
        return x

    def describe(self):
        print(f"{'Decoder Summary':-^90}")
        summary(
            self,
            input_size=(self.encoder_latent_dim,),
            batch_size=-1
        )


class AE(nn.Module):
    def __init__(self, encoder_params):
        super().__init__()
        self.encoder = Encoder(**encoder_params)
        self.decoder = Decoder(self.encoder)

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

    def describe(self):
        self.encoder.describe()
        self.decoder.describe()


if __name__ == '__main__':
    input_dim = 256
    latent_dim = 16
    encoder = Encoder(6, expand_factor=2, input_dim=input_dim, latent_dim=latent_dim)
    encoder.describe()

    decoder = Decoder(encoder)
    decoder.describe()

    inp = torch.randn((2, 3, input_dim, input_dim))
    latent = encoder(inp)
    rec = decoder(latent)
