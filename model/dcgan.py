import torch
import torch.nn as nn


class Generator(nn.Module):
    r"""
    Generator layers for gan:
    1. Conv Transpose Layer
    2. BatchNorm
    3. Activation(Tanh for last layer else LeakyRELU)
    LATENT_DIM x 1 x 1  →  IMG_CHANNELS x IMG_SIZE x IMG_SIZE
    """
    def __init__(self, latent_dim, im_size, im_channels,
                 conv_channels, kernels, strides, paddings,
                 output_paddings):
        super().__init__()
        self.latent_dim = latent_dim
        self.im_size = im_size
        self.im_channels = im_channels
        
        activation = nn.ReLU()
        layers_dim = [self.latent_dim] + conv_channels + [self.im_channels]
        self.layers = nn.ModuleList([
            nn.Sequential(
                nn.ConvTranspose2d(layers_dim[i], layers_dim[i + 1],
                                   kernel_size=kernels[i],
                                   stride=strides[i],
                                   padding=paddings[i],
                                   output_padding=output_paddings[i],
                                   bias=False),
                nn.BatchNorm2d(layers_dim[i + 1]) if i != len(layers_dim) - 2 else nn.Identity(),
                activation if i != len(layers_dim) - 2 else nn.Tanh()
            )
            for i in range(len(layers_dim) - 1)
        ])
    
    def forward(self, z):
        batch_size = z.shape[0]
        out = z.reshape(-1, self.latent_dim, 1, 1)
        for layer in self.layers:
            out = layer(out)
        out = out.reshape(batch_size, self.im_channels, self.im_size, self.im_size)
        return out


class Discriminator(nn.Module):
    r"""
    DCGAN Discriminator
    IMG_CHANNELS x IMG_SIZE x IMG_SIZE → 1
    """
    
    def __init__(self, im_size, im_channels,
                 conv_channels, kernels, strides, paddings):
        super().__init__()
        self.img_size = im_size
        self.im_channels = im_channels
        activation = nn.LeakyReLU()
        layers_dim = [self.im_channels] + conv_channels + [1]
        self.layers = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(layers_dim[i], layers_dim[i + 1],
                          kernel_size=kernels[i],
                          stride=strides[i],
                          padding=paddings[i],
                          bias=False if i != 0 else True),
                nn.BatchNorm2d(layers_dim[i + 1]) if i != len(layers_dim) - 2 and i != 0 else nn.Identity(),
                activation if i != len(layers_dim) - 2 else nn.Identity()
            )
            for i in range(len(layers_dim) - 1)
        ])
    
    def forward(self, x):
        out = x
        for layer in self.layers:
            out = layer(out)
        return out.reshape(x.size(0))


# TEST
if __name__ == "__main__":
    latent_dim = 100
    img_size = 64
    img_channels = 3

    gen = Generator(
        latent_dim=latent_dim,
        im_size=img_size,
        im_channels=img_channels,
        conv_channels=[512, 256, 128, 64],
        kernels=[4, 4, 4, 4, 4],
        strides=[1, 2, 2, 2, 2],
        paddings=[0, 1, 1, 1, 1],
        output_paddings=[0, 0, 0, 0, 0]
    )

    disc = Discriminator(
        im_size=img_size,
        im_channels=img_channels,
        conv_channels=[64, 128, 256, 512],
        kernels=[4, 4, 4, 4, 4],
        strides=[2, 2, 2, 2, 1],
        paddings=[1, 1, 1, 1, 0]
    )

    z = torch.randn(2, latent_dim)
    fake_img = gen(z)
    prob = disc(fake_img)

    print("Generated image shape:", fake_img.shape)
    print("Discriminator output shape:", prob.shape)
