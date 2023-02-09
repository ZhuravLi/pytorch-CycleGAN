import torch.nn as nn
import torch.nn.functional as F


class ResBlock(nn.Module):
    """A class for residual block"""
    def __init__(self, f):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=f, out_channels=f, kernel_size=3, stride=1, padding=1),
            nn.InstanceNorm2d(f),
            nn.ReLU(),
            nn.Conv2d(in_channels=f, out_channels=f, kernel_size=3, stride=1, padding=1)
        )
        self.norm = nn.InstanceNorm2d(f)
    
    def forward(self, x):
        return F.relu(self.norm(self.conv(x)+x))
    

class Generator(nn.Module):
    """
    A class for generator. Architecture: incoder -> transformations -> decoder.
    Incoder: 3 convolutions. Transformations: 6 residual blocks. 
    Decoder: 2 transposed convolutions + convolution.
    """
    def __init__(self, f=64, res_blocks=6):
        super().__init__()
        # Adding 3 convolutions
        layers = [
            # in: 3 x 256 x 256
            nn.ReflectionPad2d(3),
            nn.Conv2d(in_channels=3, out_channels=f, kernel_size=7, stride=1, padding=0),
            nn.InstanceNorm2d(f),
            nn.ReLU(),
            # out: f x 256 x 256
            nn.Conv2d(in_channels=f, out_channels=2*f, kernel_size=3, stride=2, padding=1),
            nn.InstanceNorm2d(2*f),
            nn.ReLU(),
            # out: 2f x 128 x 128 
            nn.Conv2d(in_channels=2*f, out_channels=4*f, kernel_size=3, stride=2, padding=1),
            nn.InstanceNorm2d(4*f),
            nn.ReLU(),
            # out: 4f x 64 x 64 
        ]
        
        # Adding residual blocks
        for i in range(res_blocks):
            layers.append(ResBlock(4*f))
        
        # Adding two transposed convolutions and final convolution
        layers.extend([
            # in: 4f x 64 x 64 
            nn.ConvTranspose2d(in_channels=4*f, out_channels=2*f, \
                               kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.InstanceNorm2d(2*f),
            nn.ReLU(),
            # out: 2f x 128 x 128 
            nn.ConvTranspose2d(in_channels=2*f, out_channels=f, \
                               kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.InstanceNorm2d(f),
            nn.ReLU(),
            # out: f x 256 x 256
            nn.ReflectionPad2d(3),
            nn.Conv2d(in_channels=f, out_channels=3, kernel_size=7, stride=1, padding=0),
            nn.Tanh()
            # out: 3 x 256 x 256
        ])
        self.layers = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.layers(x)
    

class Discriminator(nn.Module):
    def __init__(self, f=64):
        super().__init__()
        
        layers = [
            nn.Conv2d(3, f, kernel_size=4, stride=2, padding=1, bias=False), 
            nn.LeakyReLU(0.2), # out: f x 128 x 128
        ]
        layers.extend([
            nn.Conv2d(f, 2*f, kernel_size=4, stride=2, padding=1, bias=False), 
            nn.InstanceNorm2d(2*f),
            nn.LeakyReLU(0.2), #out: 2f x 64 x 64
        ])
        layers.extend([
            nn.Conv2d(2*f, 4*f, kernel_size=4, stride=2, padding=1, bias=False), 
            nn.InstanceNorm2d(4*f),
            nn.LeakyReLU(0.2), #out: 4f x 32 x 32
        ])
        layers.extend([
            nn.Conv2d(4*f, 8*f, kernel_size=4, stride=1, padding=1, bias=False), 
            nn.InstanceNorm2d(8*f),
            nn.LeakyReLU(0.2), #out: 8f x 31 x 31
        ])
        layers.extend([
            nn.Conv2d(8*f, 1, kernel_size=4, stride=1, padding=1) # out: 1 x 30 x 30
        ])
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        x = self.layers(x)
        # Average pooling and flatten
        return F.avg_pool2d(x, x.size()[2:]).view(x.size()[0], -1)
    