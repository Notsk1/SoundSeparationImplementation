"""
------------------------------------------------------------------------
File that defines the UNet audio network. ModelBuilder calls uses this
class for audio network initialization

Last modified May 2022
Author Petteri Nuotiomaa
------------------------------------------------------------------------
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class UNet(nn.Module):
    """
    Define a class for UNet module used to process the audio
    """
    def __init__(self, n_channels, n_classes, bilinear=True):
        """
        Initialize network before traning

        Arguments:
        n_channels: Number of input channels
        n_classes: Number of predicted output classes
        bilinear: Define if bilinear upsampling is used for interpolating
        """
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        # Initialize the encoder
        self.inc = InConv(n_channels, 64)
        self.down1 = Down(64, 64*2)
        self.down2 = Down(64*2, 64*4)
        self.down3 = Down(64*4, 64*8)
        self.down4 = Down(64*8, 64*8)
        self.down5 = Down(64*8, 64*8) 
        self.down6 = Down(64*8, 64*4, inner=True)

        # Initialize the decoder
        #self.upFrame = Up(21, 64*4, inner=True)
        self.up1 = Up(64*4, 64*8, inner = True, bilinear=self.bilinear)
        self.up111 = Up(64*8, 64*8, bilinear=self.bilinear)
        self.up11 = Up(64*8, 64*8, bilinear=self.bilinear)
        self.up21 = Up(64*8, 64*4, bilinear=self.bilinear)
        self.up31 = Up(64*4, 64*2, bilinear=self.bilinear)
        self.up41 = Up(64*2, 64, bilinear=self.bilinear)
        self.outc1 = OutConv(64, n_classes, bilinear=self.bilinear)
        
        
    
    def forward(self, x):
        """
        Forward pass of the network

        Arguments:
        x: Audio STFT being processed
        frameFeatures: Framefeatures used for the multimodal separation approach

        Returns:
        outputs: Reconstructed STFTs for each possible class
        """
        # Decode:
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x6 = self.down5(x5)
        x7 = self.down6(x6)

        #frameFeatures = torch.maximum(frameFeatures[0,:,:,:,:], frameFeatures[1,:,:,:,:])

        #frameFeatures = torch.reshape(frameFeatures, frameFeatures.size()[1:])

        #x = self.upFrame(frameFeatures, x7)
        x = self.up1(x7, x6)
        x = self.up111(x, x5)
        x = self.up11(x, x4)
        x = self.up21(x, x3)
        x = self.up31(x, x2)
        x = self.up41(x, x1)
        outputs = self.outc1(x)
         
        return outputs


class Down(nn.Module):
    """Convolution with stride to make featuremap dimensions smaller"""

    def __init__(self, in_channels, out_channels, inner = False):
        """
        Initialize module before training

        Arguments:
        n_channels: Number of input channels
        n_classes: Number of output channels
        inner: Define if the module is used as the last down convolution layer
        """
        super().__init__()
        downrelu = nn.LeakyReLU(0.2, False) #inplace True
        downnorm = nn.BatchNorm2d(out_channels)
        self.conv = nn.Sequential(
            downrelu,
            nn.Conv2d(in_channels, out_channels, kernel_size=4, stride = 2, padding=1, bias=False),
            downnorm
        )
        if inner:
            self.conv = nn.Sequential(
            downrelu,
            nn.Conv2d(in_channels, out_channels, kernel_size=4, stride = 2, padding=1, bias=False)
        )

    def forward(self, x):
        """
        Forward pass of the module

        Arguments:
        x: output of the previous convolutional layer

        Returns:
        Output of this convolutional layer
        """
        return self.conv(x)


class Up(nn.Module):
    """Upsample then convolution"""

    def __init__(self, in_channels, out_channels, inner = False, bilinear=True):
        """
        Initialize module before training

        Arguments:
        n_channels: Number of input channels
        n_classes: Number of output channels
        inner: Define if the module is used as the last down convolution layer
        bilinear: Define if bilinear upsampling is used for interpolating
        """
        super().__init__()
        if inner:
            in_channels = in_channels
        else:
            in_channels = 2*in_channels
        uprelu = nn.ReLU()
        upnorm = nn.BatchNorm2d(out_channels)

        if bilinear:
            upsample = nn.Upsample(
                scale_factor=2, mode='bilinear', align_corners=True)
        else:
            upsample = nn.Upsample(
                scale_factor=2, align_corners=True)

        self.conv = nn.Sequential(
            uprelu,
            upsample,
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            upnorm
        )
    def forward(self, x1, x2):
        """
        Forward pass of the module. Concatinates the output of the corresponding
        down convolutional layer fort skip connection to help with reconstruction

        Arguments:
        x1: output of the previous convolutional layer
        x2: output of the corresponding down convolutional layer

        Returns:
        Output of this convolutional layer
        """
        x = torch.cat([x2, self.conv(x1)], 1)
        return x


class OutConv(nn.Module):
    """Upsample then convolution without batchnormalization"""
    def __init__(self, in_channels, out_channels, bilinear=True):
        """
        Initialize module before training

        Arguments:
        n_channels: Number of input channels
        n_classes: Number of output channels
        bilinear: Define if bilinear upsampling is used for interpolating
        """
        super(OutConv, self).__init__()
        uprelu = nn.ReLU()
        if bilinear:
            upsample = nn.Upsample(
                scale_factor=2, mode='bilinear', align_corners=True)
        else:
            upsample = nn.Upsample(
                scale_factor=2, align_corners=True)
        self.conv = nn.Sequential(
            uprelu,
            upsample,
            nn.Conv2d(in_channels*2, out_channels, kernel_size=3, padding=1)
        )

    def forward(self, x):
        """
        Forward pass of the module

        Arguments:
        x: output of the previous convolutional layer

        Returns:
        Output of the whole system
        """
        return self.conv(x)

class InConv(nn.Module):
    """Convolution layer to the original input"""
    def __init__(self, in_channels, out_channels):
        """
        Initialize module before training

        Arguments:
        n_channels: Number of input channels
        n_classes: Number of output channels
        """
        super(InConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=4,stride=2, padding=1, bias=False)

    def forward(self, x):
        """
        Forward pass of the module

        Arguments:
        x: input to the whole audio network

        Returns:
        Output of this convolutional layer
        """
        return self.conv(x)