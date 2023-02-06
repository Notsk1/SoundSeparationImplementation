"""
------------------------------------------------------------------------
File for model builder that initializes the networks

Last modified May 2022
Author Petteri Nuotiomaa
------------------------------------------------------------------------
"""
import torchvision
import torch.nn as nn
from .audio_net import UNet

class ModelBuilder():
    """
    Create networks for traning
    """
    def buildFrame(self, preTrained = True):
        """
        Load frame network from torchvision models

        Arguments:
        preTrained: Define if the model is downloaded as a pretrained model

        Returns:
        resnet: Resnet model based frame network
        """
        resnet = torchvision.models.resnet18(preTrained)
        
        num_ftrs = resnet.fc.in_features
 
        resnet.fc = nn.Linear(num_ftrs, 21)
        
        return resnet
    
    def buildAudio(self, n_channels, n_classes, bilinear = True):
        """
        Build audio network

        Arguments:
        n_channels: Number of input channels
        n_classes: Number of predicted output classes
        bilinear: Define if bilinear upsampling is used for interpolating

        Returns:
        unet: Unet model based audio network
        """
        unet = UNet(n_channels, n_classes, bilinear)
        return unet
