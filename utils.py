"""
------------------------------------------------------------------------
This file contains helper function that help the execution of the main
system.

Last modified May 2022
Author Petteri Nuotiomaa
------------------------------------------------------------------------
"""
import os
import shutil

import numpy as np
import torch
import torch.nn.functional as F


def preProcessAudio(ampMix, inputAmp, device, lossFunc, prob, log):
    # Transform amplitudes to desired size
    N = ampMix.size(1)
    B = ampMix.size(0)
    T = ampMix.size(3)

    grid_warp = torch.from_numpy(
                warpgrid(B, 256, T, warp=True)).to(device)
    input = F.grid_sample(inputAmp, grid_warp)
    ampMixOut = torch.zeros((B,N,256,256))
    for n in range(N):
        ampMixOut[:,n:n+1,:,:] = F.grid_sample(ampMix[:,n:n+1,:,:], grid_warp)
    

    groundTruthMax = torch.argmax(ampMixOut, axis = -3)
    if lossFunc == 'cross':
        groundTruth = groundTruthMax
        if prob:
            groundTruth = ampMixOut.softmax(dim = -3)
            

    elif lossFunc == 'BCE':
        groundTruth = torch.zeros(ampMixOut.size())
        index = 0
        for i in range(groundTruth.size(1)):
            trueArray = index == groundTruthMax
            groundTruth[:,index,:,:] = trueArray.int()
            index += 1
    
    return input.to(device), groundTruth.to(device)

def postProcessAudio(outputAmps, groundTruth, device, lossFunc, prob):
    N = outputAmps.size(1)
    B = outputAmps.size(0)
    T = outputAmps.size(3)
    grid_warp = torch.from_numpy(
                warpgrid(B, 512, T, warp=False)).to(device)
    output = F.grid_sample(outputAmps, grid_warp)

    output = torch.zeros((B,N,512,256))
    for n in range(N):
        output[:,n:n+1,:,:] = F.grid_sample(outputAmps[:,n:n+1,:,:], grid_warp)

    if lossFunc == 'BCE':
        groundTruths = torch.zeros((B,N,512,256))
        for n in range(N):
            groundTruths[:,n:n+1,:,:] = F.grid_sample(groundTruth[:,n:n+1,:,:], grid_warp, mode='nearest', padding_mode = 'border')
    elif lossFunc == 'cross':
        if prob:
            grid_warp = torch.from_numpy(
                        warpgrid(B, 512, T, warp=False)).to(device)
            groundTruths = F.grid_sample(groundTruth.float(), grid_warp, mode='nearest', padding_mode = 'border')
        else:
            grid_warp = torch.from_numpy(
                        warpgrid(B, 512, T, warp=False)).to(device)
            groundTruths = F.grid_sample(groundTruth[:,None,:,:].float(), grid_warp, mode='nearest', padding_mode = 'border')

    
    return output.to(device), groundTruths.to(device)

def warpgrid(bs, HO, WO, warp=True):
    # meshgrid
    x = np.linspace(-1, 1, WO)
    y = np.linspace(-1, 1, HO)
    xv, yv = np.meshgrid(x, y)
    grid = np.zeros((bs, HO, WO, 2))
    grid_x = xv
    if warp:
        grid_y = (np.power(21, (yv+1)/2) - 11) / 10
    else:
        grid_y = np.log(yv * 10 + 11) / np.log(21) * 2 - 1
    grid[:, :, :, 0] = grid_x
    grid[:, :, :, 1] = grid_y
    grid = grid.astype(np.float32)
    return grid


def makedirs(path, remove=False):
    if os.path.isdir(path):
        if remove:
            shutil.rmtree(path)
            print('removed existing directory...')
        else:
            return
    os.makedirs(path)
