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
import librosa as lb


def preProcessAudio(ampMix, inputAmp, device, lossFunc, prob, log):
    # Transform amplitudes to desired size
    N = ampMix.size(1)
    B = ampMix.size(0)
    T = ampMix.size(3)

    grid_warp = torch.from_numpy(
        warpgrid(B, 256, T, warp=True)).to(device)
    input = F.grid_sample(inputAmp, grid_warp)
    ampMixOut = torch.zeros((B, N, 256, 256))
    for n in range(N):
        ampMixOut[:, n:n+1, :,
                  :] = F.grid_sample(ampMix[:, n:n+1, :, :], grid_warp)

    groundTruthMax = torch.argmax(ampMixOut, axis=-3)
    if lossFunc == 'cross':
        groundTruth = groundTruthMax
        if prob:
            groundTruth = ampMixOut.softmax(dim=-3)

    elif lossFunc == 'BCE':
        groundTruth = torch.zeros(ampMixOut.size())
        index = 0
        for i in range(groundTruth.size(1)):
            trueArray = index == groundTruthMax
            groundTruth[:, index, :, :] = trueArray.int()
            index += 1

    return input.to(device), groundTruth.to(device)


def postProcessAudio(outputAmps, groundTruth, device, lossFunc, prob):
    N = outputAmps.size(1)
    B = outputAmps.size(0)
    T = outputAmps.size(3)
    grid_warp = torch.from_numpy(
        warpgrid(B, 512, T, warp=False)).to(device)
    output = F.grid_sample(outputAmps, grid_warp)

    output = torch.zeros((B, N, 512, 256))
    for n in range(N):
        output[:, n:n+1, :,
               :] = F.grid_sample(outputAmps[:, n:n+1, :, :], grid_warp)

    if lossFunc == 'BCE':
        groundTruths = torch.zeros((B, N, 512, 256))
        for n in range(N):
            groundTruths[:, n:n+1, :, :] = F.grid_sample(
                groundTruth[:, n:n+1, :, :], grid_warp, mode='nearest', padding_mode='border')
    elif lossFunc == 'cross':
        if prob:
            grid_warp = torch.from_numpy(
                warpgrid(B, 512, T, warp=False)).to(device)
            groundTruths = F.grid_sample(
                groundTruth.float(), grid_warp, mode='nearest', padding_mode='border')
        else:
            grid_warp = torch.from_numpy(
                warpgrid(B, 512, T, warp=False)).to(device)
            groundTruths = F.grid_sample(groundTruth[:, None, :, :].float(
            ), grid_warp, mode='nearest', padding_mode='border')

    return output.to(device), groundTruths.to(device)


def form_audio(inputAmps, inputPhase, groundTruth, outputs, labels, args, ampMix):
    """Forms the input, output and ground truth audios"""
    # Detach tensors, move them to the cpu and change them to numpy arrays
    inputAmps = inputAmps[0].detach().cpu().numpy()
    groundTruth = groundTruth[0].detach().cpu().numpy()
    ampMix = ampMix.detach().cpu().numpy()

    output1 = outputs[0][labels[0][0]].detach().cpu().numpy()
    output2 = outputs[0][labels[0][1]].detach().cpu().numpy()
    phase = inputPhase[0].detach().cpu().numpy()

    # Get the binary mask of the ground truth
    if args['realAudio']:
        groundTruth1 = ampMix[0][labels[0][0]]
        groundTruth2 = ampMix[0][labels[0][1]]
    else:
        if args['lossFunc'] == 'BCE':
            groundTruth1 = groundTruth[labels[0][0].item()]
            groundTruth2 = groundTruth[labels[0][1].item()]

        elif args['lossFunc'] == 'cross':
            if args['crossProb']:
                groundTruth1 = (
                    groundTruth[labels[0][0].item()] > 0.1).astype(int)
                groundTruth2 = (
                    groundTruth[labels[0][1].item()] > 0.1).astype(int)
            else:
                groundTruth1 = (groundTruth == labels[0][0].item()).astype(int)
                groundTruth2 = (groundTruth == labels[0][1].item()).astype(int)
        groundTruth1 = inputAmps*groundTruth1
        groundTruth2 = inputAmps*groundTruth2

    semanticMask = 0
    # Form output from argmax information
    if args['semantic']:
        # Get correct labels
        classNumber1 = labels[0][0].item()
        classNumber2 = labels[0][1].item()

        thresholdArray = np.ones(outputs[0].size()[1:])*args['threshold']
        semanticMask = np.argmax(np.concatenate(
            (outputs[0].detach().cpu().numpy(), thresholdArray[np.newaxis, :, :]), 0), 0)

        unique, counts = np.unique(semanticMask, return_counts=True)
        result = np.column_stack((unique, counts))
        print(result)

        '''
        output1 = output1*(semanticMask == classNumber1).astype(int)
        output2 = output2*(semanticMask == classNumber2).astype(int)
        '''
        # Multiply with binary mask to get the output audio magnitudes
        output1 = inputAmps*(semanticMask == classNumber1).astype(int)
        output2 = inputAmps*(semanticMask == classNumber2).astype(int)

    else:
        # Form output by thresholding the network output
        output1 = inputAmps*(output1 > args['threshold'])
        output2 = inputAmps*(output2 > args['threshold'])

    # Multiply the signals with the phase information of the mixed audio
    inputAudio = inputAmps.astype(complex) * np.exp(1j*phase)

    groundTruthAudio1 = groundTruth1.astype(complex) * np.exp(1j*phase)
    groundTruthAudio2 = groundTruth2.astype(complex) * np.exp(1j*phase)

    outputAudio1 = output1.astype(complex) * np.exp(1j*phase)
    outputAudio2 = output2.astype(complex) * np.exp(1j*phase)

    # Get the time domain signal with ISTFT
    inputAudio = lb.istft(
        inputAudio[0], hop_length=args['stftHop'], window='hann')

    groundTruthAudio1 = lb.istft(
        groundTruthAudio1[0], hop_length=args['stftHop'], window='hann')  # if not real [0]
    groundTruthAudio2 = lb.istft(
        groundTruthAudio2[0], hop_length=args['stftHop'], window='hann')

    outputAudio1 = lb.istft(
        outputAudio1[0], hop_length=args['stftHop'], window='hann')
    outputAudio2 = lb.istft(
        outputAudio2[0], hop_length=args['stftHop'], window='hann')

    groundTruthAudios = [groundTruthAudio1, groundTruthAudio2]
    outputAudios = [outputAudio1, outputAudio2]

    return inputAudio, np.array(groundTruthAudios), np.array(outputAudios), semanticMask


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
