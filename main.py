"""
------------------------------------------------------------------------
Main file for the Sound Source Semantic Segmentation project
Run this file on your python environment with necessary libraries installed

Last modified February 2023
Author Petteri Nuotiomaa
------------------------------------------------------------------------
"""
from random import randint
import time
import os
import torch
import arguments
import scipy.io.wavfile as wavfile
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import librosa as lb
import matplotlib.pyplot as plt
import cv2

from skimage import io
from skimage import color
from skimage import segmentation
from matplotlib import colors
from models import ModelBuilder
from dataset.imgDataset import ImgDataset
from dataset.duetDataset import DuetDataset
from dataset.audioDataset import AudioDataset
from dataset.combinedDataset import CombinedDataset
from mir_eval.separation import bss_eval_sources
from utils import  warpgrid, \
     makedirs, preProcessAudio, postProcessAudio

from viz import  HTMLVisualizer

def saveAndPlotMetrics(path, trainLosses, evalLosses, evalSDRs, evalSIRs, evalSARs):
    """Plots Losses, SDR, SAR and SIR to specified path"""
    x = np.linspace(0, len(evalLosses), len(evalLosses))
    # Losses:
    plt.figure()
    if len(trainLosses) > 0:
        plt.plot(x,trainLosses, label='Train loss')
    plt.plot(x, evalLosses, label='Eval Loss')
    plt.legend()
    plt.savefig(path + "/Losses.png")
    plt.close()
    # Metrics
    plt.figure()
    plt.plot(x,evalSDRs, label='Eval SDR')
    plt.plot(x, evalSARs, label='Eval SAR')
    plt.plot(x, evalSIRs, label='Eval SIR')
    plt.legend()
    plt.savefig(path + "/Metrics.png")

def output_visualsDuet(viz_rows, inputAmps, output, labels, inputAudio, outputAudios, path, filePath, images, args):
    """Saves the duet visuals and audios to correct folders and stores the file locations for html visualization"""
    makedirs(path + filePath, False)
    row_elements = []

    # Input audio
    inputAudioFile =  path + filePath + "/InputAudio.wav"
    wavfile.write(inputAudioFile, args['audRate'], inputAudio)

    # Input spectrogram
    inputSpecFile = path + filePath + "/InputSpec.png"
    plt.figure()
    plt.imshow(torch.log(torch.abs(inputAmps[0][0])))
    plt.savefig(inputSpecFile)
    plt.close()

    # Stores the file paths for the html visualization
    row_elements += [{'text': path}, {'image': args['vizPath'] + inputSpecFile, 'audio': args['vizPath'] + inputAudioFile}]
    
    # Input image
    inputFrameFile = path + filePath + "/frame{}.png".format(0)
    plt.figure()
    plt.imshow(images[0])
    plt.savefig(inputFrameFile)
    plt.close()
    for i in range(len(outputAudios)):
        # Output audio
        outputAudioFile = path + filePath + "/OutputAudio{}.wav".format(i)
        wavfile.write(outputAudioFile, args['audRate'], outputAudios[i])

        # Output spectrogram
        outputMaskFile = path + filePath + "/outputMask{}.png".format(i)
        plt.figure()
        plt.imshow((output[labels[0][i].item()]) > args['threshold'])
        plt.savefig(outputMaskFile)
        plt.close()

        
        row_elements += [
                {'image': args['vizPath'] + inputFrameFile},
                {'audio': args['vizPath'] + outputAudioFile},
                {'image': args['vizPath'] + outputMaskFile}
                ]
    viz_rows.append(row_elements)

def output_visuals(viz_rows, inputAmps, output, groundTruth, labels, inputAudio, groundTruthAudios, outputAudios, path, filePath, images, args):
    """Saves the combined audios (test set) visuals and audios to correct folders and saves the file locations for html visualization"""
    makedirs(path + filePath, False)
    row_elements = []

    # Input audio
    inputAudioFile =  path + filePath + "/InputAudio.wav"
    wavfile.write(inputAudioFile, args['audRate'], inputAudio)

    # Input spectrogram
    inputSpecFile = path + filePath + "/InputSpec.png"
    plt.figure()
    plt.imshow(torch.log(torch.abs(inputAmps[0][0])))
    plt.savefig(inputSpecFile)
    plt.close()

    row_elements += [{'text': path}, {'image': args['vizPath'] + inputSpecFile, 'audio': args['vizPath'] + inputAudioFile}]
    

    for i in range(len(outputAudios)):
        # GT audio
        groundAudioFile = path + filePath + "/GTAudio{}.wav".format(i)
        wavfile.write(groundAudioFile, args['audRate'], groundTruthAudios[i])

        # Output audio
        outputAudioFile = path + filePath + "/OutputAudio{}.wav".format(i)
        wavfile.write(outputAudioFile, args['audRate'], outputAudios[i])

        # GT spectrogram
        if args['lossFunc'] == 'BCE':
            groundTruth1 = groundTruth[labels[0][i].item()]
        elif args['lossFunc'] == 'cross':
            if args['crossProb']:
                groundTruth1 = (groundTruth[labels[0][i].item()] > 0.5)
            else:
                groundTruth1 = (groundTruth == labels[0][i].item())
        groundMaskFile = path + filePath + "/GTMask{}.png".format(i)
        plt.figure()
        plt.imshow(groundTruth1)
        plt.savefig(groundMaskFile)
        plt.close()

        # Output spectrogram
        outputMaskFile = path + filePath + "/outputMask{}.png".format(i)
        plt.figure()
        plt.imshow((output[labels[0][i].item()]) > args['threshold'])
        plt.savefig(outputMaskFile)
        plt.close()

        # Input frame
        inputFrameFile = path + filePath + "/frame{}.png".format(i)
        plt.figure()
        plt.imshow(images[i])
        plt.savefig(inputFrameFile)
        plt.close()

        row_elements += [
                {'image': args['vizPath'] + inputFrameFile},
                {'audio': args['vizPath'] + outputAudioFile},
                {'audio': args['vizPath'] + groundAudioFile},
                {'image': args['vizPath'] + outputMaskFile},
                {'image': args['vizPath'] + groundMaskFile}
                ]
    viz_rows.append(row_elements)

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
                groundTruth1 = (groundTruth[labels[0][0].item()] > 0.1).astype(int)
                groundTruth2 = (groundTruth[labels[0][1].item()] > 0.1).astype(int)
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
        semanticMask = np.argmax(np.concatenate((outputs[0].detach().cpu().numpy(), thresholdArray[np.newaxis,:,:]),0), 0)
        
        unique, counts = np.unique(semanticMask, return_counts=True)
        result = np.column_stack((unique, counts)) 
        print (result)
        
        
        '''
        output1 = output1*(semanticMask == classNumber1).astype(int)
        output2 = output2*(semanticMask == classNumber2).astype(int)
        '''
        # Multiply with binary mask to get the output audio magnitudes
        output1 = inputAmps*(semanticMask == classNumber1).astype(int)
        output2 = inputAmps*(semanticMask == classNumber2).astype(int)

    else:
        # Form output by thresholding the network output
        output1 = inputAmps*(output1>args['threshold'])
        output2 = inputAmps*(output2>args['threshold'])

    # Multiply the signals with the phase information of the mixed audio
    inputAudio = inputAmps.astype(complex) * np.exp(1j*phase)

    groundTruthAudio1 = groundTruth1.astype(complex) * np.exp(1j*phase)
    groundTruthAudio2 = groundTruth2.astype(complex) * np.exp(1j*phase)

    outputAudio1 = output1.astype(complex) * np.exp(1j*phase)
    outputAudio2 = output2.astype(complex) * np.exp(1j*phase)

    # Get the time domain signal with ISTFT
    inputAudio = lb.istft(inputAudio[0], hop_length = args['stftHop'], window='hann')

    groundTruthAudio1 = lb.istft(groundTruthAudio1[0], hop_length = args['stftHop'], window='hann') #if not real [0]
    groundTruthAudio2 = lb.istft(groundTruthAudio2[0], hop_length = args['stftHop'], window='hann')

    outputAudio1 = lb.istft(outputAudio1[0], hop_length = args['stftHop'], window='hann')
    outputAudio2 = lb.istft(outputAudio2[0], hop_length = args['stftHop'], window='hann')

    groundTruthAudios = [groundTruthAudio1, groundTruthAudio2]
    outputAudios = [outputAudio1, outputAudio2]

    return inputAudio, np.array(groundTruthAudios), np.array(outputAudios), semanticMask

def evaluate_audio(net_audio, loader, epoch, args, crit, device, path):
    """Evaluate the performance of the learned audio model using the test set"""
    print('Evaluating at {} epochs...'.format(epoch))

    # Turn off gradient calculation for better performance and to make sure system doesn't learn during eval
    torch.set_grad_enabled(False)

    size = 0
    averageLoss = 0

    # Init HTML for visualization
    visualizer = HTMLVisualizer(os.path.join(path, 'index.html'))
    header = ['Filename', 'Input Mixed Audio']
    for n in range(1, 3):
        header += ['Frame {:d}'.format(n),
                   'Predicted Audio {:d}'.format(n),
                   'GroundTruth Audio {}'.format(n),
                   'Predicted Mask {}'.format(n),
                   'GroundTruth Mask {}'.format(n)]
    header += ['Metrics']
    visualizer.add_header(header)
    vis_rows = []


    # Init the evaluation metrics
    averageSDR = 0
    averageSIR = 0
    averageSAR = 0
    averageCorrectLabel = 0
    sdrs = []
    totalRightClasses = 0

    # Sigmoid layer for output activation
    sigmoid = nn.Sigmoid()

    for i, data in enumerate(loader, 0):
        ampMix, inputAmp, inputPhase, labels, images  = data[0].to(device), data[1].to(device), data[2].to(device), data[3].to(device), data[4]
        '''
        ampMix = [batchSize, 21, 512, 256] Per instrument audios
        inputAmp =  [batchSize, 512, 256] The magnitude of all audio sources combined
        inputPhase = [batchSize, 512, 256] The phase of all audio sources combined
        labels = [batchSize, mixSize] All the labels used in the mixture
        image = [batchSize, mixSize, (imageSize)] All images stored in a python array for vizualisation
        '''

        size += args['batch_size_gpu']

        # Forward pass
        inputAmpWarped, groundTruth = preProcessAudio(ampMix, inputAmp, device, args['lossFunc'], args['crossProb'], args['log'])

        # Calculate log STFT if desired
        if args['log']:
            inputAmpWarped = torch.log(inputAmpWarped+0.0001) # Add small number as you can't calculate log for 0 value
        
        outputs = net_audio(inputAmpWarped).to(device)

        loss = crit(outputs, groundTruth)
        averageLoss += loss

        # Reshape into the original size
        outputs, groundTruth = postProcessAudio(outputs, groundTruth, device, args['lossFunc'], args['crossProb'])
        
        # Activation with sigmoid
        outputs = sigmoid(outputs)
        
        # Reform input audio, GT audio and output audio
        inputAudio, groundTruthAudios, outputAudios, semanticMask = form_audio(inputAmp, inputPhase, groundTruth, outputs, labels, args, ampMix)
        
        #Semantic mask plot
        plt.figure()

        cmap = colors.ListedColormap(['darkorange','mediumblue','paleturquoise','darkred','antiquewhite','palegreen','magenta','midnightblue','chocolate','peru','forestgreen',
                                'rebeccapurple','lime','lightsteelblue', 'lightseagreen','mediumvioletred','red','indianred','limegreen','lightgreen','cornflowerblue', 'white'])
        heat = plt.pcolor(semanticMask, vmin=0, vmax=22, cmap=cmap)
        cbar = plt.colorbar(heat, ticks=[0.5,1.5,2.5,3.5,4.5,5.5,6.5,7.5,8.5,9.5,10.5,11.5,12.5,13.5,14.5,15.5,16.5,17.5,18.5,19.5,20.5,21.5])
        cbar.ax.set_yticklabels(['accordion','acoustic_guitar','bagpipe','banjo','bassoon','cello','clarinet','congas','drum','electric_bass',
                        'erhu','flute','guzheng','piano','pipa','saxophone','trumpet','tuba','ukulele','violin','xylophone', 'background'])
        ax = plt.gca()
        ax.set_ylim(ax.get_ylim()[::-1])
        plt.savefig("gt2.png")
        plt.close()
        
        

        # Calculate metrics
        sdr, sir, sar, perms = 0,0,0,0
        try:
            sdr, sir, sar, perms = bss_eval_sources(groundTruthAudios+0.0001, outputAudios+0.0001)
        except:
            print('null audio')
        
        print(sdr, " ", sir, " ", sar)
        averageSDR += np.mean(sdr)
        averageSIR += np.mean(sir)
        averageSAR += np.mean(sar)
        
        
        outputs = torch.Tensor.cpu(outputs[0])
        groundTruth = torch.Tensor.cpu(groundTruth[0])
        inputAmps = torch.Tensor.cpu(inputAmp)


        classNumber = [labels[0][0].item(),labels[0][1].item()]

        # TODO: Change to get more meaningful results
        semanticMask = torch.argmax(outputs, 0)
        count = torch.sum((semanticMask == (classNumber[0] or classNumber[1])))
        averageCorrectLabel += (count/(outputs[0].nelement())).item()

        # Save the audios and images
        filePath = "/{}".format(i)
        output_visuals(vis_rows, inputAmps, outputs, groundTruth, labels,inputAudio, groundTruthAudios, outputAudios, path, filePath, images[0], args)

        # Handle and save metrics
        metrics = ""
        if not isinstance(sdr, int):
            # SDR for median calculation
            sdrs.append(sdr[0])
            sdrs.append(sdr[1])

            # Save metrics
            for metIdx in range(len(sdr)):
                metrics += "<br> SDR{}: {:.2f}".format(metIdx, sdr[metIdx])
                metrics += "<br> SIR{}: {:.2f}".format(metIdx, sir[metIdx])
                metrics += "<br> SAR{}: {:.2f}".format(metIdx, sar[metIdx])
            metrics += str(labels[:,0])
            #metrics += str(outputLabel)
        else:
            sdrs.append(sdr)
        metrics += str(labels[:,0])
        #metrics += str(outputLabel)
        vis_rows.append([{'text': metrics}])
   
    visualizer.add_rows(vis_rows)
    visualizer.write_html()

    
    sdrs = np.array(sdrs)
    print("Frame accuracy: ", str(totalRightClasses/size))
    print('Average SDR: ',str(averageSDR/(size/args['batch_size_gpu'])))
    print("Median SDR: ", str(np.median(sdrs)))
    print("Average correct label: ", str(averageCorrectLabel/size))
    return (averageLoss/size), (averageSDR/(size/args['batch_size_gpu'])), (averageSIR/(size/args['batch_size_gpu'])), (averageSAR/(size/args['batch_size_gpu']))
  
def evaluate_combined(nets, loader, epoch, args, crit, device, path):
    """Evaluate the performance of the learned model using the test set"""
    print('Evaluating at {} epochs...'.format(epoch))

    # Turn off gradient calculation for better performance and to make sure system doesn't learn during eval
    torch.set_grad_enabled(False)

    size = 0
    averageLoss = 0
    (net_frame, net_audio) = nets

    # Init HTML for visualization
    visualizer = HTMLVisualizer(os.path.join(path, 'index.html'))
    header = ['Filename', 'Input Mixed Audio']
    for n in range(1, 3):
        header += ['Frame {:d}'.format(n),
                   'Predicted Audio {:d}'.format(n),
                   'GroundTruth Audio {}'.format(n),
                   'Predicted Mask {}'.format(n),
                   'GroundTruth Mask {}'.format(n)]
    header += ['Metrics']
    visualizer.add_header(header)
    vis_rows = []


    # Init the evaluation metrics
    averageSDR = 0
    averageSIR = 0
    averageSAR = 0
    averageCorrectLabel = 0
    sdrs = []
    totalRightClasses = 0

    # Sigmoid layer for output activation
    sigmoid = nn.Sigmoid()

    for i, data in enumerate(loader, 0):
        ampMix, inputAmp, inputPhase, labels, frames, images  = data[0].to(device), data[1].to(device), data[2].to(device), data[3].to(device), data[4].to(device), data[5]
        '''
        ampMix = [batchSize, 21, 512, 256] Per instrument audios
        inputAmp =  [batchSize, 512, 256] The magnitude of all audio sources combined
        inputPhase = [batchSize, 512, 256] The phase of all audio sources combined
        labels = [batchSize, mixSize] All the labels used in the mixture
        frames = [batchSize, mixSize, 3, 500, 500] All the frames stored in separate tensors
        image = [batchSize, mixSize, (imageSize)] All images stored in a python array for vizualisation
        '''

        size += args['batch_size_gpu']

        # Forward pass
        inputAmpWarped, groundTruth = preProcessAudio(ampMix, inputAmp, device, args['lossFunc'], args['crossProb'], args['log'])

        # Get the instrument prediction and combine the predictions from the two combined frames
        
        frame1 = torch.reshape(frames[:,0,:,:,:], (args['batch_size_gpu'],3,500,500))
        frameFeature1 = net_frame(frame1)
        frameWeights1 = sigmoid(frameFeature1)
        frameWeights1 = frameWeights1.reshape((1,args['batch_size_gpu'],21,1,1))

        frame2 = torch.reshape(frames[:,1,:,:,:], (args['batch_size_gpu'],3,500,500))
        frameFeature2 = net_frame(frame2)
        frameWeights2 = sigmoid(frameFeature2)
        frameWeights2 = frameWeights2.reshape((1,args['batch_size_gpu'],21,1,1))
        framesFeatures = torch.cat([frameWeights1, frameWeights2],1)

        outputLabel = torch.argmax(frameFeature1, dim=1)

        print(labels[:,0], "  ", outputLabel)
        
        
        
        totalRightClasses += (labels[:,0] == outputLabel).sum()
        

        # Calculate the outputs given the input audio mix and frame predictions
        if args['log']:
            inputAmpWarped = torch.log(inputAmpWarped+0.0001) # Add small number as you can't calculate log for 0 value
        
        outputs = net_audio(inputAmpWarped).to(device)

        loss = crit(outputs, groundTruth)
        averageLoss += loss

        '''
        GT before reshape
        groundTruthh = torch.Tensor.cpu(groundTruth[0])
        groundTruth1 = (groundTruthh[labels[0][0].item()] > 0.5)
        groundTruth2 = (groundTruthh[labels[0][1].item()] > 0.5)
        plt.figure()
        plt.imshow(groundTruth1)
        plt.savefig("gt1.png")
        plt.close()
        plt.figure()
        plt.imshow(groundTruth2)
        plt.savefig("gt2.png")
        plt.close()
        
        '''
        
        # Reshape into the original size
        outputs, groundTruth = postProcessAudio(outputs,groundTruth, device, args['lossFunc'],args['crossProb'])
        
        # Activation with sigmoid
        outputs = sigmoid(outputs)
        
        # Reform input audio, GT audio and output audio
        inputAudio, groundTruthAudios, outputAudios, semanticMask = form_audio(inputAmp, inputPhase, groundTruth, outputs, labels, args, ampMix)
        
        #Semantic mask plot
        plt.figure()
        #colormap = colors.Colormap("jotain", 22)
        cmap = colors.ListedColormap(['darkorange','mediumblue','paleturquoise','darkred','antiquewhite','palegreen','magenta','midnightblue','chocolate','peru','forestgreen',
                                'rebeccapurple','lime','lightsteelblue', 'lightseagreen','mediumvioletred','red','indianred','limegreen','lightgreen','cornflowerblue', 'white'])
        heat = plt.pcolor(semanticMask, vmin=0, vmax=22, cmap=cmap)
        cbar = plt.colorbar(heat, ticks=[0.5,1.5,2.5,3.5,4.5,5.5,6.5,7.5,8.5,9.5,10.5,11.5,12.5,13.5,14.5,15.5,16.5,17.5,18.5,19.5,20.5,21.5])
        cbar.ax.set_yticklabels(['accordion','acoustic_guitar','bagpipe','banjo','bassoon','cello','clarinet','congas','drum','electric_bass',
                        'erhu','flute','guzheng','piano','pipa','saxophone','trumpet','tuba','ukulele','violin','xylophone', 'background'])
        ax = plt.gca()
        ax.set_ylim(ax.get_ylim()[::-1])
        plt.savefig("gt2.png")
        plt.close()
        
        

        # Calculate metrics
        sdr, sir, sar, perms = 0,0,0,0
        try:
            sdr, sir, sar, perms = bss_eval_sources(groundTruthAudios+0.0001, outputAudios+0.0001)
        except:
            print('null audio')
        
        print(sdr, " ", sir, " ", sar)
        averageSDR += np.mean(sdr)
        averageSIR += np.mean(sir)
        averageSAR += np.mean(sar)
        
        
        outputs = torch.Tensor.cpu(outputs[0])
        groundTruth = torch.Tensor.cpu(groundTruth[0])
        inputAmps = torch.Tensor.cpu(inputAmp)


        classNumber = [labels[0][0].item(),labels[0][1].item()]

        # TODO: Change to get more meaningful results
        semanticMask = torch.argmax(outputs, 0)
        count = torch.sum((semanticMask == (classNumber[0] or classNumber[1])))
        averageCorrectLabel += (count/(outputs[0].nelement())).item()

        # Save the audios and images
        filePath = "/{}".format(i)
        output_visuals(vis_rows, inputAmps, outputs, groundTruth, labels,inputAudio, groundTruthAudios, outputAudios, path, filePath, images[0], args)

        # Handle and save metrics
        metrics = ""
        if not isinstance(sdr, int):
            # SDR for median calculation
            sdrs.append(sdr[0])
            sdrs.append(sdr[1])

            # Save metrics
            for metIdx in range(len(sdr)):
                metrics += "<br> SDR{}: {:.2f}".format(metIdx, sdr[metIdx])
                metrics += "<br> SIR{}: {:.2f}".format(metIdx, sir[metIdx])
                metrics += "<br> SAR{}: {:.2f}".format(metIdx, sar[metIdx])
            metrics += str(labels[:,0])
            #metrics += str(outputLabel)
        else:
            sdrs.append(sdr)
        metrics += str(labels[:,0])
        #metrics += str(outputLabel)
        vis_rows.append([{'text': metrics}])
   
    visualizer.add_rows(vis_rows)
    visualizer.write_html()

    
    sdrs = np.array(sdrs)
    print("Frame accuracy: ", str(totalRightClasses/size))
    print('Average SDR: ',str(averageSDR/(size/args['batch_size_gpu'])))
    print("Median SDR: ", str(np.median(sdrs)))
    print("Average correct label: ", str(averageCorrectLabel/size))
    return (averageLoss/size), (averageSDR/(size/args['batch_size_gpu'])), (averageSIR/(size/args['batch_size_gpu'])), (averageSAR/(size/args['batch_size_gpu']))

def evaluateDuet(nets, loader, epoch, args, crit, device, path):
    """Evaluate the performance of the learned model using the duet data
    TODO: Clean the code and try to locate the fault
    """
    print('Evaluating at {} epochs...'.format(epoch))
    torch.set_grad_enabled(False)
    size = 0
    (net_frame, net_audio) = nets

    # Init HTML for vizualisation
    visualizer = HTMLVisualizer(os.path.join(path, 'index.html'))
    header = ['Filename', 'Input Mixed Audio']
    for n in range(1, 3):
        header += ['Frame {:d}'.format(n),
                   'Predicted Audio {:d}'.format(n),
                   'Predicted Mask {}'.format(n)]
    visualizer.add_header(header)
    vis_rows = []


    sigmoid = nn.Sigmoid()
    totalRightClasses = 0
    groundTruth = 0
    for i, data in enumerate(loader, 0):
        inputAmp, inputPhase, labels, frame, image = data[0].to(device), data[1].to(device), data[2].to(device), data[3].to(device), data[4]
        '''
        inputAmp =  [batchSize, 512, 256] The magnitude of all audio sources combined
        inputPhase = [batchSize, 512, 256] The pahse of all audio sources combined
        labels = [batchSize, mixSize] All the labels used in the mixture
        frames = [batchSize, mixSize, 3, 500, 500] All the frames stored in separate tensors
        image = [batchSize, mixSize, (imageSize)] All images stored in a python array for vizualisation
        '''

        size += args['batch_size_gpu']
        # forward pass
        ampMix = 0
        grid_warp = torch.from_numpy(
                warpgrid(args['batch_size_gpu'], 256, 256, warp=True)).to(device)
        input = F.grid_sample(inputAmp, grid_warp)

        # Calculate the classification results and interpolate the results to match audio size
        frame = torch.reshape(frame[:,0,:,:,:], (args['batch_size_gpu'],3,500,500))

        frameFeature = net_frame(frame)
        frameWeights = sigmoid(frameFeature)
        frameWeights = frameWeights.reshape((1,args['batch_size_gpu'],21,1,1))
        
        
        
    
        frameWeights = torch.cat([frameWeights, frameWeights],0)

        # Calculate the outputs given the input audio mix and the interpolated frame weights
        if args['log']:
            input = torch.log(input+0.0001)
        outputs = net_audio(input, frameWeights).to(device)
        
        # Reshape into the original size
        grid_warp = torch.from_numpy(
                warpgrid(args['batch_size_gpu'], 512, 256, warp=False)).to(device)

        output = torch.zeros((args['batch_size_gpu'],21,512,256))
        for n in range(21):
            output[:,n:n+1,:,:] = F.grid_sample(outputs[:,n:n+1,:,:], grid_warp)
        
        # Activation with sigmoid
        output = sigmoid(output)
        
        # Reform input audio, GT audio and output audio
        inputAmps = inputAmp[0].detach().cpu().numpy()
        
        output1 = output[0][labels[0][0]].detach().cpu().numpy()
        output2 = output[0][labels[0][1]].detach().cpu().numpy()
        phase = inputPhase[0].detach().cpu().numpy()
        
        semanticMask = 0
        # Form output from argmax information
        if args['semantic']:
            classNumber1 = labels[0][0].item()
            classNumber2 = labels[0][1].item()
            threshHoldArray = np.ones(output[0].size()[1:])*args['threshold']
            semanticMask = np.argmax(np.concatenate((output[0].detach().cpu().numpy(), threshHoldArray[np.newaxis,:,:]),0), 0)
            '''
            unique, counts = np.unique(semanticMask, return_counts=True)
            result = np.column_stack((unique, counts)) 
            print (result)
            '''
            
            '''
            output1 = output1*(semanticMask == classNumber1).astype(int)
            output2 = output2*(semanticMask == classNumber2).astype(int)
            '''
            
            output1 = inputAmps*(semanticMask == classNumber1).astype(int)
            output2 = inputAmps*(semanticMask == classNumber2).astype(int)

        else:
            # Form output by thresholding the netowrk output
            output1 = inputAmps*(output1>args['threshold'])
            output2 = inputAmps*(output2>args['threshold'])

        # Multiply the signals with the phase information of the mixed audio
        inputAudio = inputAmps.astype(complex) * np.exp(1j*phase)

        outputAudio1 = output1.astype(complex) * np.exp(1j*phase)
        outputAudio2 = output2.astype(complex) * np.exp(1j*phase)

        inputAudio = lb.istft(inputAudio[0], hop_length = args['stftHop'], window='hann')

        outputAudio1 = lb.istft(outputAudio1[0], hop_length = args['stftHop'], window='hann')
        outputAudio2 = lb.istft(outputAudio2[0], hop_length = args['stftHop'], window='hann')

        outputAudios = [outputAudio1, outputAudio2]
        
        output = torch.Tensor.cpu(output[0])
        inputAmp = torch.Tensor.cpu(inputAmp)

        filePath = "/{}".format(i)
        output_visualsDuet(vis_rows, inputAmp, output, labels, inputAudio, outputAudios, path, filePath, image[0], args)

   
    visualizer.add_rows(vis_rows)
    visualizer.write_html()

    return

def evaluate_img(net, loader, epoch, args, crit, device):
    """Evaluate the performance of the learned image model using the test set"""

    print('Evaluating at {} epochs...'.format(epoch))
    # Turn off gradient calculation for better performance and to make sure system dosen't learn durin eval
    torch.set_grad_enabled(False)

    # Init variables for different metric calculations
    totalRightClasses = 0
    size = 0
    averageLoss = 0
    confMatrix = np.zeros((21,21))
    count = np.zeros((21,1))

    for i, data in enumerate(loader, 0):
        inputs, labels = data[0].to(device), data[1].to(device)
        size += args['batch_size_gpu']

        # forward pass
        outputs = net(inputs)

        loss = crit(outputs, labels)
        averageLoss += loss
        sigmoid = nn.Sigmoid()
        outputLabel = torch.argmax(sigmoid(outputs), dim = 1)

        # Calculate the labels for confusion matrix
        for i in range(args['batch_size_gpu']):
            confMatrix[labels[i], outputLabel[i]] += 1
            count[labels[i]] += 1
        
        print("labels: ",labels)
        print("outputLabel: ", outputLabel)

        # Calculate correct labels for accuracy calculation
        rightClass = (labels == outputLabel).sum()
        totalRightClasses += rightClass
    
    # Form and save the confusion matrix
    confMatrix = confMatrix/count
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.matshow(confMatrix, cmap=plt.cm.Blues, alpha=1)
    for i in range(confMatrix.shape[0]):
        for j in range(confMatrix.shape[1]):
            num = "{:.2f}".format(confMatrix[i, j])
            ax.text(x=j, y=i,s=num, va='center', ha='center', size='x-small')
    
    plt.xlabel('Predictions', fontsize=18)
    plt.ylabel('Actuals', fontsize=18)
    plt.title('Confusion Matrix', fontsize=18)
    plt.savefig("confusion.png")
    plt.close()

    # Calculate the accuracy of the model
    percentage = (totalRightClasses/size)*100
    averageLoss = averageLoss/(size/args['batch_size_gpu'])
    print('[Eval] epoch {}, percentage: {} , average loss: {}'.format(epoch, percentage,averageLoss))
    print(size)

    return percentage, averageLoss

def create_optimizer_frame(net, args):
    """Create a optimizer for frame network"""
    param_groups = [{'params': net.parameters(), 'lr': args['lr_image']}]
    return torch.optim.SGD(param_groups, momentum=args['momentum'])

def create_optimizer_audio(net, args):
    """Create a optimizer for audio network"""
    param_groups = [{'params': net.parameters(), 'lr': args['lr_audio']}]
    return torch.optim.SGD(param_groups, momentum=args['momentum'])

def create_optimizer(nets, args):
    """Create a optimizer for combined network"""
    (net_frame, net_audio) = nets
    print(net_frame.parameters())
    param_groups = [{'params': net_audio.parameters(), 'lr': args['lr_audio']},
                    {'params': net_frame.parameters(), 'lr': args['lr_image']}]
    return torch.optim.SGD(param_groups, momentum=args['momentum'])

def train_audio(net_audio, loader, optimizer, device, crit, epoch_iters, epoch, args):
    """Trains the audio network for one epoch"""

    # Turn on gradient calculation so that the system keeps and updates gradients
    torch.set_grad_enabled(True)

    outputs = 0

    # Init variables for different metric calculations
    trainLoss = 0
    size = 0

    for i, data in enumerate(loader, 0):
        ampMix, inputAmp, inputPhase, labels, images  = data[0].to(device), data[1].to(device), data[2].to(device), data[3], data[4]
        '''
        ampMix = [batchSize, 21, 512, 256] Per instrument audios
        inputAmp =  [batchSize, 512, 256] The magnitude of all audio sources combined
        inputPhase = [batchSize, 512, 256] The phase of all audio sources combined
        labels = [batchSize, mixSize] All the labels used in the mixture
        images = [batchSize, mixSize, (imageSize)] All images stored in a python array for vizualisation
        '''

        # Set gradients of all optimized tensors to zero
        optimizer.zero_grad()

        size += args['batch_size_gpu']

        # Warp the audio and calculate masks
        inputAmp, groundTruth = preProcessAudio(ampMix, inputAmp, device, args['lossFunc'], args['crossProb'], args['log'])

        # Calculate log of the input if desired
        if args['log']:
            inputAmp = torch.log(inputAmp+0.0001)

        outputs = net_audio(inputAmp).to(device)

        # Calculate loss and calculate the gradients for the layers
        loss = crit(outputs, groundTruth)
        trainLoss += loss
        loss.backward()

        # Use the gradients to update the weights of the system
        optimizer.step()

        print(str(loss.item()) + " " + str(i) + "/" + str(epoch_iters) + "    " + str(epoch) )

    return (trainLoss/size)
    
def trainFrame(net_image, loader, optimizer, device, crit):
    """Trains the frame network for one epoch"""
    running_loss = 0.0
    torch.set_grad_enabled(True)

    for i, data in enumerate(loader, 0):
        inputs, labels = data[0].to(device), data[1].to(device)
        '''
        inputs = [batchSize, 3, 500, 500] 
        labels =  [batchSize, mixSize] true labels of the frames
        '''

        optimizer.zero_grad()
        outputs = net_image(inputs)

        loss = crit(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()


def main(args):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)
    batch_size = args['num_gpus'] * args['batch_size_gpu']

    # Model builder takes care of the network init
    builder = ModelBuilder()

    # Build networks
    net_image = builder.buildFrame()
    net_audio = builder.buildAudio(1, 21, True)

    # Define loss function
    if args['lossFunc'] == 'cross':
        crit = torch.nn.CrossEntropyLoss(reduction = 'mean')
    elif args['lossFunc'] == 'BCE':
        pos_scalar = 30
        crit = torch.nn.BCEWithLogitsLoss(pos_weight = torch.Tensor(np.array([pos_scalar])).to(device))

    # Init the datasets
    dataset_train_image = ImgDataset(args['csvTrain'])
    dataset_val_image = ImgDataset(args['csvVal'], setType = 'val')

    dataset_train_audio = AudioDataset(args['csvTrain'], args)
    dataset_test_audio = AudioDataset(args['csvVal'], args, setType='val')

    dataset_val_duet = DuetDataset(args['csvDuetTrain'], args, setType='val')
    dataset_test_combined = CombinedDataset(args['csvVal'], args, setType='val')
    
    # Init the dataloader classes with datasets
    loader_image_train = torch.utils.data.DataLoader(
        dataset_train_image,
        batch_size = batch_size,
        num_workers = args['workers'],
        drop_last = True,
        shuffle = True
    )
    loader_image_val = torch.utils.data.DataLoader(
        dataset_val_image,
        batch_size = batch_size,
        num_workers = args['workers'],
        drop_last = True,
        shuffle = False
    )
    loader_audio_train = torch.utils.data.DataLoader(
        dataset_train_audio,
        batch_size = batch_size,
        num_workers = args['workers'],
        drop_last = True,
        shuffle = True
    )
    loader_audio_test = torch.utils.data.DataLoader(
        dataset_test_audio,
        batch_size = batch_size,
        num_workers = args['workers'],
        drop_last = True,
        shuffle = False
    ) 
    loader_duet_test = torch.utils.data.DataLoader(
        dataset_val_duet,
        batch_size = batch_size,
        num_workers = args['workers'],
        drop_last = True,
        shuffle = False
    )
    loader_combined_test = torch.utils.data.DataLoader(
        dataset_test_combined,
        batch_size = batch_size,
        num_workers = args['workers'],
        drop_last = True,
        shuffle = False
    )

    # TRAIN AUDIO
    if args['trainAudio']:

        epoch_iters = len(dataset_train_audio) // batch_size
        print('1 Epoch = {} iters'.format(epoch_iters))

        # Add network to used device
        net_audio.to(device)

        # Create optimizer
        optimizer = create_optimizer_audio(net_audio, args)

        if args['loadWeights']:
            if args['loadBest']:
                net_audio.load_state_dict(torch.load('weigths/combined/bestAudioOnly.pth'), strict = False)
            else:
                net_audio.load_state_dict(torch.load('weigths/combined/latestAudio.pth'))

        evalLosses = []
        trainLosses = []
        evalSDRs = []
        evalSIRs = []
        evalSARs = []
        
        for epoch in range(args['epochs']):
            # Init the path object for saving plots and images
            path = 'viz/audio'

            trainLoss = train_audio(net_audio, loader_audio_train, optimizer, device, crit, epoch_iters, epoch, args)
            trainLosses.append(trainLoss.item())

            #Eval classfication accuracy
            path += "/" + args['lossFunc']
            t = time.localtime()
            path += "/" + time.strftime("%H_%M_%S", t)
            if epoch % 1 == 0:
                makedirs(path, False)
                # Evaluate model and store metrics for plotting
                evalLoss, evalSDR, evalSIR, evalSAR = evaluate_audio(net_audio, loader_audio_test, epoch, args, crit, device, path)
                evalLosses.append(evalLoss.item())
                evalSDRs.append(evalSDR.item())
                evalSIRs.append(evalSIR.item())
                evalSARs.append(evalSAR.item())


                if args['saveWeights']:
                    torch.save(net_audio.state_dict(), 'weigths/combined/latestAudio.pth')
                
                # Plot metrics
                saveAndPlotMetrics(path, trainLosses, evalLosses, evalSDRs, evalSIRs, evalSARs)

                #if epoch % 4 == 0:
                #    args['threshold'] += 0.1

        print("Average SDR: ", str(sum(evalSDRs)/args['epochs']))
        print("Average SIR: ", str(sum(evalSIRs)/args['epochs']))
        print("Average SAR: ", str(sum(evalSARs)/args['epochs']))


    # TRAIN VIDEO
    if args['trainFrame']:
        epoch_iters = len(loader_image_train) // batch_size
        print('1 Epoch = {} iters'.format(epoch_iters))

        # Define loss function
        crit = torch.nn.CrossEntropyLoss()
        optimizer = create_optimizer_frame(net_image, args)

        evalAccuracy = []
        evalLoss = []

        net_image.to(device)
        if args['loadWeights']:
            net_image.load_state_dict(torch.load('weigths/frame/latest.pth'))
        
        for epoch in range(args['epochs']):
            trainFrame(net_image, loader_image_train, optimizer, device, crit)

            #Eval classfication accuracy
            if epoch % 3 == 0:
                accuracy, loss = evaluate_img(net_image, loader_image_val, epoch, args, crit, device)
                evalAccuracy.append(accuracy)
                evalLoss.append(loss)
                if args['saveWeights']:
                    torch.save(net_image.state_dict(),
                            'weigths/frame/latest.pth')
        
        if args['saveWeights']:
                    torch.save(net_image.state_dict(),
                            'weigths/frame/latest.pth')
        print(evalAccuracy)

    # TRAIN COMBINED
    if args['evalCombined']:

        epoch_iters = len(loader_combined_test) // batch_size
        print('1 Epoch = {} iters'.format(epoch_iters))
        

        if args['loadPreFrame']:
            net_image.load_state_dict(torch.load('weigths/frame/latest.pth'))
        
        if args['useConv']:
            # Modify ResNet18 so that the last 2 layers are removed 
            net_image = nn.Sequential(*list(net_image.children())[:-2])

        if args['freezeParams']:
            for param in net_image.parameters():
                param.requires_grad = False

        if args['useConv']:
            # Add a convolution layer so that the image features match the audio features
            net_image.add_module("conv2d",nn.Conv2d(512, 512, kernel_size=4, stride = 4, bias=False))

        # Add both networks to GPU 
        net_image.to(device)
        net_audio.to(device)

        # Create optimizer
        nets = (net_image, net_audio)
        optimizer = create_optimizer(nets, args)

        if args['loadWeights']:
            if args['loadBest']:
                if not args['loadPreFrame']:
                    net_image.load_state_dict(torch.load('weigths/combined/bestAudioOnlyImg.pth'))
                
                net_audio.load_state_dict(torch.load('weigths/combined/bestAudioOnly.pth'), strict = False)
            else:
                if not args['loadPreFrame']:
                    net_image.load_state_dict(torch.load('weigths/combined/latestImage.pth'))
                
                net_audio.load_state_dict(torch.load('weigths/combined/latestAudio.pth'))

        evalLosses = []
        trainLosses = []
        evalSDRs = []
        evalSIRs = []
        evalSARs = []
        
        for epoch in range(args['epochs']):
            # Init the path object for saving plots and images
            path = 'viz/combined'

            #Eval classfication accuracy
            path += "/" + args['lossFunc']
            t = time.localtime()
            path += "/" + time.strftime("%H_%M_%S", t)
            if epoch % 1 == 0:
                makedirs(path, False)
                # Evaluate model and store metrics for plotting
                evalLoss, evalSDR, evalSIR, evalSAR = evaluate_combined(nets, loader_combined_test, epoch, args, crit, device, path)
                evalLosses.append(evalLoss.item())
                evalSDRs.append(evalSDR.item())
                evalSIRs.append(evalSIR.item())
                evalSARs.append(evalSAR.item())


                if args['saveWeights']:
                    torch.save(net_audio.state_dict(), 'weigths/combined/latestAudio.pth')
                    torch.save(net_image.state_dict(), 'weigths/combined/latestImage.pth')
                
                # Plot metrics
                saveAndPlotMetrics(path, trainLosses, evalLosses, evalSDRs, evalSIRs, evalSARs)
                #if epoch % 4 == 0:
                #    args['threshold'] += 0.1
        print("Average SDR: ", str(sum(evalSDRs)/args['epochs']))
        print("Average SIR: ", str(sum(evalSIRs)/args['epochs']))
        print("Average SAR: ", str(sum(evalSARs)/args['epochs']))


    #PROCESS DUETS
    if args['evalDuet']:
        if args['useConv']:
            # Modify ResNet18 so that the last 2 layers are removed
            net_image = nn.Sequential(*list(net_image.children())[:-2])
            net_image.add_module("conv2d",nn.Conv2d(512, 512, kernel_size=4, stride = 4, bias=False))
        
        if args['loadPreFrame']:
            net_image.load_state_dict(torch.load('weigths/frame/latest.pth'))

        # Add both networks to GPU 
        net_image.to(device)
        net_audio.to(device)
        nets = (net_image, net_audio)

        if args['loadWeights']:
            if args['loadBest']:
                if not args['loadPreFrame']:
                    net_image.load_state_dict(torch.load('weigths/combined/bestImage.pth'))
                
                net_audio.load_state_dict(torch.load('weigths/combined/bestAudio.pth'))
            else:
                if not args['loadPreFrame']:
                    net_image.load_state_dict(torch.load('weigths/combined/latestImage.pth'))
                
                net_audio.load_state_dict(torch.load('weigths/combined/latestAudio.pth'))
        
        for epoch in range(args['epochs']):
            # Init the path object for saving plots and images
            path = 'viz/combined/duet'
            path += "/" + args['lossFunc']
            t = time.localtime()
            path += "/" + time.strftime("%H_%M_%S", t)
            makedirs(path, False)

            evaluateDuet(nets, loader_duet_test, epoch, args, crit, device, path)
            #if epoch % 1 == 0:
                #args['threshold'] += 0.1
        
if __name__ == '__main__':

    args = arguments.getArgs()

    main(args)
