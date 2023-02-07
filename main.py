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
from utils import warpgrid, \
    makedirs, preProcessAudio, postProcessAudio, form_audio

from viz import HTMLVisualizer, save_and_plot_metrics, output_visuals_duet, output_visuals


def evaluate_audio(net_audio, loader, epoch, args, crit, device, path):
    """Evaluate the performance of the learned audio model using the test set"""
    print('Evaluating at {} epochs...'.format(epoch))

    # Turn off gradient calculation for better performance and to make sure system doesn't learn during eval
    torch.set_grad_enabled(False)

    size = 0
    average_loss = 0

    # Init HTML for visualization
    visualizer = HTMLVisualizer(os.path.join(path, 'index.html'))
    header = ['Filename', 'Input Mixed Audio']
    for n in range(1, 3):
        header += ['Frame {:d}'.format(n),
                   'Predicted Audio {:d}'.format(n),
                   'ground_truth Audio {}'.format(n),
                   'Predicted Mask {}'.format(n),
                   'ground_truth Mask {}'.format(n)]
    header += ['Metrics']
    visualizer.add_header(header)
    vis_rows = []

    # Init the evaluation metrics
    average_sdr = 0
    average_sir = 0
    average_sar = 0
    average_correct_label = 0
    sdrs = []
    total_right_classes = 0

    # Sigmoid layer for output activation
    sigmoid = nn.Sigmoid()

    for i, data in enumerate(loader, 0):
        amp_mix, input_amp, input_phase, labels, images = data[0].to(device), data[1].to(
            device), data[2].to(device), data[3].to(device), data[4]
        '''
        amp_mix = [batchSize, 21, 512, 256] Per instrument audios
        input_amp =  [batchSize, 512, 256] The magnitude of all audio sources combined
        input_phase = [batchSize, 512, 256] The phase of all audio sources combined
        labels = [batchSize, mixSize] All the labels used in the mixture
        image = [batchSize, mixSize, (imageSize)] All images stored in a python array for vizualisation
        '''

        size += args['batch_size_gpu']

        # Forward pass
        input_ampWarped, ground_truth = preProcessAudio(
            amp_mix, input_amp, device, args['lossFunc'], args['crossProb'], args['log'])

        # Calculate log STFT if desired
        if args['log']:
            # Add small number as you can't calculate log for 0 value
            input_ampWarped = torch.log(input_ampWarped+0.0001)

        outputs = net_audio(input_ampWarped).to(device)

        loss = crit(outputs, ground_truth)
        average_loss += loss

        # Reshape into the original size
        outputs, ground_truth = postProcessAudio(
            outputs, ground_truth, device, args['lossFunc'], args['crossProb'])

        # Activation with sigmoid
        outputs = sigmoid(outputs)

        # Reform input audio, GT audio and output audio
        input_audio, ground_truth_audios, output_audios, semantic_mask = form_audio(
            input_amp, input_phase, ground_truth, outputs, labels, args, amp_mix)

        # Semantic mask plot
        plt.figure()

        cmap = colors.ListedColormap(['darkorange', 'mediumblue', 'paleturquoise', 'darkred', 'antiquewhite', 'palegreen', 'magenta', 'midnightblue', 'chocolate', 'peru', 'forestgreen',
                                      'rebeccapurple', 'lime', 'lightsteelblue', 'lightseagreen', 'mediumvioletred', 'red', 'indianred', 'limegreen', 'lightgreen', 'cornflowerblue', 'white'])
        heat = plt.pcolor(semantic_mask, vmin=0, vmax=22, cmap=cmap)
        cbar = plt.colorbar(heat, ticks=[0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5,
                            9.5, 10.5, 11.5, 12.5, 13.5, 14.5, 15.5, 16.5, 17.5, 18.5, 19.5, 20.5, 21.5])
        cbar.ax.set_yticklabels(['accordion', 'acoustic_guitar', 'bagpipe', 'banjo', 'bassoon', 'cello', 'clarinet', 'congas', 'drum', 'electric_bass',
                                 'erhu', 'flute', 'guzheng', 'piano', 'pipa', 'saxophone', 'trumpet', 'tuba', 'ukulele', 'violin', 'xylophone', 'background'])
        ax = plt.gca()
        ax.set_ylim(ax.get_ylim()[::-1])
        plt.savefig("gt2.png")
        plt.close()

        # Calculate metrics
        sdr, sir, sar, perms = 0, 0, 0, 0
        try:
            sdr, sir, sar, perms = bss_eval_sources(
                ground_truth_audios+0.0001, output_audios+0.0001)
        except:
            print('null audio')

        print(sdr, " ", sir, " ", sar)
        average_sdr += np.mean(sdr)
        average_sir += np.mean(sir)
        average_sar += np.mean(sar)

        outputs = torch.Tensor.cpu(outputs[0])
        ground_truth = torch.Tensor.cpu(ground_truth[0])
        input_amps = torch.Tensor.cpu(input_amp)

        class_number = [labels[0][0].item(), labels[0][1].item()]

        # TODO: Change to get more meaningful results
        semantic_mask = torch.argmax(outputs, 0)
        count = torch.sum((semantic_mask == (class_number[0] or class_number[1])))
        average_correct_label += (count/(outputs[0].nelement())).item()

        # Save the audios and images
        file_path = "/{}".format(i)
        output_visuals(vis_rows, input_amps, outputs, ground_truth, labels, input_audio,
                       ground_truth_audios, output_audios, path, file_path, images[0], args)

        # Handle and save metrics
        metrics = ""
        if not isinstance(sdr, int):
            # SDR for median calculation
            sdrs.append(sdr[0])
            sdrs.append(sdr[1])

            # Save metrics
            for met_idx in range(len(sdr)):
                metrics += "<br> SDR{}: {:.2f}".format(met_idx, sdr[met_idx])
                metrics += "<br> SIR{}: {:.2f}".format(met_idx, sir[met_idx])
                metrics += "<br> SAR{}: {:.2f}".format(met_idx, sar[met_idx])
            metrics += str(labels[:, 0])
            #metrics += str(outputLabel)
        else:
            sdrs.append(sdr)
        metrics += str(labels[:, 0])
        #metrics += str(outputLabel)
        vis_rows.append([{'text': metrics}])

    visualizer.add_rows(vis_rows)
    visualizer.write_html()

    sdrs = np.array(sdrs)
    print("Frame accuracy: ", str(total_right_classes/size))
    print('Average SDR: ', str(average_sdr/(size/args['batch_size_gpu'])))
    print("Median SDR: ", str(np.median(sdrs)))
    print("Average correct label: ", str(average_correct_label/size))
    return (average_loss/size), (average_sdr/(size/args['batch_size_gpu'])), (average_sir/(size/args['batch_size_gpu'])), (average_sar/(size/args['batch_size_gpu']))


def evaluate_combined(nets, loader, epoch, args, crit, device, path):
    """Evaluate the performance of the learned model using the test set"""
    print('Evaluating at {} epochs...'.format(epoch))

    # Turn off gradient calculation for better performance and to make sure system doesn't learn during eval
    torch.set_grad_enabled(False)

    size = 0
    average_loss = 0
    (net_frame, net_audio) = nets

    # Init HTML for visualization
    visualizer = HTMLVisualizer(os.path.join(path, 'index.html'))
    header = ['Filename', 'Input Mixed Audio']
    for n in range(1, 3):
        header += ['Frame {:d}'.format(n),
                   'Predicted Audio {:d}'.format(n),
                   'ground_truth Audio {}'.format(n),
                   'Predicted Mask {}'.format(n),
                   'ground_truth Mask {}'.format(n)]
    header += ['Metrics']
    visualizer.add_header(header)
    vis_rows = []

    # Init the evaluation metrics
    average_sdr = 0
    average_sir = 0
    average_sar = 0
    average_correct_label = 0
    sdrs = []
    total_right_classes = 0

    # Sigmoid layer for output activation
    sigmoid = nn.Sigmoid()

    for i, data in enumerate(loader, 0):
        amp_mix, input_amp, input_phase, labels, frames, images = data[0].to(device), data[1].to(
            device), data[2].to(device), data[3].to(device), data[4].to(device), data[5]
        '''
        amp_mix = [batchSize, 21, 512, 256] Per instrument audios
        input_amp =  [batchSize, 512, 256] The magnitude of all audio sources combined
        input_phase = [batchSize, 512, 256] The phase of all audio sources combined
        labels = [batchSize, mixSize] All the labels used in the mixture
        frames = [batchSize, mixSize, 3, 500, 500] All the frames stored in separate tensors
        image = [batchSize, mixSize, (imageSize)] All images stored in a python array for vizualisation
        '''

        size += args['batch_size_gpu']

        # Forward pass
        input_ampWarped, ground_truth = preProcessAudio(
            amp_mix, input_amp, device, args['lossFunc'], args['crossProb'], args['log'])

        # Get the instrument prediction and combine the predictions from the two combined frames
        frame_1 = torch.reshape(
            frames[:, 0, :, :, :], (args['batch_size_gpu'], 3, 500, 500))
        frame_feature1 = net_frame(frame_1)
        frame_weights1 = sigmoid(frame_feature1)
        frame_weights1 = frame_weights1.reshape(
            (args['batch_size_gpu'], 21, 1, 1))

        frame_2 = torch.reshape(
            frames[:, 1, :, :, :], (args['batch_size_gpu'], 3, 500, 500))
        frame_feature2 = net_frame(frame_2)
        frame_weights2 = sigmoid(frame_feature2)
        frame_weights2 = frame_weights2.reshape(
            (args['batch_size_gpu'], 21, 1, 1))

        frame_features = torch.maximum(frame_weights1, frame_weights2)

        # Calculate the frame network accuracy from the first frame
        outputLabel = torch.argmax(frame_feature1, dim=1)
        print(labels[:, 0], "  ", outputLabel)
        total_right_classes += (labels[:, 0] == outputLabel).sum()

        # Calculate the outputs given the input audio mix and frame predictions
        if args['log']:
            # Add small number as you can't calculate log for 0 value
            input_ampWarped = torch.log(input_ampWarped+0.0001)

        outputs = net_audio(input_ampWarped).to(device)

        loss = crit(outputs, ground_truth)
        average_loss += loss

        '''
        GT before reshape
        ground_truthh = torch.Tensor.cpu(ground_truth[0])
        ground_truth1 = (ground_truthh[labels[0][0].item()] > 0.5)
        ground_truth2 = (ground_truthh[labels[0][1].item()] > 0.5)
        plt.figure()
        plt.imshow(ground_truth1)
        plt.savefig("gt1.png")
        plt.close()
        plt.figure()
        plt.imshow(ground_truth2)
        plt.savefig("gt2.png")
        plt.close()
        
        '''
        # Combine frame probabilities with the audio network output
        outputs = torch.mul(outputs, frame_features)

        # Reshape into the original size
        outputs, ground_truth = postProcessAudio(
            outputs, ground_truth, device, args['lossFunc'], args['crossProb'])

        # Activation with sigmoid
        outputs = sigmoid(outputs)

        # Reform input audio, GT audio and output audio
        input_audio, ground_truth_audios, output_audios, semantic_mask = form_audio(
            input_amp, input_phase, ground_truth, outputs, labels, args, amp_mix)

        # Semantic mask plot
        plt.figure()
        #colormap = colors.Colormap("jotain", 22)
        cmap = colors.ListedColormap(['darkorange', 'mediumblue', 'paleturquoise', 'darkred', 'antiquewhite', 'palegreen', 'magenta', 'midnightblue', 'chocolate', 'peru', 'forestgreen',
                                      'rebeccapurple', 'lime', 'lightsteelblue', 'lightseagreen', 'mediumvioletred', 'red', 'indianred', 'limegreen', 'lightgreen', 'cornflowerblue', 'white'])
        heat = plt.pcolor(semantic_mask, vmin=0, vmax=22, cmap=cmap)
        cbar = plt.colorbar(heat, ticks=[0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5,
                            9.5, 10.5, 11.5, 12.5, 13.5, 14.5, 15.5, 16.5, 17.5, 18.5, 19.5, 20.5, 21.5])
        cbar.ax.set_yticklabels(['accordion', 'acoustic_guitar', 'bagpipe', 'banjo', 'bassoon', 'cello', 'clarinet', 'congas', 'drum', 'electric_bass',
                                 'erhu', 'flute', 'guzheng', 'piano', 'pipa', 'saxophone', 'trumpet', 'tuba', 'ukulele', 'violin', 'xylophone', 'background'])
        ax = plt.gca()
        ax.set_ylim(ax.get_ylim()[::-1])
        plt.savefig("gt2.png")
        plt.close()

        # Calculate metrics
        sdr, sir, sar, perms = 0, 0, 0, 0
        try:
            sdr, sir, sar, perms = bss_eval_sources(
                ground_truth_audios+0.0001, output_audios+0.0001)
        except:
            print('null audio')

        print(sdr, " ", sir, " ", sar)
        average_sdr += np.mean(sdr)
        average_sir += np.mean(sir)
        average_sar += np.mean(sar)

        outputs = torch.Tensor.cpu(outputs[0])
        ground_truth = torch.Tensor.cpu(ground_truth[0])
        input_amps = torch.Tensor.cpu(input_amp)

        class_number = [labels[0][0].item(), labels[0][1].item()]

        # TODO: Change to get more meaningful results
        semantic_mask = torch.argmax(outputs, 0)
        count = torch.sum((semantic_mask == (class_number[0] or class_number[1])))
        average_correct_label += (count/(outputs[0].nelement())).item()

        # Save the audios and images
        file_path = "/{}".format(i)
        output_visuals(vis_rows, input_amps, outputs, ground_truth, labels, input_audio,
                       ground_truth_audios, output_audios, path, file_path, images[0], args)

        # Handle and save metrics
        metrics = ""
        if not isinstance(sdr, int):
            # SDR for median calculation
            sdrs.append(sdr[0])
            sdrs.append(sdr[1])

            # Save metrics
            for met_idx in range(len(sdr)):
                metrics += "<br> SDR{}: {:.2f}".format(met_idx, sdr[met_idx])
                metrics += "<br> SIR{}: {:.2f}".format(met_idx, sir[met_idx])
                metrics += "<br> SAR{}: {:.2f}".format(met_idx, sar[met_idx])
            metrics += str(labels[:, 0])
            #metrics += str(outputLabel)
        else:
            sdrs.append(sdr)
        metrics += str(labels[:, 0])
        #metrics += str(outputLabel)
        vis_rows.append([{'text': metrics}])

    visualizer.add_rows(vis_rows)
    visualizer.write_html()

    sdrs = np.array(sdrs)
    print("Frame accuracy: ", str(total_right_classes/size))
    print('Average SDR: ', str(average_sdr/(size/args['batch_size_gpu'])))
    print("Median SDR: ", str(np.median(sdrs)))
    print("Average correct label: ", str(average_correct_label/size))
    return (average_loss/size), (average_sdr/(size/args['batch_size_gpu'])), (average_sir/(size/args['batch_size_gpu'])), (average_sar/(size/args['batch_size_gpu']))


def evaluate_duet(nets, loader, epoch, args, crit, device, path):
    """Evaluate the performance of the learned model using the duet data"""
    print('Evaluating at {} epochs...'.format(epoch))

    torch.set_grad_enabled(False)
    (net_frame, net_audio) = nets

    # Init HTML for visualization
    visualizer = HTMLVisualizer(os.path.join(path, 'index.html'))
    header = ['Filename', 'Input Mixed Audio']
    for n in range(1, 3):
        header += ['Frame {:d}'.format(n),
                   'Predicted Audio {:d}'.format(n),
                   'Predicted Mask {}'.format(n)]
    visualizer.add_header(header)
    vis_rows = []

    sigmoid = nn.Sigmoid()
    size = 0
    total_right_classes = 0
    ground_truth = 0
    for i, data in enumerate(loader, 0):
        input_amp, input_phase, labels, frame, image = data[0].to(device), data[1].to(
            device), data[2].to(device), data[3].to(device), data[4]
        '''
        input_amp =  [batchSize, 512, 256] The magnitude of all audio sources combined
        input_phase = [batchSize, 512, 256] The phase of all audio sources combined
        labels = [batchSize, mixSize] All the labels used in the mixture
        frames = [batchSize, mixSize, 3, 500, 500] All the frames stored in separate tensors
        image = [batchSize, (imageSize)] All images stored in a python array for visualization
        '''

        size += args['batch_size_gpu']

        # forward pass
        grid_warp = torch.from_numpy(
            warpgrid(args['batch_size_gpu'], 256, 256, warp=True)).to(device)
        input = F.grid_sample(input_amp, grid_warp)

        if not args['evalDuetAudioOnly']:
            # Calculate frame the classification results
            frame_feature = net_frame(frame)
            frame_weights = sigmoid(frame_feature)
            frame_weights = frame_weights.reshape(
                (args['batch_size_gpu'], 21, 1, 1))


        # Calculate the outputs given the input audio mix and the interpolated frame weights
        if args['log']:
            input = torch.log(input+0.0001)

        outputs = net_audio(input).to(device)

        if not args['evalDuetAudioOnly']:
            # Combine frame probabilities with the audio network output
            outputs = torch.mul(outputs, frame_weights)

        # Reshape into the original size
        grid_warp = torch.from_numpy(
            warpgrid(args['batch_size_gpu'], 512, 256, warp=False)).to(device)

        output = torch.zeros((args['batch_size_gpu'], 21, 512, 256))
        for n in range(21):
            output[:, n:n+1, :,
                   :] = F.grid_sample(outputs[:, n:n+1, :, :], grid_warp)

        # Activation with sigmoid
        output = sigmoid(output)

        # Reform input audio, GT audio and output audio
        input_amps = input_amp[0].detach().cpu().numpy()

        output_1 = output[0][labels[0][0]].detach().cpu().numpy()
        output_2 = output[0][labels[0][1]].detach().cpu().numpy()
        phase = input_phase[0].detach().cpu().numpy()

        semantic_mask = 0
        # Form output from argmax information
        if args['semantic']:
            class_number_1 = labels[0][0].item()
            class_number_2 = labels[0][1].item()
            threshold_array = np.ones(output[0].size()[1:])*args['threshold']
            semantic_mask = np.argmax(np.concatenate(
                (output[0].detach().cpu().numpy(), threshold_array[np.newaxis, :, :]), 0), 0)
            '''
            unique, counts = np.unique(semantic_mask, return_counts=True)
            result = np.column_stack((unique, counts)) 
            print (result)
            '''

            '''
            output_1 = output_1*(semantic_mask == class_number_1).astype(int)
            output_2 = output_2*(semantic_mask == class_number_2).astype(int)
            '''

            output_1 = input_amps*(semantic_mask == class_number_1).astype(int)
            output_2 = input_amps*(semantic_mask == class_number_2).astype(int)

        else:
            # Form output by thresholding the netowrk output
            output_1 = input_amps*(output_1 > args['threshold'])
            output_2 = input_amps*(output_2 > args['threshold'])

        # Multiply the signals with the phase information of the mixed audio
        input_audio = input_amps.astype(complex) * np.exp(1j*phase)

        output_audio_1 = output_1.astype(complex) * np.exp(1j*phase)
        output_audio_2 = output_2.astype(complex) * np.exp(1j*phase)

        input_audio = lb.istft(
            input_audio[0], hop_length=args['stftHop'], window='hann')

        output_audio_1 = lb.istft(
            output_audio_1[0], hop_length=args['stftHop'], window='hann')
        output_audio_2 = lb.istft(
            output_audio_2[0], hop_length=args['stftHop'], window='hann')

        output_audios = [output_audio_1, output_audio_2]

        output = torch.Tensor.cpu(output[0])
        input_amp = torch.Tensor.cpu(input_amp)

        file_path = "/{}".format(i)
        print(image[0].size())
        output_visuals_duet(vis_rows, input_amp, output, labels,
                            input_audio, output_audios, path, file_path, image[0], args)

    visualizer.add_rows(vis_rows)
    visualizer.write_html()

    return


def evaluate_img(net, loader, epoch, args, crit, device):
    """Evaluate the performance of the learned image model using the test set"""

    print('Evaluating at {} epochs...'.format(epoch))
    # Turn off gradient calculation for better performance and to make sure system dosen't learn durin eval
    torch.set_grad_enabled(False)

    # Init variables for different metric calculations
    total_right_classes = 0
    size = 0
    average_loss = 0
    confMatrix = np.zeros((21, 21))
    count = np.zeros((21, 1))

    for i, data in enumerate(loader, 0):
        inputs, labels = data[0].to(device), data[1].to(device)
        size += args['batch_size_gpu']

        # forward pass
        outputs = net(inputs)

        loss = crit(outputs, labels)
        average_loss += loss
        sigmoid = nn.Sigmoid()
        outputLabel = torch.argmax(sigmoid(outputs), dim=1)

        # Calculate the labels for confusion matrix
        for i in range(args['batch_size_gpu']):
            confMatrix[labels[i], outputLabel[i]] += 1
            count[labels[i]] += 1

        print("labels: ", labels)
        print("outputLabel: ", outputLabel)

        # Calculate correct labels for accuracy calculation
        rightClass = (labels == outputLabel).sum()
        total_right_classes += rightClass

    # Form and save the confusion matrix
    confMatrix = confMatrix/count
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.matshow(confMatrix, cmap=plt.cm.Blues, alpha=1)
    for i in range(confMatrix.shape[0]):
        for j in range(confMatrix.shape[1]):
            num = "{:.2f}".format(confMatrix[i, j])
            ax.text(x=j, y=i, s=num, va='center', ha='center', size='x-small')

    plt.xlabel('Predictions', fontsize=18)
    plt.ylabel('Actuals', fontsize=18)
    plt.title('Confusion Matrix', fontsize=18)
    plt.savefig("confusion.png")
    plt.close()

    # Calculate the accuracy of the model
    percentage = (total_right_classes/size)*100
    average_loss = average_loss/(size/args['batch_size_gpu'])
    print('[Eval] epoch {}, percentage: {} , average loss: {}'.format(
        epoch, percentage, average_loss))
    print(size)

    return percentage, average_loss


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
        amp_mix, input_amp, input_phase, labels, images = data[0].to(
            device), data[1].to(device), data[2].to(device), data[3], data[4]
        '''
        amp_mix = [batchSize, 21, 512, 256] Per instrument audios
        input_amp =  [batchSize, 512, 256] The magnitude of all audio sources combined
        input_phase = [batchSize, 512, 256] The phase of all audio sources combined
        labels = [batchSize, mixSize] All the labels used in the mixture
        images = [batchSize, mixSize, (imageSize)] All images stored in a python array for vizualisation
        '''

        # Set gradients of all optimized tensors to zero
        optimizer.zero_grad()

        size += args['batch_size_gpu']

        # Warp the audio and calculate masks
        input_amp, ground_truth = preProcessAudio(
            amp_mix, input_amp, device, args['lossFunc'], args['crossProb'], args['log'])

        # Calculate log of the input if desired
        if args['log']:
            input_amp = torch.log(input_amp+0.0001)

        outputs = net_audio(input_amp).to(device)

        # Calculate loss and calculate the gradients for the layers
        loss = crit(outputs, ground_truth)
        trainLoss += loss
        loss.backward()

        # Use the gradients to update the weights of the system
        optimizer.step()

        print(str(loss.item()) + " " + str(i) + "/" +
              str(epoch_iters) + "    " + str(epoch))

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
        crit = torch.nn.CrossEntropyLoss(reduction='mean')
    elif args['lossFunc'] == 'BCE':
        pos_scalar = 30
        crit = torch.nn.BCEWithLogitsLoss(
            pos_weight=torch.Tensor(np.array([pos_scalar])).to(device))

    # Init the datasets
    dataset_train_image = ImgDataset(args['csvTrain'])
    dataset_val_image = ImgDataset(args['csvVal'], setType='val')

    dataset_train_audio = AudioDataset(args['csvTrain'], args)
    dataset_test_audio = AudioDataset(args['csvVal'], args, setType='val')

    dataset_val_duet = DuetDataset(args['csvDuetTrain'], args, setType='val')
    dataset_test_combined = CombinedDataset(
        args['csvVal'], args, setType='val')

    # Init the dataloader classes with datasets
    loader_image_train = torch.utils.data.DataLoader(
        dataset_train_image,
        batch_size=batch_size,
        num_workers=args['workers'],
        drop_last=True,
        shuffle=True
    )
    loader_image_val = torch.utils.data.DataLoader(
        dataset_val_image,
        batch_size=batch_size,
        num_workers=args['workers'],
        drop_last=True,
        shuffle=False
    )
    loader_audio_train = torch.utils.data.DataLoader(
        dataset_train_audio,
        batch_size=batch_size,
        num_workers=args['workers'],
        drop_last=True,
        shuffle=True
    )
    loader_audio_test = torch.utils.data.DataLoader(
        dataset_test_audio,
        batch_size=batch_size,
        num_workers=args['workers'],
        drop_last=True,
        shuffle=False
    )
    loader_duet_test = torch.utils.data.DataLoader(
        dataset_val_duet,
        batch_size=batch_size,
        num_workers=args['workers'],
        drop_last=True,
        shuffle=False
    )
    loader_combined_test = torch.utils.data.DataLoader(
        dataset_test_combined,
        batch_size=batch_size,
        num_workers=args['workers'],
        drop_last=True,
        shuffle=False
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
                net_audio.load_state_dict(torch.load(
                    'weigths/audio/bestAudio.pth'), strict=False)
            else:
                net_audio.load_state_dict(torch.load(
                    'weigths/audio/latestAudio.pth'))

        eval_losses = []
        train_losses = []
        eval_sdrs = []
        eval_sirs = []
        eval_sars = []

        for epoch in range(args['epochs']):
            # Init the path object for saving plots and images
            path = 'viz/audio'

            trainLoss = train_audio(
                net_audio, loader_audio_train, optimizer, device, crit, epoch_iters, epoch, args)
            train_losses.append(trainLoss.item())

            # Eval classfication accuracy
            path += "/" + args['lossFunc']
            t = time.localtime()
            path += "/" + time.strftime("%H_%M_%S", t)
            if epoch % 1 == 0:
                makedirs(path, False)
                # Evaluate model and store metrics for plotting
                eval_loss, eval_sdr, eval_sir, eval_sar = evaluate_audio(
                    net_audio, loader_audio_test, epoch, args, crit, device, path)
                eval_losses.append(eval_loss.item())
                eval_sdrs.append(eval_sdr.item())
                eval_sirs.append(eval_sir.item())
                eval_sars.append(eval_sar.item())

                if args['saveWeights']:
                    torch.save(net_audio.state_dict(),
                               'weigths/audio/latestAudio.pth')

                # Plot metrics
                save_and_plot_metrics(
                    path, train_losses, eval_losses, eval_sdrs, eval_sirs, eval_sars)

                # if epoch % 4 == 0:
                #    args['threshold'] += 0.1

        print("Average SDR: ", str(sum(eval_sdrs)/args['epochs']))
        print("Average SIR: ", str(sum(eval_sirs)/args['epochs']))
        print("Average SAR: ", str(sum(eval_sars)/args['epochs']))

    # TRAIN VIDEO
    if args['trainFrame']:
        epoch_iters = len(loader_image_train) // batch_size
        print('1 Epoch = {} iters'.format(epoch_iters))

        # Define loss function
        crit = torch.nn.CrossEntropyLoss()
        optimizer = create_optimizer_frame(net_image, args)

        eval_accuracy = []
        eval_loss = []

        net_image.to(device)
        if args['loadWeights']:
            net_image.load_state_dict(torch.load(
                'weigths/frame/latestFrame.pth'))

        for epoch in range(args['epochs']):
            trainFrame(net_image, loader_image_train, optimizer, device, crit)

            # Eval classfication accuracy
            if epoch % 3 == 0:
                accuracy, loss = evaluate_img(
                    net_image, loader_image_val, epoch, args, crit, device)
                eval_accuracy.append(accuracy)
                eval_loss.append(loss)
                if args['saveWeights']:
                    torch.save(net_image.state_dict(),
                               'weigths/frame/latestFrame.pth')

        if args['saveWeights']:
            torch.save(net_image.state_dict(),
                       'weigths/frame/latestFrame.pth')
        print(eval_accuracy)

    # EVAL COMBINED
    if args['evalCombined']:

        epoch_iters = len(loader_combined_test) // batch_size
        print('1 Epoch = {} iters'.format(epoch_iters))

        # Add both networks to GPU
        net_image.to(device)
        net_audio.to(device)

        # Create optimizer
        nets = (net_image, net_audio)
        optimizer = create_optimizer(nets, args)

        if args['loadWeights']:
            if args['loadBest']:
                if not args['loadPreFrame']:
                    net_image.load_state_dict(torch.load(
                        'weigths/frame/bestFrame.pth'))

                net_audio.load_state_dict(torch.load(
                    'weigths/audio/bestAudio.pth'), strict=False)
            else:
                if not args['loadPreFrame']:
                    net_image.load_state_dict(
                        torch.load('weigths/frame/latestFrame.pth'))

                net_audio.load_state_dict(torch.load(
                    'weigths/audio/latestAudio.pth'))

        eval_losses = []
        train_losses = []
        eval_sdrs = []
        eval_sirs = []
        eval_sars = []

        for epoch in range(args['epochs']):
            # Init the path object for saving plots and images
            path = 'viz/combined'

            # Eval classfication accuracy
            path += "/" + args['lossFunc']
            t = time.localtime()
            path += "/" + time.strftime("%H_%M_%S", t)
            if epoch % 1 == 0:
                makedirs(path, False)
                # Evaluate model and store metrics for plotting
                eval_loss, eval_sdr, eval_sir, eval_sar = evaluate_combined(
                    nets, loader_combined_test, epoch, args, crit, device, path)
                eval_losses.append(eval_loss.item())
                eval_sdrs.append(eval_sdr.item())
                eval_sirs.append(eval_sir.item())
                eval_sars.append(eval_sar.item())

                # Plot metrics
                save_and_plot_metrics(
                    path, train_losses, eval_losses, eval_sdrs, eval_sirs, eval_sars)
                # if epoch % 4 == 0:
                #    args['threshold'] += 0.1
        print("Average SDR: ", str(sum(eval_sdrs)/args['epochs']))
        print("Average SIR: ", str(sum(eval_sirs)/args['epochs']))
        print("Average SAR: ", str(sum(eval_sars)/args['epochs']))

    # PROCESS DUETS
    if args['evalDuet']:
        if args['useConv']:
            # Modify ResNet18 so that the last 2 layers are removed
            net_image = nn.Sequential(*list(net_image.children())[:-2])
            net_image.add_module("conv2d", nn.Conv2d(
                512, 512, kernel_size=4, stride=4, bias=False))

        # Add both networks to GPU
        net_image.to(device)
        net_audio.to(device)
        nets = (net_image, net_audio)

        if args['loadWeights']:
            if args['loadBest']:
                if not args['loadPreFrame']:
                    net_image.load_state_dict(torch.load(
                        'weigths/frame/bestFrame.pth'))

                net_audio.load_state_dict(torch.load(
                    'weigths/audio/bestAudio.pth'))
            else:
                if not args['loadPreFrame']:
                    net_image.load_state_dict(torch.load(
                        'weigths/frame/latestFrame.pth'))

                net_audio.load_state_dict(torch.load(
                    'weigths/audio/latestAudio.pth'))

        for epoch in range(args['epochs']):
            # Init the path object for saving plots and images
            path = 'viz/combined/duet'
            path += "/" + args['lossFunc']
            t = time.localtime()
            path += "/" + time.strftime("%H_%M_%S", t)
            makedirs(path, False)

            evaluate_duet(nets, loader_duet_test, epoch,
                         args, crit, device, path)
            # if epoch % 1 == 0:
            #args['threshold'] += 0.1


if __name__ == '__main__':

    args = arguments.getArgs()

    main(args)
