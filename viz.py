"""
------------------------------------------------------------------------
This files defines a class that helps at visualizing the outputs of the
system.

Last modified May 2022
Author Petteri Nuotiomaa
------------------------------------------------------------------------
"""
import os
import torch
import matplotlib
import matplotlib.pyplot as plt
import scipy.io.wavfile as wavfile
import numpy as np
from utils import makedirs
matplotlib.use('Agg')


def save_and_plot_metrics(path, trainLosses, evalLosses, evalSDRs, evalSIRs, evalSARs):
    """Plots Losses, SDR, SAR and SIR to specified path"""
    x = np.linspace(0, len(evalLosses), len(evalLosses))
    # Losses:
    plt.figure()
    if len(trainLosses) > 0:
        plt.plot(x, trainLosses, label='Train loss')
    plt.plot(x, evalLosses, label='Eval Loss')
    plt.legend()
    plt.savefig(path + "/Losses.png")
    plt.close()
    # Metrics
    plt.figure()
    plt.plot(x, evalSDRs, label='Eval SDR')
    plt.plot(x, evalSARs, label='Eval SAR')
    plt.plot(x, evalSIRs, label='Eval SIR')
    plt.legend()
    plt.savefig(path + "/Metrics.png")


def output_visuals_duet(viz_rows, inputAmps, output, labels, inputAudio, outputAudios, path, filePath, image, args):
    """Saves the duet visuals and audios to correct folders and stores the file locations for html visualization"""
    makedirs(path + filePath, False)
    row_elements = []

    # Input audio
    inputAudioFile = path + filePath + "/InputAudio.wav"
    wavfile.write(inputAudioFile, args['audRate'], inputAudio)

    # Input spectrogram
    inputSpecFile = path + filePath + "/InputSpec.png"
    plt.figure()
    plt.imshow(torch.log(torch.abs(inputAmps[0][0])))
    plt.savefig(inputSpecFile)
    plt.close()

    # Stores the file paths for the html visualization
    row_elements += [{'text': path}, {'image': args['vizPath'] +
                                      inputSpecFile, 'audio': args['vizPath'] + inputAudioFile}]

    # Input image
    inputFrameFile = path + filePath + "/frame{}.png".format(0)
    plt.figure()
    plt.imshow(image)
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
    inputAudioFile = path + filePath + "/InputAudio.wav"
    wavfile.write(inputAudioFile, args['audRate'], inputAudio)

    # Input spectrogram
    inputSpecFile = path + filePath + "/InputSpec.png"
    plt.figure()
    plt.imshow(torch.log(torch.abs(inputAmps[0][0])))
    plt.savefig(inputSpecFile)
    plt.close()

    row_elements += [{'text': path}, {'image': args['vizPath'] +
                                      inputSpecFile, 'audio': args['vizPath'] + inputAudioFile}]

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


class HTMLVisualizer():
    def __init__(self, fn_html):
        self.fn_html = fn_html
        self.content = '<table>'
        self.content += '<style> table, th, td {border: 1px solid black;} </style>'

    def add_header(self, elements):
        self.content += '<tr>'
        for element in elements:
            self.content += '<th>{}</th>'.format(element)
        self.content += '</tr>'

    def add_rows(self, rows):
        for row in rows:
            self.add_row(row)

    def add_row(self, elements):
        self.content += '<tr>'

        # a list of cells
        for element in elements:
            self.content += '<td>'

            # fill a cell
            for key, val in element.items():
                if key == 'text':
                    self.content += val
                elif key == 'image':
                    self.content += '<img src="{}" style="max-height:256px;max-width:256px;">'.format(
                        val)
                elif key == 'audio':
                    self.content += '<audio controls><source src="{}"></audio>'.format(
                        val)
                elif key == 'video':
                    self.content += '<video src="{}" controls="controls" style="max-height:256px;max-width:256px;">'.format(
                        val)
            self.content += '</td>'

        self.content += '</tr>'

    def write_html(self):
        self.content += '</table>'
        with open(self.fn_html, 'w') as f:
            f.write(self.content)
