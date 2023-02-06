"""
------------------------------------------------------------------------
Duet dataset that will be used to evaluate the duet performance

Last modified May 2022
Author Petteri Nuotiomaa
------------------------------------------------------------------------
"""
from posixpath import split
from torchvision import transforms
import torch.utils.data as torchdata
import torch
import csv
import librosa
import numpy as np
from PIL import Image
from random import randint
import os


class DuetDataset(torchdata.Dataset):
    def __init__(self, csvFile, args, setType = 'train',  transform = None):
        self.transform = transform
        self.setType = setType
        self.dataList = []
        self.classes = ['accordion','acoustic_guitar','bagpipe','banjo','bassoon','cello','clarinet','congas','drum','electric_bass',
                        'erhu','flute','guzheng','piano','pipa','saxophone','trumpet','tuba','ukulele','violin','xylophone']
        # Waveform specs
        self.audLen = args['audLen']
        self.audRate = args['audRate']
        self.audSec = 1. * self.audLen/self.audRate
        
        # STFT specs
        self.stftLength = args['stftLength']
        self.stftHop = args['stftHop']

        setCutOff = 0
        for row in csv.reader(open(csvFile, 'r'), delimiter=','):
            if len(row) < 2:
                continue
            #if setCutOff > 10:
            #    break
            self.dataList.append(row) # [0] audio path, [1] frame path, [2] number of frames
            setCutOff += 1
    
    def __len__(self):
        return len(self.dataList)     
    
    def __getitem__(self, idx):
        inputAmp, inputPhase, labels, center = self._getaudio_(idx)
        return inputAmp, inputPhase, torch.tensor(labels)

    # Audio modules
    def _getaudio_(self, idx):
        # Get audio path and the label
        path = self.dataList[idx][0]
        labels = []
        label = os.path.normpath(path).split(os.sep)[3]
        label = label.split()
        labels.append(self.classes.index(label[0]))
        labels.append(self.classes.index(label[1]))


        # Load original audio and process it
        inputAudio, rate = librosa.load(path, sr=None, mono=True) 
        inputAudio_proc, center = self._processAudio_(inputAudio)

        inputAmp, inputPhase = self._stft_(inputAudio_proc)
        inputAmp = inputAmp[None,...]

        return inputAmp, inputPhase, labels, center
    
    def _stft_(self, audio):
        spec = librosa.stft(
            audio, n_fft=self.stftLength, hop_length=self.stftHop)
        amp = np.abs(spec)
        phase = np.angle(spec)
        return torch.from_numpy(amp), torch.from_numpy(phase)

    def _processAudio_(self, audio_raw, time = 0):

        if audio_raw.shape[0] < self.audRate * self.audSec:
            n = int(self.audRate * self.audSec / audio_raw.shape[0]) + 1
            audio_raw = np.tile(audio_raw, n)
        
        len_raw = audio_raw.shape[0]

        if time == 0:
            # Sample random point from the audio for the interference audio
            center = randint(self.audLen + 1, audio_raw.shape[0]-self.audLen)
        else:
            center = int(time * self.audRate)
        
        # crop N seconds
        start = max(0, center - self.audLen // 2)
        end = min(len_raw, center + self.audLen // 2)
        audio = audio_raw[start:end]
        return audio, center
        