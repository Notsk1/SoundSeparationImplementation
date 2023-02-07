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
    def __init__(self, csvFile, args, setType='train',  transform=None):
        self.transform = transform
        self.setType = setType
        self.dataList = []
        self.classes = ['accordion', 'acoustic_guitar', 'bagpipe', 'banjo', 'bassoon', 'cello', 'clarinet', 'congas', 'drum', 'electric_bass',
                        'erhu', 'flute', 'guzheng', 'piano', 'pipa', 'saxophone', 'trumpet', 'tuba', 'ukulele', 'violin', 'xylophone']
        # Waveform specs
        self.audLen = args['audLen']
        self.audRate = args['audRate']
        self.audSec = 1. * self.audLen/self.audRate

        # STFT specs
        self.stftLength = args['stftLength']
        self.stftHop = args['stftHop']

        # Frame specs
        self.fps = args['FPS']
        self.imgSize = (500, 500)

        setCutOff = 0
        for row in csv.reader(open(csvFile, 'r'), delimiter=','):
            if len(row) < 2:
                continue
            # if setCutOff > 10:
            #    break
            # [0] audio path, [1] frame path, [2] number of frames
            self.dataList.append(row)
            setCutOff += 1

    def __len__(self):
        return len(self.dataList)

    def __getitem__(self, idx):
        inputAmp, inputPhase, labels, center = self._getaudio_(idx)
        frame, image = self._getframe_(idx, center)
        return inputAmp, inputPhase, torch.tensor(labels), frame, torch.tensor(image)

    def _getframe_(self, idx, center):
        # Get path
        path = self.dataList[idx][1]
        #numOfFrames = self.dataList[idx][2]

        frameIdx = int((center/self.audRate)*self.fps)

        imagePath = path + '\{}.jpg'.format(str(frameIdx).zfill(6))

        image = Image.open(imagePath).convert('RGB')
        self._img_transform()
        transImg = self.img_transform(image)

        image = image.resize((300, 300))

        return transImg, np.array(image)

    def _img_transform(self):
        mean = [0.485, 0.456, 0.406]  # Needed for the resnet18 architecture
        std = [0.229, 0.224, 0.225]

        if self.setType == 'train':
            self.img_transform = transforms.Compose([
                transforms.Resize(self.imgSize),
                transforms.RandomCrop(self.imgSize),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean, std)])
        else:
            self.img_transform = transforms.Compose([
                transforms.Resize(self.imgSize),
                transforms.ToTensor(),
                transforms.Normalize(mean, std)])

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
        inputAmp = inputAmp[None, ...]

        return inputAmp, inputPhase, labels, center

    def _stft_(self, audio):
        spec = librosa.stft(
            audio, n_fft=self.stftLength, hop_length=self.stftHop)
        amp = np.abs(spec)
        phase = np.angle(spec)
        return torch.from_numpy(amp), torch.from_numpy(phase)

    def _processAudio_(self, audio_raw, time=0):

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
