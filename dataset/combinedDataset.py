"""
------------------------------------------------------------------------
Definition of the combined dataset class that is used by the dataloader to
test the system.

Last modified September 2022
Author Petteri Nuotiomaa
------------------------------------------------------------------------
"""
import torch.utils.data as torchdata
import numpy as np

import csv
import os
import librosa
import torch
import random


from PIL import Image
from random import randint
from torchvision import transforms


class CombinedDataset(torchdata.Dataset):
    def __init__(self, csvFile, args, setType='train'):
        super().__init__()
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

        classFiles = {}
        classAmount = {}
        setCutOff = 0
        for row in csv.reader(open(csvFile, 'r'), delimiter=','):
            if len(row) < 2:
                continue
            #if setCutOff > 10:
            #    break
            # [0] audio path, [1] frame path, [2] number of frames
            self.dataList.append(row)

            if self.setType == 'train':
                label = os.path.normpath(row[0]).split(os.sep)[3]
                if label not in classFiles:
                    classAmount[label] = 1
                    classFiles[label] = [row]
                else:
                    classAmount[label] = classAmount[label] + 1
                    classFiles[label].append(row)
            setCutOff += 1
        
        if self.setType == 'val':
            self.dataList = self.dataList[0:(len(self.dataList)//2)]
        if self.setType == 'test':
            self.dataList = self.dataList[(len(self.dataList)//2):]


        # Add more samples to instruments that are less represented
        if self.setType == 'train':
            classMax = classAmount[max(classAmount, key=classAmount.get)]
            for key, value in classAmount.items():
                maxIdx = value
                while value < classMax:
                    randPath = classFiles[key][randint(0, maxIdx-1)]
                    self.dataList.append(randPath)
                    value += 1

        if self.setType == 'train':
            self.dataList *= 25
            random.shuffle(self.dataList)

    def __len__(self):
        return len(self.dataList)

    def __getitem__(self, idx):

        ampMix, inputAmp, inputPhase, labels, centers, idxs, full_audios, inputAudio_proc = self._getaudio_(
            idx)
        frames = []
        images = []
        for i in range(len(centers)):
            frame, image = self._getframe_(idxs[i], centers[i])
            if i == 0:
                frames = frame[None, ...]
            else:
                frame = frame[None, ...]
                frames = torch.cat([frames, frame], 0)
            images.append(image)

        images = np.array(images)
        return ampMix, inputAmp, inputPhase, torch.tensor(labels), frames, torch.tensor(images), full_audios, inputAudio_proc

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
        labels.append(self.classes.index(label))

        # Get random amount of music to add to each input
        rint = 1  # randint(1, 3)
        # Define center list for frame retrival
        centers = []
        idxs = []
        idxs.append(idx)

        # Load original audio and process it
        inputAudio, rate = librosa.load(path, sr=None, mono=True)
        inputAudio_proc, center = self._processAudio_(inputAudio)

        amp, phase = self._stft_(inputAudio_proc)

        centers.append(center)

        # Define amplitude mix
        tupleSize = tuple(amp.size())
        size = (21, tupleSize[0], tupleSize[1])  # +1 for the actual audio
        ampMix = torch.zeros(size)
        ampMix[self.classes.index(label), :, :] = amp

        # Store audio for SDR, SAR, SIR
        full_audios = np.zeros((2, inputAudio_proc.shape[0]))
        full_audios[0, :] = inputAudio_proc


        # Use each one of the spectograms magnitude information
        # so that every channel of the U-net outputs one instrument (adds up to 21 channels)
        i = 0
        usedLabels = [label]
        while i < rint:
            sampleIdx = randint(0, self.__len__()-1)
            pathRand = self.dataList[sampleIdx][0]
            sampleLabel = os.path.normpath(pathRand).split(os.sep)[3]
            if sampleLabel not in usedLabels:
                labels.append(self.classes.index(sampleLabel))

                # Get the audio and process it
                audio_raw, rate = librosa.load(pathRand, sr=None, mono=True)
                audio_proc, centerRand = self._processAudio_(audio_raw)

                centers.append(centerRand)
                idxs.append(sampleIdx)

                # Sum all audio clips together
                inputAudio_proc = inputAudio_proc + audio_proc

                amp, phase = self._stft_(audio_proc)

                ampMix[self.classes.index(sampleLabel), :, :] = amp

                full_audios[1, :] = audio_proc

                # Limit to one instance of the same instrument per audiomix
                usedLabels.append(sampleLabel)
            else:
                i += -1
            i += 1

        inputAmp, inputPhase = self._stft_(inputAudio_proc)
        inputAmp = inputAmp[None, ...]

        return ampMix, inputAmp, inputPhase, labels, centers, idxs, full_audios, inputAudio_proc

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
