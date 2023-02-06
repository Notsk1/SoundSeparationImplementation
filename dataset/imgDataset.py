"""
------------------------------------------------------------------------
Image dataset class that is used by the dataloader to train an test the
image network.

Last modified May 2022
Author Petteri Nuotiomaa
------------------------------------------------------------------------
"""
from torchvision import transforms
import torch.utils.data as torchdata
from torch import tensor
import csv
import numpy as np
from PIL import Image
from random import randint


class ImgDataset(torchdata.Dataset):
    def __init__(self, csvFile, setType = 'train',  transform = None):
        self.imgSize = (500,500)
        self.transform = transform
        self.setType = setType
        self.dataList = []
        self.classes = ['accordion','acoustic_guitar','bagpipe','banjo','bassoon','cello','clarinet','congas','drum','electric_bass',
                        'erhu','flute','guzheng','piano','pipa','saxophone','trumpet','tuba','ukulele','violin','xylophone']
        setCutOff = 0
        for row in csv.reader(open(csvFile, 'r'), delimiter=','):
            if len(row) < 2:
                continue
            #if setCutOff > 10:
            #    break
            self.dataList.append(row[1:]) #Don't take the audio
            setCutOff += 1
    
    def __len__(self):
        return len(self.dataList)
    
    def _img_transform(self):
        mean = [0.485, 0.456, 0.406] #Needed for the resnet18 architecture
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
        
    
    def __getitem__(self, idx):

        path = self.dataList[idx][0]
        numOfFrames = self.dataList[idx][1]
        label = path.split("\\")[1]


        rint = randint(1, int(numOfFrames)-1)

        imagePath = path + '\{}.jpg'.format(str(rint).zfill(6))
        image = Image.open(imagePath).convert('RGB')
        self._img_transform()
        transImg = self.img_transform(image)

        #sample = {'image' : transImg, 'label':label}
        return transImg, self.classes.index(label)
    
