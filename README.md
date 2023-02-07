# Multi Modal Sound Source Separation 

NOTE: Work in Progress. Some of the instructions and descriptions of the code in this readme can be outdated. 

This repository contains a machine learning model used for sound source separation. Model uses Visual and audio features to separate different audio sources from single duet video. Model has similar structure to the model introduced in paper https://arxiv.org/pdf/1804.03160.pdf.  

This project was done as a part of a bachelor's thesis which studies the modern approach for sound source separation. 

## Getting Started 

Following the instructions below allows user to run the model on wanted data. Data used in the study was the dataset downloadable from 
https://github.com/roudimit/MUSIC_dataset. 

### Prerequisites 

- Anaconda or another virtual environment manager. 
- Some video data which has a singular music/sound source (preferable the same data used in the study) 
- Python libraries in requirements.txt file 

### Installing 

Clone the repository to your local computer and then set up desired settings from arguments.py.  

### Training the model 

Before training the model, data needs to be stored and the path to data defined. Train the model with command "python main.py" and setting "trainCombared" has to be True. If only visual network training desired, "trainCombared" must be False and "trainFrame" must be True. 

### Evaluating the model 

After the model has been trained, you can evaluate the model with setting "evalCombined" being True and then running the main script. To evaluate the performance on duet videos, run script with setting "evalDuet" being True. 

## Model Output 

Model outputs two spectrograms that can be formed back to a separated audio track. For training, two videos are combined and their mixture is used to train the model and as an output, the two videos' audios are separated from the mixture. As for evaluation on duet audio, the input is a video with two instruments playing and the output is each instrument track separately. 

### Training Output

Example training sample output:

Input audio mixture spectogram:

![InputSpec](https://user-images.githubusercontent.com/66205961/217004727-c2b5171b-7703-4bfd-a98f-e006d14e86ec.png)

Instrument 1 Frame: 

![frame0](https://user-images.githubusercontent.com/66205961/216768730-4493b26f-9d1a-45ed-8bb8-8c0f8f790fef.png) 

Instrument 1 Ground Truth Audio Spectrogram Binary Mask: 

![GTMask0](https://user-images.githubusercontent.com/66205961/216768765-0267bf45-459e-443f-9e1e-ca668e831cd4.png) 

Instrument 1 Predicted Audio Spectrogram Binary Mask: 

![outputMask0](https://user-images.githubusercontent.com/66205961/216768912-f9cda656-a610-4ef0-aa15-91f3bf5ea549.png) 

Instrument 2 Frame: 

![frame1](https://user-images.githubusercontent.com/66205961/216768802-1fff3aab-b674-4599-bbfb-1967e8d10e04.png) 

Instrument 2 Ground Truth Audio Spectrogram Binary Mask: 

![GTMask1](https://user-images.githubusercontent.com/66205961/216768845-186f9157-fcd5-4378-aceb-f1bdb60fbb5f.png) 

Instrument 2 Predicted Audio Spectrogram Binary Mask: 

![outputMask1](https://user-images.githubusercontent.com/66205961/216768848-7909e83e-5aab-450a-b251-3f9f55be0ad7.png) 

### Duet Output 

## Authors 

Petteri Nuotiomaa Github:Notsk1

## Acknowledgments 

As mentioned before, the techniques used were inspired from many sources, but most prominently from the paper https://arxiv.org/pdf/1804.03160.pdf.
