"""
------------------------------------------------------------------------
This file defines the arguments used by the network. In order to test
different set ups and hyper parameters modify the values in this file.

Last modified February 2023
Author Petteri Nuotiomaa
------------------------------------------------------------------------
"""


def getArgs():
    args = {}

    args['csvTrain'] = '../data/train.csv'
    args['csvVal'] = '../data/val.csv'
    args['csvDuetTrain'] = '../data/trainDuet.csv'
    args['csvDuetVal'] = '../data/valDuet.csv'

    args['num_gpus'] = 1
    args['batch_size_gpu'] = 4  # Res-net only works when batch_size > 1
    args['workers'] = 4

    args['epochs'] = 20
    args['lr_audio'] = 0.0005
    args['lr_image'] = 0.001
    args['momentum'] = 0.9
    args['activation'] = 'sigmoid'
    args['threshold'] = 0.85

    # Loss function
    args['lossFunc'] = 'cross'  # BCE and cross
    args['crossProb'] = True

    # Frame parameters
    args['FPS'] = 5

    # Audio parameters
    args['audLen'] = 65535
    args['audRate'] = 11025
    args['stftLength'] = 1023
    args['stftHop'] = 256

    args['semantic'] = True
    args['log'] = True
    args['realAudio'] = False

    # Define if weights are loaded from memory and if the weights are saved during and after training
    args['loadWeights'] = True
    args['loadBest'] = False
    args['saveWeights'] = True

    # Define which type of training procedure is used
    args['trainAudio'] = False
    args['trainFrame'] = False
    args['evalCombined'] = False
    args['evalDuet'] = True

    # Define if frame features are also used in duet evaluation
    args['evalDuetAudioOnly'] = False

    args['freezeParams'] = False
    args['loadPreFrame'] = False
    args['useConv'] = False

    args['vizPath'] = "D:\Code\kandiProject\SemanticsProject/"
    return args
