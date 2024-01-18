"""
------------------------------------------------------------------------
This file defines the arguments used by the network. In order to test
different set ups and hyper parameters modify the values in this file.

Last modified September 2023
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
    args['batch_size_gpu'] = 2
    args['workers'] = 2

    args['epochs'] = 10
    args['lr_audio'] = 0.0001
    args['lr_image'] = 0.001
    args['momentum'] = 0.9
    args['activation'] = 'sigmoid'
    args['threshold'] = 0.65
    args['gt_threshold'] = 0.1

    # Loss function
    args['lossFunc'] = 'cross'  # BCE and cross
    args['crossProb'] = True

    # Frame parameters
    args['FPS'] = 5

    # Audio parameters
    args['audLen'] = 65534
    args['audRate'] = 11025
    args['stftLength'] = 1023
    args['stftHop'] = 256

    args['semantic'] = True
    args['log'] = True
    args['realAudio'] = False

    # Define if weights are loaded from memory and if the weights are saved during and after training
    args['loadWeights'] = True
    args['loadBest'] = False
    args['saveWeights'] = False

    # Define which type of training procedure is used
    args['trainAudio'] = False
    args['trainFrame'] = True
    args['evalCombined'] = False # Both eval and test can't be chosen at the same time
    args['testCombined'] = False
    args['evalDuet'] = False

    # Define if frame features are also used in duet evaluation
    args['evalDuetAudioOnly'] = False

    args['freezeParams'] = False
    args['loadPreFrame'] = False
    args['useConv'] = False

    args['vizPath'] = "D:\Code\kandiProject\SemanticsProject/"
    return args
