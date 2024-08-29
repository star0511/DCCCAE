# DCCA训练
import numpy as np
import os
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # or any {'0', '1', '2'}
import tensorflow.compat.v1 as tf

import requests
# tf.logging.set_verbosity(tf.logging.ERROR)
tf.disable_v2_behavior()
import scipy.io as sio
import wd
import argparse
import DCCCAE as dcccae
from CCA import linCCA
from myreadinput import read_dataset,read_dataset_train
from drawgraph import savephoto
def traindcccae():
    classfile="2_viewdata/signet.npy"
    # if os.path.isfile(classfile):
        # print("Job is already finished!")
        # return classfile
    data = np.load('../data/CEDAR_signet_norm.npz')
    x, y, yforg = data.f.x, data.f.y, data.f.yforg
    cedar = {'development': x, 'devy': y, 'devlabel': yforg}
    data = np.load('../data/UTSig_signet_norm.npz')
    x, y, yforg = data.f.x, data.f.y, data.f.yforg
    utsig = {'development': x, 'devy': y, 'devlabel': yforg}
    data = np.load('../data/BHSigB_signet_norm.npz')
    x,  y, yforg = data.f.x, data.f.y, data.f.yforg
    bhsigb = {'development': x, 'devy': y, 'devlabel': yforg}
    data = np.load('../data/BHSigH_signet_norm.npz')
    x, y, yforg = data.f.x, data.f.y, data.f.yforg
    bhsigh = {'development': x, 'devy': y, 'devlabel': yforg}
    data = np.load('../data/SigComp2011_signet_norm.npz')
    x, y, yforg = data.f.x, data.f.y, data.f.yforg
    sigcomp2011 = {'development': x, 'devy': y, 'devlabel': yforg}
    np.save(classfile,[cedar,utsig,bhsigb,bhsigh,sigcomp2011])
    return classfile
for dataset in ['cedar','utsig','bhsigb','bhsigh','sigcomp2011']:
    parser = argparse.ArgumentParser()
    parser.add_argument('--modelclass', default="dcccae")
    parser.add_argument('--svm-type', choices=['rbf', 'linear'], default='rbf')
    parser.add_argument('--datasetname', choices=['cedar','utsig','bhsigb','bhsigh','sigcomp2011'], default=dataset)
    parser.add_argument('--gen_for_train', type=int, default=1)
    parser.add_argument('--eername', default="实验5signet")
    parser.add_argument('--gpuid', type=float, default=3)
    parser.add_argument('--svm-c', type=float, default=1)
    parser.add_argument('--svm-gamma', type=float, default=2 ** -11)
    parser.add_argument('--folds', type=int, default=10)
    arguments = parser.parse_args()
    print(arguments)
    filename = traindcccae()
    # savephoto(filename)
    wd.main(arguments,filename)

