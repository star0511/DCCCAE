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
from drawgraph import savephoto


def traindcccae(argparse):
    hand = argparse.hand
    classfile = "2_viewdata/"+hand+"hand.npy"
    data = np.load('../data/'+hand+'/CEDAR_hand_norm.npz')
    x, y, yforg = data.f.x, data.f.y, data.f.yforg
    cedar = {'development': x, 'devy': y, 'devlabel': yforg}
    data = np.load('../data/'+hand+'/UTSig_hand_norm.npz')
    x, y, yforg = data.f.x, data.f.y, data.f.yforg
    utsig = {'development': x, 'devy': y, 'devlabel': yforg}
    data = np.load('../data/'+hand+'/BHSigB_hand_norm.npz')
    x,  y, yforg = data.f.x, data.f.y, data.f.yforg
    bhsigb = {'development': x, 'devy': y, 'devlabel': yforg}
    data = np.load('../data/'+hand+'/BHSigH_hand_norm.npz')
    x, y, yforg = data.f.x, data.f.y, data.f.yforg
    bhsigh = {'development': x, 'devy': y, 'devlabel': yforg}
    data = np.load('../data/'+hand+'/SigComp2011_hand_norm.npz')
    x, y, yforg = data.f.x, data.f.y, data.f.yforg
    sigcomp2011 = {'development': x, 'devy': y, 'devlabel': yforg}
    np.save(classfile,[cedar,utsig,bhsigb,bhsigh,sigcomp2011])
    return classfile
for hand in ["hog","lbp","sift", "glcm", "daisy"]:
    for dataset in ['cedar', 'utsig', 'bhsigb', 'bhsigh','sigcomp2011']:
        parser = argparse.ArgumentParser()
        parser.add_argument('--modelclass', default="hand")
        parser.add_argument('--hand', default=hand)
        parser.add_argument('--svm-type', choices=['rbf', 'linear'], default='rbf')
        parser.add_argument('--datasetname', choices=['cedar', 'utsig', 'bhsigb', 'bhsigh','sigcomp2011'], default=dataset)
        parser.add_argument('--gen_for_train', type=int, default=1)
        parser.add_argument('--eername', default="实验5hand")
        parser.add_argument('--svm-c', type=float, default=1)
        parser.add_argument('--svm-gamma', type=float, default=2 ** -11)
        parser.add_argument('--gpuid', type=float, default=4)
        parser.add_argument('--folds', type=int, default=10)
        arguments = parser.parse_args()
        print(arguments)
        filename = traindcccae(arguments)
        # savephoto(filename)
        wd.main(arguments, filename)

