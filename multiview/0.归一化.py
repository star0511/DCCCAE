# 融合signet_f的数据
import numpy as np
import os
import scipy.io as sio
def create_dataset(dataset):

    data = np.load("../data/"+dataset+"_signet.npz")
    x, y, yforg = data.f.x,data.f.y, data.f.yforg
    maxcols = x.max()
    mincols = x.min()
    x = (x - mincols) / (maxcols - mincols)
    np.savez("../data/"+dataset+"_signet_norm",
             x=x,
             y=y,
             yforg=yforg)
    # data = np.load("../data/lbp_hog/" + dataset + "_hand.npz")
    # x, y, yforg = data.f.x, data.f.y, data.f.yforg
    # maxcols = x.max()
    # mincols = x.min()
    # x = (x - mincols) / (maxcols - mincols)
    # np.savez("../data/lbp_hog/" + dataset + "_hand_norm",
    #          x=x,
    #          y=y,
    #          yforg=yforg)

    return
for dataset in ["CEDAR","UTSig","BHSigH","BHSigB","SigComp2011"]:
    create_dataset(dataset)