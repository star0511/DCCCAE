import pickle

import torch
import numpy as np
import argparse
from skimage.feature import *
import cv2 as cv
from tqdm import tqdm
from multiprocessing import Pool, Manager
import pandas as pd
import time
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
def exthogfeeature():
    filelist = ["../data/SigComp2011.npz", "../data/CEDAR.npz", "../data/UTSig.npz", "../data/BHSigH.npz",
                "../data/BHSigB.npz"]
    pca = initpca()
    for file in filelist:
        pca_feature_hog = []
        data = np.load(file)
        x, y, yforg = data.f.x, data.f.y, data.f.yforg
        hog_features = []
        p=Pool(100)
        for k in x:
            hog_features.append(p.apply_async(phog,args=(k)))
        p.close()
        p.join()
        for hog_feature in hog_features:
            hogfeature = hog_feature._value
            pca_feature_hog.append(pca.transform([hogfeature])[0])

        print("hog提取完毕")
        x = np.array(pca_feature_hog)
        maxcols = x.max()
        mincols = x.min()
        x = (x - mincols) / (maxcols - mincols)
        np.savez(file.replace(".npz", "_hand_norm").replace("data", "data/hog").replace("maxsize", ""),
                 x=x,
                 y=y,
                 yforg=yforg)
def initpca():
    data = np.load("../data/CEDAR.npz")
    hog_features = []
    x = data.f.x
    for k in tqdm(x):
        hog_features.append(phog(k[0]))
    pca = PCA(n_components=256)  # 实例化
    pca = pca.fit(hog_features)  # 拟合模型
    return pca
def phog(img):

    hogfeature = hog(img)
    return hogfeature

if __name__ == '__main__':

    # filelist = ["../data/SigComp2011.npz","../data/CEDAR.npz","../data/UTSig.npz","../data/BHSigH.npz","../data/BHSigB.npz"]
    # parser = argparse.ArgumentParser()
    # parser.add_argument('--file', choices=filelist,default="../data/CEDAR.npz")
    # args = parser.parse_args()
    # for file in filelist:
    exthogfeeature()
    # pcato2048hog("../data/CEDAR_hog.npz")
    # extkazefeeature(args.file)
    # pcato2048(args.file)
    # tsneto2048(args.file)