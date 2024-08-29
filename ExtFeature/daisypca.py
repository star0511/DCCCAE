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
    # filelist = ["../data/BHSigH.npz",
    #             "../data/BHSigB.npz"]
    # filelist = ["../data/SigComp2011maxsize.npz", "../data/CEDARmaxsize.npz", "../data/UTSigmaxsize.npz",
    #             "../data/BHSigHmaxsize.npz",
    #             "../data/BHSigBmaxsize.npz"]
    # filelist = ["../data/UTSigmaxsize.npz",
    #             "../data/BHSigHmaxsize.npz",
    #             "../data/BHSigBmaxsize.npz"]
    for file in filelist:
        pca_feature_lbp = []
        data = np.load(file)
        x, y, yforg = data.f.x, data.f.y, data.f.yforg
        lbp_features = []
        p=Pool(100)
        i = 0
        l = len(x)
        for k in x:
            lbp_features.append(p.apply_async(plbp,args=(k[0],i,l)))
            i = i + 1
        p.close()
        p.join()
        for lbp_feature in lbp_features:
            lbpfeature = lbp_feature._value
            pca_feature_lbp.append(lbpfeature)
        print("lbp提取完毕")

        pca = PCA(n_components=256)  # 实例化
        pca = pca.fit(pca_feature_lbp[:500])  # 拟合模型
        print("拟合完毕")
        # pca_feature_lbp = pca.transform(pca_feature_lbp)
        pca_feature_lbplist=[]
        lendata=len(pca_feature_lbp)
        for batchid in tqdm(range(int(lendata/500)+1)):
            t = pca.transform(pca_feature_lbp[batchid*500:min(batchid*500+500,lendata)])
            for item in t:
                pca_feature_lbplist.append(item)
        print("特征提取完毕")
        pca_feature_lbp = np.array(pca_feature_lbplist)
        # np.savez(file.replace(".npz","_hand").replace("data","data/lbp"),
        #          x=pca_feature_lbp,
        #          y=y,
        #          yforg=yforg)

        x = pca_feature_lbp
        maxcols = x.max()
        mincols = x.min()
        x = (x - mincols) / (maxcols - mincols)
        np.savez(file.replace(".npz","_hand_norm").replace("data","data/daisy").replace("maxsize",""),
                 x=x,
                 y=y,
                 yforg=yforg)
def plbp(img,i,l):
    # lbpfeatures=[]
    # lbpfeatures.append(local_binary_pattern(img, 8, 1).flatten())
    # lbpfeatures.append(local_binary_pattern(img, 19, 2).flatten())
    # lbpfeatures.append(local_binary_pattern(img, 24, 3).flatten())
    # lbpfeatures.append(local_binary_pattern(img, 32, 4).flatten())
    print("\r{}/{}".format(i,l),end="")
    # glcm = daisy(img)
    # print(glcm.shape)
    glcm = daijsy(img).flatten()
    return glcm
# 填充缺失值
def fill_missing(feature):
    feature_df = pd.DataFrame(feature)  # 转为DataFrame格式，才能使用fillna函数
    feature_df_fill = feature_df.fillna(0)  # 将缺失值部分填充0
    # 返回array格式
    return feature_df_fill

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