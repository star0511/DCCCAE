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
    data = np.load("../data/CEDAR.npz".replace(".npz","maxsize.npz"))
    x0,_,_ = data.f.x,data.f.y,data.f.yforg
    sift_features = sift_feature(x0)
    print(sift_features)
    bow = bow_init(sift_features)
    for file in filelist:
        data = np.load(file.replace(".npz", "maxsize.npz"))
        x0,y,yforg = data.f.x, data.f.y, data.f.yforg
        bowfeatures = bow_feature(bow,x0)
        fullbowlist = []
        for i in range(len(bowfeatures)):
            if bowfeatures[i] is not None:
                fullbowlist.append(bowfeatures[i][0].tolist())
            else:
                fullbowlist.append(np.zeros((128)).tolist())
        fullbowlist = np.array(fullbowlist)
        print("sift提取完毕")
        np.savez(file.replace(".npz","_hand"),
                 x=fullbowlist,
                 y=y,
                 yforg=yforg)

        x = fullbowlist
        maxcols = x.max()
        mincols = x.min()
        x = (x - mincols) / (maxcols - mincols)
        np.savez(file.replace(".npz", "_hand_norm").replace("data", "data/sift").replace("maxsize", ""),
                 x=x,
                 y=y,
                 yforg=yforg)

# 提取图像的sift特征
def sift_feature(image_list):
    feature_sift_list = []  # sift特征向量列表
    p = Pool(100)
    for i in range(len(image_list)):
        # 获取sift特征，kp为关键点信息，des为关键点特征矩阵形式
        feature_sift_list.append(p.apply_async(psift, args=(image_list[i][0], i, len(image_list))))
    p.close()
    p.join()
    for i in tqdm(range(len(feature_sift_list))):
        feature_sift_list[i]=feature_sift_list[i]._value
    return np.array(feature_sift_list)
def psift(img,i,lenimg):
    print("\rsift[{}/{}]".format(i,lenimg),end=" ")
    sift = cv.SIFT_create()
    kp, des = sift.detectAndCompute(img, None)
    return des
# 初始化BOW训练器
def bow_init(feature_sift_list):
    # 创建BOW训练器，指定k-means参数k   把处理好的特征数据全部合并，利用聚类把特征词分为若干类，此若干类的数目由自己设定，每一类相当于一个视觉词汇
    bow_kmeans_trainer = cv.BOWKMeansTrainer(128)

    for feature_sift in feature_sift_list:
        bow_kmeans_trainer.add(feature_sift)

    # 进行k-means聚类，返回词汇字典 也就是聚类中心
    voc = bow_kmeans_trainer.cluster()

    # 输出词汇字典
    print(voc)
    print(type(voc), voc.shape)

    # FLANN匹配
    # algorithm用来指定匹配所使用的算法，可以选择的有LinearIndex、KTreeIndex、KMeansIndex、CompositeIndex和AutotuneInde
    # 这里选择的是KTreeIndex(使用kd树实现最近邻搜索)
    flann_params = dict(algorithm=1, tree=5)
    flann = cv.FlannBasedMatcher(flann_params, {})

    print(flann)

    # 初始化bow提取器(设置词汇字典),用于提取每一张图像的BOW特征描述
    sift = cv.SIFT_create()
    bow_img_descriptor_extractor = cv.BOWImgDescriptorExtractor(sift, flann)
    bow_img_descriptor_extractor.setVocabulary(voc)

    print(bow_img_descriptor_extractor)

    return bow_img_descriptor_extractor
# 提取BOW特征
def bow_feature(bow_img_descriptor_extractor, image_list):
    # 分别对每个图片提取BOW特征，获得BOW特征列表
    feature_bow_list = []
    sift = cv.SIFT_create()
    for i in tqdm(range(len(image_list))):
        feature_bow_list.append(bow_img_descriptor_extractor.compute(image_list[i][0],sift.detect(image_list[i][0])))
    return feature_bow_list
# 填充缺失值
def fill_missing(feature):
    feature_df = pd.DataFrame(feature)  # 转为DataFrame格式，才能使用fillna函数
    feature_df_fill = feature_df.fillna(0)  # 将缺失值部分填充0
    # 返回array格式
    return feature_df_fill


# 标准化
def normalize(feature):
    scaler = StandardScaler()
    scaler.fit(feature)
    feature_normal = scaler.transform(feature)

    return feature_normal


# 降维 使用PCA(Principal Component Analysis)主成分分析
def dimensionalityReduction(feature, n=100, is_whiten=False, is_show=False):
    estimator = PCA(n_components=n, whiten=is_whiten)
    pca_feature = estimator.fit_transform(feature)
    # 输出降维后的各主成分的方差值占总方差值的比例的累加
    sum = 0
    for ratio in estimator.explained_variance_ratio_:
        sum += ratio
        if is_show:
            print(sum)
    print('降维后特征矩阵shape为:', pca_feature.shape)
    # print('主成分比例为:', sum)
    return pca_feature


def sift(img,i,lenimg):
    print("{}/{}".format(i,lenimg))
    sift = cv.SIFT_create()
    keypoints = sift.detect(img, None)
    keypoints, descriptors = sift.compute(img, keypoints)
    descriptors = descriptors.flatten()
    lendesc = len(descriptors)
    dim = 40000
    if lendesc>=dim:
        descriptors = descriptors[:dim]
    else:
        descriptors = np.concatenate((descriptors,np.zeros((dim-lendesc))))
    return descriptors
def pcato2048(filename):
    data = np.load(filename.replace(".npz","_2view.npz"))
    x, y, yforg = data.f.x, data.f.y, data.f.yforg
    pca = PCA(n_components=2048)  # 实例化
    pca = pca.fit(x)  # 拟合模型
    X_dr = pca.transform(x)  # 获取新矩阵
    np.savez(filename.replace(".npz", "_pca"),
             x=X_dr,
             y=y,
             yforg=yforg)
def pcato2048hog(filename):
    data = np.load(filename)
    x, y, yforg = data.f.x, data.f.y, data.f.yforg
    pca = PCA(n_components=2048)  # 实例化
    pca = pca.fit(x)  # 拟合模型
    X_dr = pca.transform(x)  # 获取新矩阵
    np.savez(filename.replace(".npz", ""),
             x=X_dr,
             y=y,
             yforg=yforg)
def tsneto2048(filename):
    from sklearn.manifold import TSNE
    data = np.load(filename.replace(".npz", "_2view.npz"))
    x, y, yforg = data.f.x, data.f.y, data.f.yforg

    tsne = TSNE(n_components=2048)
    x = tsne.fit_transform(x)  # 将dccafeaturs数据进行降维
    np.savez(filename.replace(".npz", "_tsne"),
             x=x,
             y=y,
             yforg=yforg)
if __name__ == '__main__':

    # filelist = ["../data/SigComp2011.npz","../data/CEDAR.npz","../data/UTSig.npz","../data/BHSigH.npz","../data/BHSigB.npz"]
    # parser = argparse.ArgumentParser()
    # parser.add_argument('--file', choices=filelist,default="../data/CEDAR.npz")
    # args = parser.parse_args()
    # for file in filelist:
    exthogfeeature()
    # pcato2048hog("../data/CEDAR_hog.npz")
    # extsiftfeeature(args.file)
    # pcato2048(args.file)
    # tsneto2048(args.file)