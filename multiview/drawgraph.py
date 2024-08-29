import scipy.io as sio
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import numpy as np
def dccae_dcccae(filename1,filename2):
    data1 = np.load(filename1, allow_pickle=True)[0]
    data2 = np.load(filename2, allow_pickle=True)[0]
    dccaefeature = data1['development'].real
    dcccaefeature = data2['development'].real
    # 将featurs 中的2048维进行降维
    tsne = TSNE(n_components=1)
    Y1 = tsne.fit_transform(dccaefeature)  # 将dccafeaturs数据进行降维
    Y2 = tsne.fit_transform(dcccaefeature)  # 将dccafeaturs数据进行降维
    # 将降维之后的数据进行可视化
    for i in range(0,55):
        # plt.rcParams['axes.unicode_minus'] = False  # 显示负号\n",
        plt.figure(figsize=(6, 4))  ## 设置画布\n",
        plt.hist(Y1[48*i:48*i+24], bins=50, density=True, color='r')
        plt.hist(Y1[48*i+24:48*i+48], bins=50, density=True, color='b')
        plt.show()
        plt.figure(figsize=(6, 4))  ## 设置画布\n",
        plt.hist(Y2[48 * i:48 * i + 24], bins=50, density=True, color='r')
        plt.hist(Y2[48 * i + 24:48 * i + 48], bins=50, density=True, color='b')
        plt.show()
        print("user{}".format(i))

def savephoto(filename):
    data = np.load(filename, allow_pickle=True)[0]
    # data = sio.loadmat(filename)
    dccafeature = data['development'].real
    # y = data['devy'].ravel()
    labels = data['devlabel'].tolist()
    # dccafeature = data['cedar'][0]['development'][0]
    # labels = data['cedar'][0]['devlabel'][0]

    # 将featurs 中的2048维进行降维
    tsne = TSNE()
    Y = tsne.fit_transform(dccafeature)  # 将dccafeaturs数据进行降维
    # 将降维之后的数据进行可视化
    plt.figure(figsize=(80, 60), dpi=70)
    color = ['r' if l else 'b' for l in labels]
    x = Y[:,0]
    y = Y[:,1]
    plt.scatter(x, y, c=color, s=300)
    plt.axis('off')
    plt.savefig(filename+'.png', bbox_inches='tight')
    plt.show()
def saveuserphoto():
    filename=['2_viewdata/signet_fnocca.mat',
              '2_viewdata/signet_fmulticlass=noise.mat-cca.mat',
              '2_viewdata/signet_f_batchsize3000_epoch30_lr0.0001-2048-2048-4096-2048lamda2000-dccae.mat',
              '2_viewdata/signet_f_batchsize3000epoch50lr0.0001-2048-2048-4096-2048lamda2000droprate0.5-ddccae.mat']

    for c in range(4):
        data = sio.loadmat(filename[c])
        feature = data['cedar'][0]['development'][0]
        labels = data['cedar'][0]['devlabel'][0]
        person = data['cedar'][0]['devy'][0]
        # 将降维之后的数据进行可视化
        # 将featurs 中的2048维进行降维
        tsne = TSNE()
        Y = tsne.fit_transform(feature)  # 将dccafeaturs数据进行降维
        for user in range(1,56):
            plt.figure(figsize=(2, 2), dpi=70)
            index = np.array(np.where(person == user))[0]
            l = labels[index]
            f = Y[index]
            color = ['b' if l else 'r' for l in l[:, 0]]
            maxnum = f.max(axis=0)#将结果缩放到0-1
            minnum = f.min(axis=0)
            f[:, 0] = (f[:, 0] - minnum[0]) / (maxnum[0] - minnum[0])
            f[:, 1] = (f[:, 1] - minnum[1]) / (maxnum[1] - minnum[1])
            #根据坐标绘制图片
            for i in range(48):
                plt.scatter(f[i][0],f[i][1], c=color[i], s=50)
                # plt.text(f[i][0],f[i][1],str(int(user)-1),fontsize=30,fontweight='black', color=color[i])
            plt.axis('off')
            user = user*4+c-4
            plt.savefig("2_viewdata/"+("0"if user<10else"")+("0"if user<100else"")+str(user)+'.png', bbox_inches='tight')
            plt.show()
dccae_dcccae('2_viewdata/batchsize1000epoch50lr0.0001-4096-4096-4096-2048dccaelamda1000trainhandCEDARdcccaelamda0.0lbptrainlable0.npy','2_viewdata/batchsize1000epoch50lr0.0001-4096-4096-4096-2048dccaelamda1000trainhandCEDARdcccaelamda0.1lbptrainlable0.npy')
# saveuserphoto()
# savephoto('2_viewdata/signet_fnocca.mat')
# savephoto('2_viewdata/signet_fmulticlass=noise.mat-cca.mat')
# savephoto('2_viewdata/signet_f_batchsize3000_epoch30_lr0.0001-2048-2048-4096-2048lamda2000-dccae.mat')
# savephoto('2_viewdata/signet_f_batchsize3000epoch50lr0.0001-2048-2048-4096-2048lamda2000droprate0.5-ddccae.mat')