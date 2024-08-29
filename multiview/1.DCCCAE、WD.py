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
def traindcccae(arguments):
    # Some other configurations parameters for mnist.
    learning_rate = arguments.learning_rate
    l2_penalty = 0.0001
    rcov1 = 0.0001
    rcov2 = 0.0001
    classfile="2_viewdata/batchsize"+str(arguments.batch_size)+"epoch"+str(arguments.epoch)+"lr"+str(learning_rate) + "-"+str(arguments.net_hidden_layer[0])+"-"+str(arguments.net_hidden_layer[1])+"-"+str(arguments.net_hidden_layer[2])+"-"+str(arguments.net_hidden_layer[3])+"dccaelamda"+str(arguments.dccaelamda)+"trainhand"+arguments.trainhand+"dcccaelamda"+str(arguments.dcccaelamda)+arguments.hand+"trainlable"+str(arguments.trainlable)+'.npy'
    if os.path.isfile(classfile):
        print("Job is already finished!")
        return classfile
    # Handle multiple gpu issues.
    # tf.reset_default_graph()
    #
    os.environ["CUDA_VISIBLE_DEVICES"] = arguments.gpuid
    config = tf.ConfigProto(allow_soft_placement=True,log_device_placement=False)
    config.gpu_options.allow_growth = True
    config.gpu_options.per_process_gpu_memory_fraction=0.4 # （尽量用这种方式）设置最大占有GPU不超过显存的25%
    sess = tf.Session(config=config)
    
    # Set random seeds.
    np.random.seed(100)
    tf.set_random_seed(100)

    handclass=arguments.hand
    # Define network architectures.
    trainData, tuneData, testData = read_dataset_train("../data/" + arguments.trainhand + "_signet_norm.npz",
                                                       "../data/" + handclass + "/" + arguments.trainhand + "_hand_norm.npz",arguments.trainlable)

    network_architecture=dict(
        n_input1=2048, # feature1 data input (shape: 2048)
        n_input2=trainData.images2.shape[1], # feature2 data input (shape: 2048)
        n_z=arguments.net_hidden_layer[-1],  # Dimensionality of shared latent space
        F_hidden_widths=arguments.net_hidden_layer,
        F_hidden_activations=[tf.nn.tanh, tf.nn.tanh, tf.nn.tanh, None],
        G_hidden_widths=arguments.net_hidden_layer,
        G_hidden_activations=[tf.nn.tanh, tf.nn.tanh, tf.nn.tanh, None]
        )
    # First, build the model.
    model=dcccae.DCCCAE(classfile,network_architecture, rcov1, rcov2, learning_rate, l2_penalty,arguments.dccaelamda,arguments.dcccaelamda)
    print(classfile)
    # trainData, tuneData, testData = read_dataset_train("../data/CEDAR_signet_norm.npz","../data/" + handclass + "/CEDAR_hand_norm.npz")

    # Traning.
    model=dcccae.train(model, trainData, tuneData, batch_size=arguments.batch_size, max_epochs=arguments.epoch)

    # Satisfy constraint.
    FX1,_=model.compute_projection(1, trainData.images1)
    FX2,_=model.compute_projection(2, trainData.images2)
    A,B,m1,m2,_=linCCA(FX1, FX2, model.n_z, rcov1, rcov2)

    dataset = read_dataset("../data/CEDAR_signet_norm.npz","../data/CEDAR_signet_norm.npz")
    z_train = np.matmul(model.compute_projection(1, dataset.images1)[0] - m1, A)
    cedar = {'development': z_train, 'devy': dataset.y, 'devlabel': dataset.labels}
    dataset = read_dataset("../data/UTSig_signet_norm.npz","../data/UTSig_signet_norm.npz")
    z_train = np.matmul(model.compute_projection(1, dataset.images1)[0] - m1, A)
    utsig = {'development': z_train, 'devy': dataset.y, 'devlabel': dataset.labels}
    dataset = read_dataset("../data/BHSigB_signet_norm.npz","../data/BHSigB_signet_norm.npz")
    z_train = np.matmul(model.compute_projection(1, dataset.images1)[0] - m1, A)
    bhsigb = {'development': z_train, 'devy': dataset.y, 'devlabel': dataset.labels}
    dataset = read_dataset("../data/BHSigH_signet_norm.npz","../data/BHSigH_signet_norm.npz")
    z_train = np.matmul(model.compute_projection(1, dataset.images1)[0] - m1, A)
    bhsigh = {'development': z_train, 'devy': dataset.y, 'devlabel': dataset.labels}
    dataset = read_dataset("../data/SigComp2011_signet_norm.npz","../data/SigComp2011_signet_norm.npz")
    z_train = np.matmul(model.compute_projection(1, dataset.images1)[0] - m1, A)
    sigcomp2011 = {'development': z_train, 'devy': dataset.y, 'devlabel': dataset.labels}
    np.save(classfile,[cedar,utsig,bhsigb,bhsigh,sigcomp2011])
    return classfile
# for hand in ["hog","lbp","sift","glcm","daisy"]:
#     for trainhand in ['CEDAR', 'UTSig', 'BHSigB', 'BHSigH', 'SigComp2011']:
#         for dataset in ['cedar','utsig','bhsigb','bhsigh','sigcomp2011']:
parser = argparse.ArgumentParser()
parser.add_argument('--modelclass', default="dcccae")
parser.add_argument('--svm-type', choices=['rbf', 'linear'], default='rbf')
parser.add_argument('--datasetname', choices=['cedar','utsig','bhsigb','bhsigh','sigcomp2011'], default="cedar")
parser.add_argument('--trainlable', type=int,choices=[0,1,2], default=0)
parser.add_argument('--gen_for_train', type=int, default=0)
parser.add_argument('--eername', default="tsne图绘画")
parser.add_argument('--trainhand', default="CEDAR")
parser.add_argument('--svm-c', type=float, default=1)
parser.add_argument('--svm-gamma', type=float, default=2 ** -11)
parser.add_argument('--folds', type=int, default=10)
parser.add_argument('--gpuid',default="1")
parser.add_argument('--epoch', type=int, default=50)
parser.add_argument('--batch-size', type=int, default=1000)
parser.add_argument('--learning-rate', type=float, default=0.0001)

parser.add_argument('--hand', default="lbp")
parser.add_argument('--dccaelamda', type=float, default=1000)
parser.add_argument('--dcccaelamda', type=float, default=0.1)#0.1
parser.add_argument('--net-hidden-layer',  nargs='+', type=int ,default=[4096, 4096, 4096, 2048])
arguments = parser.parse_args()
print(arguments)
filename = traindcccae(arguments)
#savephoto(filename)
wd.main(arguments,filename)

