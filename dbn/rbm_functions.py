import os
import sys
import numpy as np
import time
import pickle

from scipy.special import expit
from mnist import MNIST
from PIL import Image
if not os.path.exists('params/'):
        os.mkdir('params/')
if not os.path.exists('result/'):
        os.mkdir('result/')
if not os.path.exists('images/'):
        os.mkdir('images/')



class params():
    def __init__ (self,
                  nVis,
                  nHid,
                  nEpoches = 200,
                  batchsize = 100,
                  init_rate = 0.1,
                  CDtype='CD',
                  CDstep=1,
                  alpha= 0.,
                  L1_reg = 0.,
                  L2_reg = 2e-04):
        self.nVis = nVis
        self.nHid = nHid
        self.nEpoches = nEpoches
        self.lastUpdate = nEpoches//10
        self.batchsize = batchsize
        self.init_rate = init_rate
        self.learning_rate = init_rate
        self.CDtype = CDtype
        self.CDstep = CDstep
        if CDtype == 'PCD':
            self.CDstep = 1
            self.nChain = 4
            self.chainsize = int(batchsize / nChain)
            self.model = np.random.rand(chainsize,nVis)
        self.negVis = np.zeros([batchsize,nVis])
        self.negHid = np.zeros([batchsize,nHid])
        self.alpha = alpha
        self.L1_reg = L1_reg
        self.L2_reg = L2_reg
        self.weight = np.sqrt(6./(self.nVis+self.nHid)) * np.random.randn(self.nVis, self.nHid)
        self.visBias = np.zeros([1, self.nVis])
        self.hidBias = np.zeros([1, self.nHid])
        self.dw = np.zeros([self.nVis, self.nHid])
        self.da = np.zeros([1, self.nVis])
        self.db = np.zeros([1, nHid])


def sampling(x):
    y = np.sign(x - np.random.rand(x.shape[0],x.shape[1]))
    y = y/2 + 0.5
    return y

def indices_to_one_hot(data, nb_classes):
    """Convert an iterable of indices to one-hot encoded labels."""
    targets = np.array(data).reshape(-1)
    return np.eye(nb_classes)[targets]

def softmax(y):
    partitionZ = np.array([np.sum(np.exp(y),axis=1)]).T
    return np.exp(y)/partitionZ

def loadMNIST(normalization=True,one_hot=True):

    mndata = MNIST('../data/')
    x_train, lb_train = mndata.load_training()
    x_test, lb_test = mndata.load_testing()
    x_train, lb_train, x_test, lb_test = np.array(
              x_train), np.array(lb_train), np.array(x_test), np.array(lb_test)

    if normalization==True:
        x_train = x_train / 255.
        x_test = x_test / 255.
    if one_hot == True:
        t_train = indices_to_one_hot(lb_train,lb_train.max()+1)
        t_test  = indices_to_one_hot(lb_test, lb_test.max()+1)
    else:
        t_train, t_test = lb_train, lb_test
    print('Data is loaded')
    return x_train, t_train, x_test, t_test


def saveParams(filename, weight, visBias, hidBias):
    f = open(filename, 'wb')
    pickle.dump(weight, f)
    pickle.dump(visBias, f, -1)
    pickle.dump(hidBias, f, -1)
    f.close()

def loadParams(filename):
    f = open(filename, 'rb')
    weight = pickle.load(f)
    visBias = pickle.load(f)
    hidBias = pickle.load(f)
    f.close()
    return weight, visBias, hidBias

def contrastiveDivergence(p, hid):
    if p.CDtype == 'PCD':
        for i in range(p.nChain):
            hid = expit(np.dot(p.model,p.weight) + p.hidBias)
            for cd in range(p.CDstep):
                vis = expit(np.dot(sampling(hid),p.weight.T) + p.visBias)
                hid = expit(np.dot(vis,p.weight) + p.hidBias)
            p.model = expit(np.dot(sampling(hid),p.weight.T)+p.visBias)
            p.negVis[i*p.chainsize:(i+1)*p.chainsize] = vis
            p.negHid[i*p.chainsize:(i+1)*p.chainsize] = hid

    elif p.CDtype == 'CD':
        for cd in range(p.CDstep):
            vis = expit(np.dot(sampling(hid),p.weight.T) + p.visBias)
            hid = expit(np.dot(sampling(vis),p.weight) + p.hidBias)
        p.negVis = vis
        p.negHid = hid

    else:
        print('cd type should be CD or PCD')
        exit()

def weightUpdate(data,test,p):
    for epoch in range(p.nEpoches):
        ts = time.time()
        if epoch <= 5:
            p.alpha = .5 - .1 * epoch
        else:
            p.alpha = 0.
        if epoch > p.nEpoches - p.lastUpdate:
            p.learning_rate = p.init_rate / (epoch+1-(p.nEpoches - p.lastUpdate))
        else:
            p.learning_rate = p.init_rate

        for batch in range(int(data.shape[0]/p.batchsize)):
            vis = data[batch*p.batchsize : (batch+1)*p.batchsize]
            hid = expit(np.dot(vis,p.weight)+p.hidBias)
            posVis = vis;posHid = hid
            contrastiveDivergence(p,hid)

            grad_w = (np.dot(posVis.T,posHid) - np.dot(p.negVis.T, p.negHid))/p.batchsize
            grad_a = np.mean(posVis,axis=0) - np.mean(p.negVis, axis=0)
            grad_b = np.mean(posHid,axis=0) - np.mean(p.negHid, axis=0)

            p.dw = p.learning_rate * (grad_w + p.alpha * p.dw - p.L2_reg * p.weight)
            p.da = p.learning_rate * (grad_a + p.alpha * p.da)
            p.db = p.learning_rate * (grad_b + p.alpha * p.db)

            p.weight += p.dw
            p.visBias += p.da
            p.hidBias += p.db
        
        hid = expit(np.dot(data,p.weight)+p.hidBias)
        reconTrainImg = expit(np.dot(hid,p.weight.T) + p.visBias)
        trainErr = np.mean((reconTrainImg-data)**2)
        hid = expit(np.dot(test,p.weight)+p.hidBias)
        reconTestImg = expit(np.dot(hid,p.weight.T) + p.visBias)
        testErr = np.mean((reconTestImg-test)**2)
        print(epoch, trainErr, testErr, (time.time()-ts) * (p.nEpoches-epoch),p.learning_rate,p.alpha,p.CDstep)

def trainRBM(data, test, nVis, nHid, CDtype = 'CD', nEpoches=10):
    p = params(data.shape[1], nHid, CDtype = CDtype, nEpoches = nEpoches)
    ######## weights update ########
    weightUpdate(data,test,p)

    return p.weight, p.visBias, p.hidBias

def imageReconstruction(data,test,weight,visBias,hidBias,index,CDtype,layer):
    tmp = expit(np.dot(data,weight) + hidBias)
    reconTrainImg = expit(np.dot(tmp,weight.T) + visBias)
    tmp = expit(np.dot(test,weight) + hidBias)
    reconTestImg = expit(np.dot(tmp,weight.T) + visBias)
    ###### Finished Learining of RBM ######
    for l in range(layer-1,-1,-1):
        weight, visBias, hidBias = loadParams('params/params%d%sL%d.pkl'%(0,CDtype,l))
        reconTrainImg = expit(np.dot(reconTrainImg,weight.T) + visBias)
        reconTestImg  = expit(np.dot(reconTestImg ,weight.T) + visBias)
    img_merge(reconTrainImg*255,'images/reconTrainImg%d%sL%d.png'%(index,CDtype,layer),10,10)
    img_merge(reconTestImg *255,'images/reconTestImg%d%sL%d.png' %(index,CDtype,layer),10,10)


def CalHKHS(data, filename):
    data = (data>.5).astype(float)
    M = float(data.shape[0])
    _, ks = np.unique(data, return_counts=True, axis=0)
    mk, _ = np.histogram(ks, bins = np.arange(max(ks)+1)+1)
    k = np.arange(max(ks))+1
    kmk = k[mk!=0] * mk[mk!=0]
    Hs = - np.dot(ks/M, np.log(ks/M))
    Hk = - np.dot(kmk/M, np.log(kmk/M))
    np.savetxt(filename, np.c_[k,mk], fmt = '%d')
    return Hs/np.log(M), Hk/np.log(M)

def img_merge(x_train,filename,len_x,len_y, *arg):
    if x_train.ndim==2:
        len_img = x_train.shape[1]
    elif x_train.ndim==3:
        len_img = x_train.shape[1]*x_train.shape[2]
    if len(arg)==0:
        x_img = int(np.sqrt(len_img))
        y_img = int(np.sqrt(len_img))
    elif len(arg)==2:
        x_img = int(arg[0])
        y_img = int(arg[1])
    elif len(arg)==1:
        x_img = int(arg[0])
        y_img = int(len_img/x_img)
    merging_img = np.zeros([x_img * len_x , y_img * len_y])
    x_tmp = x_train.reshape(x_train.shape[0], x_img, y_img)
    for i in range(len_x):
        for j in range(len_y):
            if (i*len_y + j < x_train.shape[0]):
                merging_img[ i*x_img : (i+1)*x_img,
                        j*y_img : (j+1)*y_img
                        ] = x_tmp[i*len_y + j]
    pil_img = Image.fromarray(np.uint8(merging_img))
    pil_img.save(filename)
    print('Image : %s is saved.' %(filename))



