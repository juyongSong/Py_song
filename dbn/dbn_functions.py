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



class rbm():
    def __init__ (self,
                  nVis,
                  nHid,
                  nEpoches = 200,
                  batchsize = 100,
                  init_rate = 0.1,
                  CDtype='CD',
                  nChain=4,
                  CDstep=4,
                  alpha= 0.,
                  L1_reg = 0.,
                  L2_reg = 1e-4,
                  random_seed = 0):
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
            self.nChain = nChain
            self.chainsize = int(batchsize / nChain)
            self.model = np.random.rand(self.chainsize,nVis)
        self.negVis = np.zeros([batchsize,nVis])
        self.negHid = np.zeros([batchsize,nHid])
        self.alpha = alpha
        self.L1_reg = L1_reg
        self.L2_reg = L2_reg
        self.random_seed = random_seed
        np.random.seed(random_seed)
        self.weight = 4.*np.sqrt(6./(self.nVis+self.nHid)) * np.random.randn(self.nVis, self.nHid)
#        self.weight = .5 * np.random.randn(self.nVis, self.nHid)
        self.visBias = np.zeros([1, self.nVis])
        self.hidBias = np.zeros([1, self.nHid])
        self.dw = np.zeros([self.nVis, self.nHid])
        self.da = np.zeros([1, self.nVis])
        self.db = np.zeros([1, nHid])


    def forwardProp(self, x):
        return expit(np.dot(x,self.weight) + self.hidBias)
    
    def backwardProp(self, x):
        return expit(np.dot(x,self.weight.T) + self.visBias)

    def contrastiveDivergence(self, hid):
        if self.CDtype == 'PCD':
            for i in range(self.nChain):
                hid = self.forwardProp(self.model)
                for cd in range(self.CDstep):
                    vis = self.backwardProp(sampling(hid))
                    hid = self.forwardProp(vis)
                self.model = self.backwardProp(sampling(hid))
                self.negVis[i*self.chainsize:(i+1)*self.chainsize] = vis
                self.negHid[i*self.chainsize:(i+1)*self.chainsize] = hid

        elif self.CDtype == 'CD':
            for cd in range(self.CDstep):
                vis = self.backwardProp(sampling(hid))
                hid = self.forwardProp(vis)
            self.negVis = vis
            self.negHid = hid

        else:
            print('cd type should be CD or PCD')
            exit()

    def weightUpdate(self, data,test):
        for epoch in range(self.nEpoches):
            ts = time.time()
            if epoch <= 5:
                self.alpha = .5 - .1 * epoch
            else:
                self.alpha = 0.
            if epoch > self.nEpoches - self.lastUpdate:
                self.learning_rate = self.init_rate / (epoch+1-(self.nEpoches - self.lastUpdate))
            else:
                self.learning_rate = self.init_rate

            for batch in range(int(data.shape[0]/self.batchsize)):
                vis = data[batch*self.batchsize : (batch+1)*self.batchsize]
                hid = self.forwardProp(vis)
                posVis = vis;posHid = hid
                self.contrastiveDivergence(hid)

                grad_w = (np.dot(posVis.T,posHid) - np.dot(self.negVis.T, self.negHid))/self.batchsize
                grad_a = np.mean(posVis,axis=0) - np.mean(self.negVis, axis=0)
                grad_b = np.mean(posHid,axis=0) - np.mean(self.negHid, axis=0)

                self.dw = self.learning_rate * (grad_w + self.alpha * self.dw
                                          - self.L2_reg * self.weight - self.L1_reg * np.sign(self.weight))
                self.da = self.learning_rate * (grad_a + self.alpha * self.da)
                self.db = self.learning_rate * (grad_b + self.alpha * self.db)

                self.weight += self.dw
                self.visBias += self.da
                self.hidBias += self.db
            
            hid = self.forwardProp(data)
            reconTrainImg = self.backwardProp(hid)
            trainErr = np.mean((reconTrainImg-data)**2)
            hid = self.forwardProp(test)
            reconTestImg = self.backwardProp(hid)
            testErr = np.mean((reconTestImg-test)**2)
            print(epoch, trainErr, testErr, (time.time()-ts) * (self.nEpoches-epoch-1),self.learning_rate,self.alpha,self.CDstep)




def histogramOfWeight(weight, filename):
    width = 1/200
    k, edges = np.histogram(data, bins = np.arange(0,1+width, width))
    k = k/data.shape[0]/data.shape[1]
    np.savetxt(filename, np.c_[edges[:-1], k], fmt = '%.4f %.4f')


def histogramOfActivity(data, filename):
    width = 1/200
    k, edges = np.histogram(data, bins = np.arange(0,1+width, width))
    k = k/data.shape[0]/data.shape[1]
    np.savetxt(filename, np.c_[edges[:-1], k], fmt = '%.4f %.4f')


def imageReconstruction(sample,index,CDtype,layer):
    if layer==0:
        reconImg = sample
    for l in range(layer-1,-1,-1):
        w,a,b = loadParams('params/params%04d%sL%02d.pkl'%(index,CDtype,l))
        if l == layer-1:
            reconImg = expit(np.dot(sample, w.T)+ a)
        else:
            reconImg = expit(np.dot(reconImg, w.T)+ a)
    return reconImg


def generation(init_state, weight, visBias, hidBias, nCD):
    hid=init_state
    vis = expit(np.dot(sampling(hid), weight.T)+ visBias)
    for cd in range(nCD):
        hid = expit(np.dot(vis, weight) + hidBias)
        vis = expit(np.dot(sampling(hid), weight.T)+ visBias)
    return vis

def trainRBM(data, test, nHid, CDtype = 'CD', nEpoches=10, seed = 0):
    p = rbm(data.shape[1], nHid, CDtype = CDtype, nEpoches = nEpoches, random_seed = seed)
    ######## weights update ########
    p.weightUpdate(data,test)
    return p.weight, p.visBias, p.hidBias


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

def sampling(x):
    y = np.sign(x - np.random.rand(x.shape[0],x.shape[1]))
    return y/2 + 0.5

def indices_to_one_hot(data, nb_classes):
    """Convert an iterable of indices to one-hot encoded labels."""
    targets = np.array(data).reshape(-1)
    return np.eye(nb_classes)[targets]

def softmax(y):
    partitionZ = np.array([np.sum(np.exp(y),axis=1)]).T
    return np.exp(y)/partitionZ

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





