import numpy as np
from scipy.special import expit
import time
from mnist import MNIST

def indices_to_one_hot(data, nb_classes):
    """Convert an iterable of indices to one-hot encoded labels."""
    targets = np.array(data).reshape(-1)
    return np.eye(nb_classes)[targets]

def softmax(y):
    partitionZ = np.array([np.sum(np.exp(y),axis=1)]).T
    return np.exp(y)/partitionZ

mndata = MNIST('../data')
x_train, lb_train = mndata.load_training()
x_test,  lb_test  = mndata.load_testing()

x_train, lb_train = np.array(x_train), np.array(lb_train)
x_test,  lb_test  = np.array(x_test),  np.array(lb_test)
print('Data loaded')

########### Setting the hyperparameters ###########
learningRate = 0.3
nEpoches = 100

nData = x_train.shape[0]
nTest = x_test.shape[0]
bsize = 60000   ##### the size of mini-batch
nBatches = nData/bsize

nHid = 500  #### the number of hidden variable
nOut = lb_train.max()+1 #### the number of output variable

print(learningRate, nEpoches, bsize, nHid)
########### One-Hot encoding ###########
t_train = indices_to_one_hot(lb_train,nOut)
t_test  = indices_to_one_hot(lb_test,nOut)

########### Initialize the parameters #############
weight1 = np.sqrt(6./(x_train.shape[1]+nHid)) * np.random.randn(x_train.shape[1], nHid) #### (nInput, nHid)
weight2 = np.sqrt(6./(nHid + nOut)) * np.random.randn(nHid, nOut) #### (nHid, nOut)

bias1 = np.zeros([1,nHid])
bias2 = np.zeros([1,nOut])

start_timing = time.time()  #### to calculate time-consuming

for epoch in range(nEpoches):
    for batch in range(nBatches):
        x = x_train[batch*bsize:(batch+1)*bsize]
        t = t_train[batch*bsize:(batch+1)*bsize]
        hid = expit(np.dot(x  , weight1) + bias1) #### (nData, nHid)
        out = expit(np.dot(hid, weight2) + bias2) #### (nData, nOut)
        
        dOut = (t-out) #### (nData, nOut)
        dHid = np.dot(dOut,weight2.T) * hid * (1 - hid) #### (nData, nHid)

        grad_weight2 = np.dot(hid.T, dOut) / nData
        grad_weight1 = np.dot(x.T, dHid) / nData

        weight2 += learningRate * grad_weight2
        weight1 += learningRate * grad_weight1

        bias2 += learningRate * np.mean(dOut,axis=0)
        bias1 += learningRate * np.mean(dHid,axis=0)
    
    x = x_train
    t = t_train
    hid = expit(np.dot(x  , weight1) + bias1) #### (nData, nHid)
    out = expit(np.dot(hid, weight2) + bias2) #### (nData, nOut)
    y = np.argmax(out,axis=1)
    trainError = np.sum((lb_train!=y).astype(int))*100./nData
    
    hid = expit(np.dot(x_test  , weight1) + bias1) #### (nTest, nHid)
    out = expit(np.dot(hid, weight2) + bias2) #### (nTest, nOut)
    y_t = np.argmax(out,axis=1)
    testError  = np.sum((lb_test!=y_t).astype(int))*100./nTest

    print(y[:10])   #### print out the first output labels predictions of the machine
    print(lb_train[:10])    #### print out the first training labels
    print(epoch, trainError, testError)

print (time.time() - start_timing, 's')



