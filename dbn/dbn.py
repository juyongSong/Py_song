'''

Deep Belief Networks Learning and Generate samples of the models.

Ones number : idx_nEpoches(0: no learning, 9: 200 Epoches, 1~8: 2^(n-1))
Tens number : idx_structure
First two digit : Random Seed

'''

import sys
import numpy as np
import dbn_functions as rbm
from scipy.special import expit

if len(sys.argv)!=2:
    index = 117
else:
    index = int(sys.argv[1])    ###### idx of learning
##########################################################
x_train, t_train, x_test, t_test = rbm.loadMNIST()
rbm.img_merge(x_train*255,'train.png',10,10)
rbm.img_merge(x_test*255,'test.png',10,10)

##########################################################

idx_nEpoches = (index%10)
idx_structure = (index//10)%10
seed = index//100

if idx_structure==1:
    structure = np.array([500, 250, 120, 60, 30, 25, 20, 15, 10, 5, 3, 2])
elif idx_structure==2:
    structure = np.array([500, 1000, 500, 250, 120, 60, 30, 25, 20, 15, 10, 5, 3, 2])
else :
    structure = np.array([500, 10])

nLayers = structure.shape[0]

if idx_nEpoches==0:
    nEpoches = 0
elif idx_nEpoches==9:
    nEpoches = 200
else:
    nEpoches = int(2**(idx_nEpoches-1))

CDtype = 'PCD'
np.random.seed(seed)

##########################################################
Hs = np.zeros(nLayers)
Hk = np.zeros(nLayers)
Hs_Test = np.zeros(nLayers)
Hk_Test = np.zeros(nLayers)

##########################################################
data = x_train ##### set the data set as training set
test = x_test  ##### set the validation set as test set
startLayer = 0
for layer in range(startLayer):
    filename = '%04d%sL%02d'%(index,CDtype,layer)
    print(layer)
    weight,visBias,hidBias = rbm.loadParams('params/params%s.pkl'%(filename))
    data = expit(np.dot(data,weight) + hidBias)
    test = expit(np.dot(test,weight) + hidBias)
    Hs[layer]     ,Hk[layer]      = rbm.CalHKHS(data,'result/trainMk%s.txt'%(filename))
    Hs_Test[layer],Hk_Test[layer] = rbm.CalHKHS(test,'result/testMk%s.txt'%(filename))

print('Starting Data is Loaded')

for layer in range(startLayer, nLayers):
    filename = '%04d%sL%02d'%(index,CDtype,layer)
    ######## Initialize the Params #######
    nVis = data.shape[1]
    nHid = structure[layer]
    print('layer, nVis, nHid')
    print(layer, nVis, nHid)
    ######### Learning of each RBM #########
    weight, visBias, hidBias = rbm.trainRBM(data, test, nHid, CDtype=CDtype,nEpoches=nEpoches,seed=seed)
    rbm.saveParams('params/params%s.pkl'%(filename),weight,visBias,hidBias)
    ######### update the data for the next layer #########
    data = expit(np.dot(data,weight) + hidBias)
    test = expit(np.dot(test,weight) + hidBias)
    ######### Clustering Analysis with H[s] and H[k] #########
    Hs[layer]     ,Hk[layer]      = rbm.CalHKHS(data,'result/trainMk%s.txt'%(filename))
    Hs_Test[layer],Hk_Test[layer] = rbm.CalHKHS(test,'result/testMk%s.txt'%(filename))

print(Hs)
print(Hk)

filehkhs = 'result/HKHStrain%s.txt'%(filename)
np.savetxt(filehkhs, np.c_[Hs,Hk])

filehkhs = 'result/HKHStest%s.txt'%(filename)
np.savetxt(filehkhs, np.c_[Hs_Test,Hk_Test])

############## Generation from the random initial conditions #############

nGeneration = 100
nCD = 2000

for layer in range(nLayers):
    filename = '%04d%sL%02d'%(index,CDtype,layer)
    params_path = 'params/params%s.pkl'%(filename)
    weight, visBias, hidBias = rbm.loadParams(params_path)
    sample = np.random.rand(nGeneration, weight.shape[1])
    tmpSample = rbm.generation(sample, weight, visBias, hidBias, nCD = nCD)
    images = rbm.imageReconstruction(tmpSample,index,CDtype,layer)
    rbm.img_merge(images*255,'images/generatedImages%s.png'%(filename),10,10)

