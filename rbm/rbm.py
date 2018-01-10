import numpy as np
import rbm_functions as rbm
from scipy.special import expit

##########################################################
x_train, t_train, x_test, t_test = rbm.loadMNIST()
rbm.img_merge(x_train*255,'train.png',10,10)
rbm.img_merge(x_test*255,'test.png',10,10)

#x_train = x_train[:100,:]

##########################################################

for index in range(30,51,20):
    structure = np.array([index%1000])
    nLayers = structure.shape[0]

    CDtype = 'PCD'
    seed = index // 1000
    np.random.seed(seed)

    ##########################################################
    Hs = np.zeros(nLayers)
    Hk = np.zeros(nLayers)
    Hs_Test = np.zeros(nLayers)
    Hk_Test = np.zeros(nLayers)
    ##########################################################
    data = x_train ##### set the data set as training set
    test = x_test  ##### set the validation set as test set

    ######## Initialize the Params #######
    nVis = data.shape[1]
    nHid = structure[0]
    print('nVis, nHid')
    print(nVis, nHid)
    weight, visBias, hidBias = rbm.trainRBM(data, test, nVis, nHid, CDtype = CDtype, nEpoches=200)
    rbm.saveParams('params/params%d%s.pkl'%(index,CDtype),weight,visBias,hidBias)
#    rbm.imageReconstruction(data,test,weight,visBias,hidBias,index,CDtype,0)
    data = expit(np.dot(data,weight) + hidBias)
    test = expit(np.dot(test,weight) + hidBias)
    Hs     ,Hk      = rbm.CalHKHS(data,'result/trainMk%d%s.txt'%(index,CDtype))
    Hs_Test,Hk_Test = rbm.CalHKHS(test,'result/testMk%d%s.txt'%(index,CDtype))

    print(Hs)
    print(Hk)

    filehkhs = 'result/HKHSTrain%d%s.txt'%(index,CDtype)
    np.savetxt(filehkhs, np.c_[Hs,Hk])

    filehkhs = 'result/HKHSTest%d%s.txt'%(index,CDtype)
    np.savetxt(filehkhs, np.c_[Hs_Test,Hk_Test])





