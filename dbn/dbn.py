import numpy as np
import rbm_functions as rbm
from scipy.special import expit

##########################################################
x_train, t_train, x_test, t_test = rbm.loadMNIST()
rbm.img_merge(x_train*255,'train.png',10,10)
rbm.img_merge(x_test*255,'test.png',10,10)

##########################################################
#index = 30
iter = 0
for index in range(1,10,4):
    print(iter)
    i_1st = (index//10) #### tens digit
    i_2nd = (index%10) #### ones digit
    
    nHid1 = 50 * (i_1st+1) #### 50 * idx1
    nHid2 = 5 * (i_2nd+1) #### 5 * idx2
    
    structure = np.array([nHid1, nHid2])
    nLayers = structure.shape[0]

    CDtype = 'CD'
    seed = index//100
    np.random.seed(seed)

    print(i_1st, i_2nd, seed)
    print(nHid1,nHid2)
    
    ##########################################################
    Hs = np.zeros(nLayers)
    Hk = np.zeros(nLayers)
    Hs_Test = np.zeros(nLayers)
    Hk_Test = np.zeros(nLayers)

    ##########################################################
    data = x_train ##### set the data set as training set
    test = x_test  ##### set the validation set as test set
    index_origin = index - i_2nd
#    if i_2nd == 0:
#        startLayer = 0
#    else:
#        startLayer = 1
    startLayer = 0
    for layer in range(startLayer):
        print(layer)
        weight,visBias,hidBias = rbm.loadParams('params/params%04d%sL%d.pkl'%(index-i_2nd,CDtype,layer))
        data = expit(np.dot(data,weight) + hidBias)
        test = expit(np.dot(test,weight) + hidBias)
        Hs[layer]     ,Hk[layer]      = rbm.CalHKHS(data,'result/trainMk%04d%sL%d.txt'%(index-i_2nd,CDtype,layer))
        Hs_Test[layer],Hk_Test[layer] = rbm.CalHKHS(test,'result/testMk%04d%sL%d.txt'%(index-i_2nd,CDtype,layer))

    print('Starting Data is Loaded')

    for layer in range(startLayer, nLayers):
        ######## Initialize the Params #######
        nVis = data.shape[1]
        nHid = structure[layer]
        print('layer, nVis, nHid')
        print(layer, nVis, nHid)
        weight, visBias, hidBias = rbm.trainRBM(data, test, nVis, nHid, CDtype = CDtype, nEpoches=200)
        rbm.saveParams('params/params%04d%sL%d.pkl'%(index,CDtype,layer),weight,visBias,hidBias)
#        rbm.imageReconstruction(data,test,weight,visBias,hidBias,index,CDtype,layer)
        data = expit(np.dot(data,weight) + hidBias)
        test = expit(np.dot(test,weight) + hidBias)
        Hs[layer]     ,Hk[layer]      = rbm.CalHKHS(data,'result/trainMk%04d%sL%d.txt'%(index,CDtype,layer))
        Hs_Test[layer],Hk_Test[layer] = rbm.CalHKHS(test,'result/testMk%04d%sL%d.txt'%(index,CDtype,layer))

    print(Hs)
    print(Hk)

    filehkhs = 'result/HKHSTrain%04d%sL%d.txt'%(index,CDtype,layer)
    np.savetxt(filehkhs, np.c_[Hs,Hk])

    filehkhs = 'result/HKHSTest%04d%sL%d.txt'%(index,CDtype,layer)
    np.savetxt(filehkhs, np.c_[Hs_Test,Hk_Test])

    iter += 1

