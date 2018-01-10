import numpy as np
import rbm_functions as rbm
from scipy.special import expit

##########################################################
#x_train, t_train, x_test, t_test = rbm.loadMNIST()

for index in range(400,401,10):
    nHid = index%1000
    nGeneration = 100
    nCD = 20000
    CDtype = 'PCD'
    filename = 'params/params%d%s.pkl'%(index,CDtype)
    weight, visBias, hidBias = rbm.loadParams(filename)
    print(weight.shape)
    print(visBias.shape)
    print(hidBias.shape)
    sample = np.random.rand(nGeneration, nHid)
    images = rbm.generation(sample, weight, visBias, hidBias, nCD = nCD)
    rbm.img_merge(images*255,'test_generation%d.png'%(index),10,10)
