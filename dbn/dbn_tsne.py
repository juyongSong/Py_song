'''

t-SNE of each layer in Deep Belief Networks

Ones number : idx_nEpoches(0: no learning, 9: 200 Epoches, 1~8: 2^(n-1))
Tens number : idx_structure
First two digit : Random Seed

'''
import os
import sys
import time
import numpy as np
import dbn_functions as rbm
import matplotlib as mpl
mpl.use('Agg')

from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from scipy.special import expit

plt.ioff()


if len(sys.argv)!=2:
    index = 117
else:
    index = int(sys.argv[1])    ###### idx of learning
##########################################################
x_train, lb_train, x_test, lb_test = rbm.loadMNIST(one_hot=False)
nSamples = 1000

#############################
CDtype = 'PCD'
tmp = 0
for i in range(20):
    filename = 'params/params%04d%sL%02d.pkl'%(index,CDtype,i)
    if os.path.exists(filename):
        print(filename)
        tmp += 1

nLayer = tmp
nSamples = 1000
data = x_train[:nSamples]

startLayer = 0
endLayer = nLayer
ts = time.time()

X = data
Y = lb_train[:nSamples]
X_embedded = TSNE(n_components=2, init='pca').fit_transform(X)

plt.figure()
#plt.scatter(X_embedded[Y==0,0], X_embedded[Y==0,1],color='red')
#plt.scatter(X_embedded[Y==1,0], X_embedded[Y==1,1],color='darksalmon')
#plt.scatter(X_embedded[Y==2,0], X_embedded[Y==2,1],color='sienna')
#plt.scatter(X_embedded[Y==3,0], X_embedded[Y==3,1],color='gold')
#plt.scatter(X_embedded[Y==4,0], X_embedded[Y==4,1],color='olivedrab')
#plt.scatter(X_embedded[Y==5,0], X_embedded[Y==5,1],color='deepskyblue')
#plt.scatter(X_embedded[Y==6,0], X_embedded[Y==6,1],color='navy')
#plt.scatter(X_embedded[Y==7,0], X_embedded[Y==7,1],color='blue')
#plt.scatter(X_embedded[Y==8,0], X_embedded[Y==8,1],color='mediumpurple')
#plt.scatter(X_embedded[Y==9,0], X_embedded[Y==9,1],color='mediumvioletred')


plt.scatter(X_embedded[:,0], X_embedded[:,1], c=Y, cmap=plt.cm.get_cmap("jet", 10))
plt.colorbar(ticks=range(10))
plt.clim(-0.5, 9.5)
plt.savefig('images/tsne_pca_mnist.png',bbox_inches='tight')


for layer in range(startLayer, endLayer):
    print(layer)
#    for l in range(layer+1):
    filename = 'params/params%04d%sL%02d.pkl'%(index,CDtype,layer)
    weight, visBias, hidBias = rbm.loadParams(filename)
    data = expit(np.dot(data,weight)+hidBias)
    X = data
    Y = lb_train[:nSamples]
    X_embedded = TSNE(n_components=2, init='random', random_state=0).fit_transform(X)

    plt.figure()
    plt.scatter(X_embedded[Y==0,0], X_embedded[Y==0,1],color='red')
    plt.scatter(X_embedded[Y==1,0], X_embedded[Y==1,1],color='darksalmon')
    plt.scatter(X_embedded[Y==2,0], X_embedded[Y==2,1],color='sienna')
    plt.scatter(X_embedded[Y==3,0], X_embedded[Y==3,1],color='gold')
    plt.scatter(X_embedded[Y==4,0], X_embedded[Y==4,1],color='olivedrab')
    plt.scatter(X_embedded[Y==5,0], X_embedded[Y==5,1],color='deepskyblue')
    plt.scatter(X_embedded[Y==6,0], X_embedded[Y==6,1],color='navy')
    plt.scatter(X_embedded[Y==7,0], X_embedded[Y==7,1],color='blue')
    plt.scatter(X_embedded[Y==8,0], X_embedded[Y==8,1],color='mediumpurple')
    plt.scatter(X_embedded[Y==9,0], X_embedded[Y==9,1],color='mediumvioletred')

    plt.savefig('images/dbn_tsne%04d%sL%02d.png'%(index, CDtype, layer),bbox_inches='tight')

tf = time.time()
print(tf-ts,'s')
