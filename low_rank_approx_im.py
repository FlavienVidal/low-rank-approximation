import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import pickle

from dim_red_functions import pca
from dim_red_functions import svd
from dim_red_functions import nmf
from dim_red_functions import mds


logo = mpimg.imread('google_logo.png')
plt.imshow(logo)
plt.show()



results_task2 = np.zeros(logo.shape + tuple([20]))
funcs = [pca, svd, nmf, mds]
for i in range(4):
    print(['PCA', 'SVD', 'NMF', 'MDS'][i])
    method = funcs[i]
    k_values = [1,5,10,30,60]
    for j in range(5) :
        print(['k = 1','k = 5','k = 10','k = 30','k = 60'][j])
        k = k_values[j]
        if not(i):
            x, _ = method(logo, k)
            results_task2[:, :, :, i*5+j] = x
        else:
            results_task2[:, :, :, i*5+j] = method(logo, k)


fig, axs = plt.subplots(4, 5, figsize=(8, 10))


titles = ['k = 1','k = 5','k = 10','k = 30','k = 60']
methods = ['PCA', 'SVD', 'NMF', 'MDS']
for i in range(4):
    axs[i,0].set_ylabel(methods[i], fontsize=16,fontweight='bold')
    for j in range(5):
        if i==0:
            axs[i,j].set_title(titles[j], fontweight='bold')
        axs[i,j].imshow(np.clip(results_task2[:,:,:,i*5+j],0,1))
        axs[i,j].set_xticks([])
        axs[i,j].set_yticks([])
fig.tight_layout(pad=0)

plt.show()

with open('results/results_task2.pickle', 'wb') as f:
    pickle.dump(results_task2, f)


