import random
import numpy as np
from scipy.spatial import cKDTree as KDTree

def random_crop(img, mask, width, height):
    assert img.shape[0] >= height
    assert img.shape[1] >= width
    assert img.shape[0] == mask.shape[0]
    assert img.shape[1] == mask.shape[1]
    x = random.randint(0, img.shape[1] - width)
    y = random.randint(0, img.shape[0] - height)
    img = img[y:y+height, x:x+width]
    mask = mask[y:y+height, x:x+width]
    return img, mask



def KLdivergence(x, y):
  #Compute the Kullback-Leibler divergence between two multivariate samples. D(P||Q)

    x = np.atleast_2d(x)
    y = np.atleast_2d(y)

    n,d = x.shape
    m,dy = y.shape

    assert(d == dy)

    xtree = KDTree(x)
    ytree = KDTree(y)

    r = xtree.query(x, k=2, eps=.01, p=2)[0][:,1]
    s = ytree.query(x, k=1, eps=.01, p=2)[0]

    for i in range(len(x[0])):
        if r[i]==0:
            r[i]=r[i]+1e-10
            
    return -np.log(r/s).sum() * d / n + np.log(m / (n - 1.))