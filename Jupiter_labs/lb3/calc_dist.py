import numpy as np
from scipy.spatial import distance

#######
p = 3
q = 4
w = [0.19,	0.25,	0.20,	0.36]
#######
def euclidean(a, b):
    return distance.euclidean(a, b)

def manhattan(a, b):
    return distance.cityblock(a, b)

def euclidean_w(a, b):
    '''
    взвеш евклидово
    :param w: размерность w как у a и b
    '''
    return distance.euclidean(a, b, w)

def stepen(a, b):
    '''
    Степенное
    :param p: 1 число
    :param q: 1 число
    '''
    sum_ = 0
    for e in range(len(a)):
        sum_+=abs((a[e]-b[e]))**p

    return sum_**(1/q)

def calc_centroid(a):
    '''
        для расчёта центроида
    '''
    index = [i for i in range(len(a[0]))]
    sum_ = [0 for i in range(len(a[0]))]
    for i in a:
        for j in index:
            sum_[j]+=i[j]

    return np.array(sum_) / len(a)








