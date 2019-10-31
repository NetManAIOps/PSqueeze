#!/usr/bin/env python3
#-*- coding: utf-8 -*-
import numpy as np

def freedman_diaconis_bins(data, weight):
    '''
    :param data: data array, a flattened numpy.ndarray
    :param weight: weight array for data, a flattend numpy.ndarray
    :return: the number of bins, an int
    '''
    idx = np.argsort(data)
    data = data[idx]
    weight = weight[idx]

    N = np.sum(weight)
    weight_distrib = np.concatenate((np.array([0.0]), np.cumsum(weight/N)))
    left_bound = np.where(weight_distrib<0.25)[0].tolist()[-1]
    right_bound = np.where(weight_distrib>0.75)[0].tolist()[0]
    IQR = np.ptp(data[left_bound:right_bound])
    bw = (2 * IQR) / np.power(N, 1/3)

    datarng = np.ptp(data)
    return int((datarng / bw) + 1)

if __name__ == "__main__":
# NOTE: run for test
    a = np.array(list(range(2001)))
    w = np.array([1 for i in range(2001)])
    a_bins = freedman_diaconis_bins(a, w)
    print(np.histogram_bin_edges(a, bins=a_bins, weights=w).tolist())
    print(np.histogram_bin_edges(a, bins='auto').tolist())
