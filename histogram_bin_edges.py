#!/usr/bin/env python3
#-*- coding: utf-8 -*-
import numpy as np

def freedman_diaconis_bins(data, weight=None, max=60000):
    '''
    :param data: data array, a numpy.ndarray
    :param weight: weight array for data, a numpy.ndarray
    :return: the number of bins, an int
    '''
    try:
        # ravel the data and weight
        data = data.ravel()
        if type(weight) == type(None):
            weight = np.ones(data.shape[0])
        weight = weight.ravel()
        assert data.shape == weight.shape, f'weights should have the same shape as data.'

        # sort by data
        idx = np.argsort(data)
        data = data[idx]
        weight = weight[idx]

        N = np.sum(weight)
        if N == 0:
            bins = 1
        else:
            weight_distrib = np.concatenate((np.array([0.0]), np.cumsum(weight)))
            left_bound = np.where(weight_distrib<0.25*N)[0].tolist()[-1]
            right_bound = np.where(weight_distrib<=0.75*N)[0].tolist()[-1]
            IQR = data[right_bound-1]-data[left_bound]
            bw = 2.0 * IQR * N ** (-1.0 / 3.0)
            if bw:
                datarng = np.ptp(data)
                bins = int(np.ceil(datarng/bw))
            else:
                bins = 1

        if np.isinf(bins) or bins > max:
            bins = max
        elif np.isnan(bins):
            bins = 1
        return bins
    except Exception:
        input("exception")
        return 1

if __name__ == "__main__":
# NOTE: run for test
    a = np.array(list(range(2001)))
    w = np.array([1 for i in range(2001)])
    a_bins = freedman_diaconis_bins(a, w)
    print(np.histogram_bin_edges(a, bins=a_bins, weights=w).tolist())
    print(np.histogram_bin_edges(a, bins='auto').tolist())
