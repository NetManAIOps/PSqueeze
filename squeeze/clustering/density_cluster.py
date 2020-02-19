from typing import List
import seaborn as sns
import numpy as np
from loguru import logger
from scipy.stats import gaussian_kde
from scipy.signal import argrelextrema
import matplotlib.pyplot as plt
from squeeze.clustering.cluster import Cluster
from squeeze.squeeze_option import SqueezeOption
from kneed import KneeLocator
from histogram_bin_edges import freedman_diaconis_bins
import json
import sys


def smooth(arr, window_size):
    new_arr = np.convolve(arr, np.ones(window_size), mode="valid") / window_size
    new_arr = np.concatenate([arr[:window_size - 1], new_arr])
    assert np.shape(new_arr) == np.shape(arr)
    return new_arr


class DensityBased1dCluster(Cluster):
    def __init__(self, option: SqueezeOption):
        super().__init__(option)
        assert option.density_estimation_method in {'kde', 'histogram', 'histogram_prob'}
        self.density_estimation_func = {
            "kde": self._kde,
            "histogram": self._histogram,
            "histogram_prob": self._histogram_prob,
        }[option.density_estimation_method]

    def _kde(self, array: np.ndarray):
        kernel = gaussian_kde(array, bw_method=self.option.kde_bw_method, weights=self.option.kde_weights)
        samples = np.arange(np.min(array), np.max(array), 0.01)
        kde_sample = kernel(points=samples)
        conv_kernel = self.option.density_smooth_conv_kernel
        kde_sample_smoothed = np.convolve(kde_sample, conv_kernel, 'full') / np.sum(conv_kernel)
        return kde_sample_smoothed, samples

    def _histogram(self, array: np.ndarray):
        assert len(array.shape) == 1, f"histogram receives array with shape {array.shape}"
        def _get_hist(_width):
            if _width == 'auto':
                _edges = np.histogram_bin_edges(array, bins='auto').tolist()
                if self.option.max_bins and len(_edges) > self.option.max_bins:
                    _edges = np.histogram_bin_edges(array, bins=self.option.max_bins).tolist()

                _edges = [_edges[0] - 0.1 * i for i in range(5, 0, -1)] + _edges + [_edges[-1] + 0.1 * i for i in range(1, 6)]
            else:
                _edges = np.arange(array_range[0] - _width * 6, array_range[1] + _width * 5, _width)
            h, edges = np.histogram(array, bins=_edges, density=True)
            h /= 100.
            return h, np.convolve(edges, [1, 1], 'valid') / 2

        array_range = np.min(array), np.max(array)
        width = self.option.histogram_bar_width

        return _get_hist(width)

    def _histogram_prob(self, array: np.ndarray, weights: np.ndarray):
        '''
        get histogram with probability weight
        '''
        # NOTE: for PSqueeze
        assert len(array.shape) == 2, f"histogram_prob receives array with shape {array.shape}"
        def _get_hist(_width):
            if _width == 'auto':
                _edges = np.histogram_bin_edges(array[:,1], bins='auto', range=array_range).tolist()
                if self.option.max_bins and len(_edges) > self.option.max_bins:
                    _edges = np.histogram_bin_edges(array[:,1], bins=self.option.max_bins).tolist()
                _edges = [_edges[0] - 0.1 * i for i in range(5, 0, -1)] + _edges + [_edges[-1] + 0.1 * i for i in range(1, 6)]
            else: _edges = np.arange(array_range[0] - _width * 6, array_range[1] + _width * 5, _width)
            h, edges = np.histogram(array, bins=_edges, weights=weights, density=True)
            h /= 100.
            return h, np.convolve(edges, [1, 1], 'valid') / 2
        array_range = np.min(array), np.max(array)
        width = self.option.histogram_bar_width
        return _get_hist(width)

    def _cluster(self, array, density_array: np.ndarray, bins):
        # NOTE: array is flattened
        def significant_greater(a, b):
            return (a - b) / (a + b) > 0.1

        order = 1
        extreme_max_indices = argrelextrema(
            density_array, comparator=lambda x, y: x > y,
            axis=0, order=order, mode='wrap')[0]
        extreme_min_indices = argrelextrema(
            density_array, comparator=lambda x, y: x <= y,
            axis=0, order=order, mode='wrap')[0]
        extreme_max_indices = list(filter(lambda x: density_array[x] > 0, extreme_max_indices))

        cluster_list = []
        boundaries = np.asarray([float('-inf')] + [bins[index] for index in extreme_min_indices] + [float('+inf')])
        if self.option.max_normal_deviation == 'auto':
            mu = np.mean(np.abs(array))
            max_normal = mu
            logger.debug(f"max normal {max_normal}")
            self.option.max_normal_deviation = max_normal
        for index in extreme_max_indices:
            left_boundary = boundaries[np.searchsorted(boundaries, bins[index], side='right') - 1]
            right_boundary = boundaries[np.searchsorted(boundaries, bins[index], side='left')]
            cluster_indices = np.where(
                np.logical_and(
                    array <= right_boundary,
                    array >= left_boundary,
                    )
            )[0]
            cluster = array[cluster_indices]
            mu = np.mean(np.abs(cluster))
            logger.debug(f"({left_boundary, right_boundary}, {mu})")
            if np.abs(mu) < self.option.max_normal_deviation or len(cluster) <= 0:
                continue
            cluster_list.append(cluster_indices)
        return cluster_list, (extreme_max_indices, extreme_min_indices)

    def __call__(self, array, weights=None):
        array = array.copy()
        if type(weights) == type(None):
            density_array, bins = self.density_estimation_func(array)
        else:
            density_array, bins = self.density_estimation_func(array, weights)
        density_array = np.copy(density_array)
        if self.option.cluster_smooth_window_size == "auto":
            window_size = max(np.count_nonzero(density_array > 0) // 10, 1)
            logger.debug(f"auto window size: {window_size} {np.count_nonzero(density_array > 0)}")
        else:
            window_size = self.option.cluster_smooth_window_size
        smoothed_density_array = smooth(density_array, window_size)
        array = array.ravel()
        clusters, extreme_indices = self._cluster(array, smoothed_density_array, bins)

        plot_kwargs = {}
        if self.option.debug:
            for cluster in clusters:
                left_boundary, right_boundary = np.min(array[cluster]), np.max(array[cluster])
                logger.debug(f"cluster: [{left_boundary}, {right_boundary}]")
            plot_kwargs = {
                "ds_values": array,
                "extreme_indices": extreme_indices,
                "bins": bins,
            }
        return clusters, plot_kwargs
