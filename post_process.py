#!/usr/bin/env python3
#-*- coding: utf-8 -*-

import json 
import os
import itertools
import numpy as np 
from scipy.signal import argrelextrema

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

def post_process(results):
    scores_min = np.array([
        i["info_collect"]["scores_min"]
        for i in results if "scores_min" in i["info_collect"]
    ])
    scores_min = scores_min[~np.isnan(scores_min)]
    threshold = get_threshold(scores_min)
    for result in results:
        result["external_rc"] = False
        n_rc = len([i for i in result["root_cause"].split(";") if i])
        if n_rc == 0:
            result["external_rc"] = True
        else:
            if result["info_collect"]["scores_min"] < threshold:
                result["external_rc"] = True
            # else:
            #     result["external_rc"] = bool((result["ep"] < 0.65) and (n_rc > 3))

def get_threshold(array):
    density_array, edges = np.histogram(array, bins='auto', range=(-0.2, 1.2), density=True)
    density_array /= 100.
    bins = np.convolve(edges, [1, 1], 'valid') / 2

    window_size = max(np.count_nonzero(density_array > 0) // 10, 1)
    density_array = np.convolve(density_array, np.ones(window_size), mode="valid") / window_size
    density_array = np.concatenate([density_array[:window_size - 1], density_array])

    extreme_max_indices = argrelextrema(
        density_array, comparator=lambda x, y: x >= y,
        axis=0, order=1, mode='wrap')[0]
    extreme_min_indices = argrelextrema(
        density_array, comparator=lambda x, y: x <= y,
        axis=0, order=1, mode='wrap')[0]
    extreme_max_indices = list(filter(lambda x: density_array[x] > 0, extreme_max_indices))

    try:
        threshold = bins[extreme_min_indices[extreme_min_indices < extreme_max_indices[-1]][-1]]
    except IndexError as e:
        threshold = 0.8
    return threshold


def change_result():
    paths = [
        ["E1", "E2", "E3"],
        ["B0", "B1", "B2", "B3", "B4"],
        # [12, 34, 56, 78],
        [1, 2, 3],
        [1, 2, 3],
    ]
    paths = list(map(
        lambda x: f"{SCRIPT_DIR}/psq_{x[0]}_result/{x[1]}/B_cuboid_layer_{x[2]}_n_ele_{x[3]}/B_cuboid_layer_{x[2]}_n_ele_{x[3]}.json",
        # lambda x: f"{SCRIPT_DIR}/psq_{x[0]}_result/A"+f"/new_dataset_A_week_{x[1]}_n_elements_{x[2]}_layers_{x[3]}"*2+".json",
        itertools.product(*paths)
    ))
    for p in paths:
        with open(p, 'r') as f:
            results = json.load(f)
        post_process(results)
        with open(p, "w+") as f:
            json.dump(results, f, indent=4)

if __name__ == "__main__":
    change_result()
