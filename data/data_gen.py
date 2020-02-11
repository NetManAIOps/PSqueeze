#!/usr/bin/env python3
#-*- coding: utf-8 -*-
import os
import sys
import pandas as pd 
import numpy as np
import shutil
from joblib import Parallel, delayed
from pathlib import Path
from functools import reduce

SCRIPT_DIR = Path(os.path.dirname(os.path.abspath(__file__)))

def make_dir(path, remove_flag=0):
    if remove_flag == 1 and os.path.exists(path):
        shutil.rmtree(path)
    isExists = os.path.exists(path)
    if not isExists:
        os.makedirs(path)

def group_for(timestamp, ex_rc, input_path, output_dir):
    source_file = input_path / f"{timestamp}.csv"
    target_file = output_dir / f"{timestamp}.csv"
    if type(ex_rc) == type(None):
        shutil.copyfile(source_file, target_file)
        return

    df = pd.read_csv(source_file, engine='python', dtype='str', delimiter=r"\s*,\s*")
    keys = sorted(list(set(df.columns.values) - set(ex_rc)))
    
    df.to_csv(target_file, index=False, columns=keys)

def get_rc_dim(rc_str, n_ex_dim):
    rc_dim = list(map(lambda x: x.split("&"), rc_str.split(";")))
    rc_dim = [item for sublist in rc_dim for item in sublist]
    ret = list(set(map(lambda x: x.split("=")[0], rc_dim)))
    return ret[:min(n_ex_dim, len(ret))]

def get_rc_dim_wrapper(n_ex_dim):
    return lambda x: get_rc_dim(x, n_ex_dim)

def gen_exrc_data_for(dataset, n_elements, cuboid_layer, n_ex_dim=1, n_workers=20, ex_rc_ratio=0.5, only_info=True):
    if dataset.find("A_week") != -1:
        setting = f"new_dataset_{dataset}_n_elements_{n_elements}_layers_{cuboid_layer}"
        input_path = SCRIPT_DIR / "A" / setting
        output_dir = SCRIPT_DIR / f"E{n_ex_dim}" / "A" / setting
    else:
        assert dataset in [f"B{i}" for i in range(5)]
        setting = f"B_cuboid_layer_{cuboid_layer}_n_ele_{n_elements}"
        input_path = SCRIPT_DIR / dataset / setting
        output_dir = SCRIPT_DIR / f"E{n_ex_dim}" / dataset / setting

    injection_info = pd.read_csv(input_path / 'injection_info.csv', engine='c')
    rc = injection_info["set"].values
    timestamps = injection_info['timestamp'].values

    ex_rc = np.array(list(map(get_rc_dim_wrapper(int(n_ex_dim)), rc)), dtype=object)

    mask = np.arange(rc.size)
    np.random.shuffle(mask)
    mask = mask[:int(mask.size*ex_rc_ratio)]
    ex_rc[mask] = None
    def ex_rc_to_str(s):
        if type(s) == list: return ";".join(sorted(s))
        else: return None
    injection_info["ex_rc_dim"] = [ex_rc_to_str(i) for i in ex_rc.tolist()]

    make_dir(output_dir.resolve(), remove_flag=1)
    injection_info.to_csv(output_dir / "injection_info.csv", index=False)

    if not only_info:
        Parallel(n_jobs=int(n_workers), backend="multiprocessing", verbose=100)(
            delayed(group_for)(*i, input_path=input_path, output_dir=output_dir) for i in zip(timestamps, ex_rc))

if __name__ == "__main__":
    gen_exrc_data_for(*sys.argv[1:])
