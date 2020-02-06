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

def parse_groundtruth(gt_str):
    def parse_one(s):
        return dict([i.split("=") for i in s])
    gt_list = [i.split('&') for i in gt_str.split(";")]
    gt_list = map(parse_one, gt_list)
    return list(gt_list)

def group_for(timestamp, ex_rc, input_path, output_dir):
    df = pd.read_csv(input_path / f"{timestamp}.csv", engine='python', dtype='str', delimiter=r"\s*,\s*")
    ex_rc = parse_groundtruth(ex_rc)
    ex_keys = reduce(lambda x,y: x.extend(y), map(lambda t: t.keys(), ex_rc))
    keys = sorted(list(set(df.columns.values) - set(ex_keys)))
    
    df.to_csv(output_dir / f"{timestamp}.csv", index=False, columns=keys)

def gen_exrc_data_for(dataset, n_elements, cuboid_layer, n_ex_rc=1, n_workers=20):
    if dataset.find("A_week") != -1:
        setting = f"new_dataset_{dataset}_n_elements_{n_elements}_layers_{cuboid_layer}"
        input_path = SCRIPT_DIR / "A" / setting
        output_dir = SCRIPT_DIR / "E" / "A" / setting
    else:
        assert dataset in [f"B{i}" for i in range(5)]
        setting = f"B_cuboid_layer_{cuboid_layer}_n_ele_{n_elements}"
        input_path = SCRIPT_DIR / dataset / setting
        output_dir = SCRIPT_DIR / "E" / dataset / setting
    injection_info = pd.read_csv(input_path / 'injection_info.csv', engine='c')
    rc = injection_info["set"].values
    timestamps = injection_info['timestamp'].values
    ex_rc = list(map(
        lambda x: ";".join(x.split(";")[:min(len(x.split(";")), int(n_ex_rc))]),
        rc
    ))
    injection_info["ex_rc"] = ex_rc
    make_dir(output_dir.resolve(), remove_flag=1)
    injection_info.to_csv(output_dir / "injection_info.csv", index=False)

    Parallel(n_jobs=int(n_workers), backend="multiprocessing", verbose=100)(
        delayed(group_for)(*i, input_path=input_path, output_dir=output_dir) for i in zip(timestamps, ex_rc))

if __name__ == "__main__":
    gen_exrc_data_for(*sys.argv[1:])
