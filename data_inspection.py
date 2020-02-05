#!/usr/bin/env python3
#-*- coding: utf-8 -*-
import os
import sys
import pandas as pd 
import numpy as np
import shutil
from joblib import Parallel, delayed
from pathlib import Path

SCRIPT_DIR = Path(os.path.dirname(os.path.abspath(__file__)))
DEBUG = False

def debugP(*args, **kwargs):
    if DEBUG: print(*args, **kwargs)

def print_line():
    debugP(f"================================")

def make_dir(path, remove_flag=0):
    if remove_flag == 1 and os.path.exists(path):
        shutil.rmtree(path)
    isExists = os.path.exists(path)
    if not isExists:
        os.makedirs(path)

def get_dataframe(file_path):
    debugP(f"read from {file_path}")
    df = pd.read_csv(file_path.resolve(), engine='python', dtype='str', delimiter=r"\s*,\s*")
    df['real'] = df['real'].astype(float)
    df['predict'] = df['predict'].astype(float)
    df = df.loc[(df['real'] > 0.0) & (df['predict'] > 0.0)]
    debugP(f"dataframe shape: {df.shape}")
    print_line()
    return df

def get_attributes(df):
    keys = sorted(list(set(df.columns.values) - {"predict", "real"}))
    attributes = dict(zip(keys, [sorted(list(set(df[k].values))) for k in keys]))
    for k, v in attributes.items():
        debugP(f"{k}: {v}")
    debugP()

    leaf_info = (df.groupby(keys).agg([("count", "count"), ("average", "mean"), ("max", "max"), ("min", "min")]))
    leaf_info.columns = leaf_info.columns.map('_'.join)
    leaf_info = leaf_info.reset_index()

    debugP(leaf_info)
    print_line()
    return attributes, leaf_info

def executor(timestamp, input_path, output_dir):
    output_path = output_dir / f"{timestamp}.csv"
    df = get_dataframe(input_path / f'{timestamp}.csv')
    attrs, leaf_info = get_attributes(df)
    leaf_info.to_csv(output_path.resolve(), index=False)
    print(f"\t{timestamp} finished.")

def data_inspect(name, n_elements, cuboid_layer, num_workers=1):
    if name.find("A_week") != -1:
        setting = f"new_dataset_{name}_n_elements_{n_elements}_layers_{cuboid_layer}"
        input_path = SCRIPT_DIR / "data" / "A" / setting
        output_dir = SCRIPT_DIR / "debug" / "data_info" / "A" / setting
    else:
        assert name in [f"B{i}" for i in range(5)]
        setting = f"B_cuboid_layer_{cuboid_layer}_n_ele_{n_elements}"
        input_path = SCRIPT_DIR / "data" / name / setting
        output_dir = SCRIPT_DIR / "debug" / "data_info" / name / setting
    injection_info = pd.read_csv(input_path / 'injection_info.csv', engine='c')
    timestamps = sorted(injection_info['timestamp'])
    make_dir(output_dir.resolve(), remove_flag=1)
    print("inspect for {name} {setting}")

    Parallel(n_jobs=int(num_workers), backend="multiprocessing", verbose=100)(
        delayed(executor)(timestamp, input_path, output_dir) for timestamp in timestamps)

if __name__ == "__main__":
    data_inspect(*sys.argv[1:])
