#!/usr/bin/env python3
#-*- coding: utf-8 -*-
import os
import sys
import pandas as pd 
import numpy as np
import shutil
import matplotlib.pyplot as plt
import json
from joblib import Parallel, delayed
from pathlib import Path

SCRIPT_DIR = Path(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_DIR = SCRIPT_DIR / "result_inspect"
INPUT_DIR = SCRIPT_DIR / "debug"
DATA_DIR = SCRIPT_DIR / "data" / "E"
DEBUG = True

def make_dir(path, remove_flag=0):
    if remove_flag == 1 and os.path.exists(path):
        shutil.rmtree(path)
    isExists = os.path.exists(path)
    if not isExists:
        os.makedirs(path)

def result_inspect(name, fig1, fig2, num):
    n_elements = 1+(num//3)
    cuboid_layer = 1+(num%3)
    if name.find("A_week") != -1:
        setting = f"new_dataset_{name}_n_elements_{n_elements}_layers_{cuboid_layer}"
        input_path = INPUT_DIR / "A" / setting / f"{setting}.json"
        injection_info = DATA_DIR / "A" / setting / "injection_info.csv"
    else:
        assert name in [f"B{i}" for i in range(5)]
        setting = f"B_cuboid_layer_{cuboid_layer}_n_ele_{n_elements}"
        input_path = INPUT_DIR / name / setting / f"{setting}.json"
        injection_info = DATA_DIR / name / setting / "injection_info.csv"

    with open(input_path.resolve(), "r") as f:
        result = pd.DataFrame.from_dict(json.load(f))
    result.set_index(["timestamp"], inplace=True)
    
    injection_info = pd.read_csv(injection_info.resolve(), engine='c')
    injection_info.set_index(["timestamp"], inplace=True)

    result = pd.concat([result, injection_info], axis=1, join="inner") 
    result["ex_rc_dim"] = result["ex_rc_dim"].astype(str)
    result_non_exrc = result.loc[result["ex_rc_dim"] == "nan"]
    result_with_exrc = result.loc[~(result["ex_rc_dim"] == "nan")]

    x1 = [len(i.split(";")) for i in result_non_exrc.root_cause.values]
    x2 = [len(i.split(";")) for i in result_with_exrc.root_cause.values]
    kwargs = {
        "alpha": 0.5,
        "bins": np.arange(0, 2+max(max(x1), max(x2)))-0.5,
    }
    ax = fig1.add_subplot(f"33{num+1}")
    ax.set_title(f"{name} ({n_elements}, {cuboid_layer})")
    ax.set_ylabel("num")
    ax.set_xlabel("n_root_cause")
    ax.hist(x1, **kwargs, color='g', label='non_ex_rc')
    ax.hist(x2, **kwargs, color='r', label='ex_rc')
    ax.legend()

    ax = fig2.add_subplot(f"33{num+1}")
    ax.set_title(f"{name} ({n_elements}, {cuboid_layer})")
    ax.set_ylabel("num")
    ax.set_xlabel("explanatory power")
    ax.hist(result_non_exrc.ep.values, alpha=0.5, color='g', label='non_ex_rc')
    ax.hist(result_with_exrc.ep.values, alpha=0.5, color='r', label='ex_rc')
    ax.legend()

def get_figure(name):
    plt.clf()
    fig1 = plt.figure(figsize=(18,18))
    fig2 = plt.figure(figsize=(18,18))
    for i in range(9):
        result_inspect(name, fig1, fig2, i)
    make_dir(OUTPUT_DIR, remove_flag=1)
    fig1.savefig(OUTPUT_DIR / f"{name}_n_root_cause.png")
    fig2.savefig(OUTPUT_DIR / f"{name}_ep.png")

if __name__ == "__main__":
    tasks = [f"B{i}" for i in range(5)]
    Parallel(n_jobs=len(tasks), backend="multiprocessing", verbose=100)(
        delayed(get_figure)(task) for task in tasks)


