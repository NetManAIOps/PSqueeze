#!/usr/bin/env python3
#-*- coding: utf-8 -*-
import os
import sys
import pandas as pd 
import numpy as np
import shutil
import matplotlib.pyplot as plt
from joblib import Parallel, delayed
from pathlib import Path

SCRIPT_DIR = Path(os.path.dirname(os.path.abspath(__file__)))
DEBUG = True

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
    df = df.loc[(df['real'] > 0.0) | (df['predict'] > 0.0)]
    debugP(f"dataframe shape: {df.shape}")
    print_line()
    return df

def get_attributes(df):
    keys = sorted(list(set(df.columns.values) - {"predict", "real"}))
    attributes = dict(zip(keys, [sorted(list(set(df[k].values))) for k in keys]))
    for k, v in attributes.items():
        debugP(f"{k}: {v}")

    leaf_info = (df.groupby(keys).agg([("count", "count"), ("average", "mean"), ("max", "max"), ("min", "min")]))
    leaf_info.columns = leaf_info.columns.map('_'.join)
    leaf_info = leaf_info.reset_index()

    return attributes, leaf_info

def parse_groundtruth(gt_str):
    def parse_one(s):
        return dict([i.split("=") for i in s])
    gt_list = [i.split('&') for i in gt_str.split(";")]
    gt_list = map(parse_one, gt_list)
    return list(gt_list)

def get_groundtruth(injection_info):
    return lambda t: parse_groundtruth(
            injection_info.loc[
                injection_info['timestamp'] == int(t)
            ].set.values[0]
        )

def split_dataframe(leaf, gt):
    abnormal = []
    normal = leaf
    rc = []
    for one in gt:
        abn = leaf
        rc.append("&".join(["=".join(i) for i in one.items()]))
        for k, v in one.items():
            abn = abn.loc[abn[k] == v]
        abnormal.append(abn)
        normal = normal[~normal.index.isin(abn.index)]
    return normal, abnormal, rc

def get_report(df, report, dataname):
    report['data'].append(dataname)
    debugP(f"total: {df.shape[0]}")
    report["total"].append(df.shape[0])

    delta = 5
    small_df = df.loc[df['real_average'] <= delta]
    debugP(f"small part total: {small_df.shape[0]}")
    report["v_small"].append(small_df.shape[0])

    this = small_df.loc[
        small_df['real_average'] == small_df['predict_average']
    ]
    debugP(f"[0, 0]: {this.shape[0]}")
    report["[0, 0]"].append(this.shape[0])

    for i in range(delta):
        this = small_df.loc[
            (abs(small_df['real_average']-small_df['predict_average']) > i)
            &
            (abs(small_df['real_average']-small_df['predict_average']) <= i+1)
        ]
        debugP(f"({i}, {i+1}]: {this.shape[0]}")
        report[f"({i}, {i+1}]"].append(this.shape[0])

    this = small_df.loc[
        abs(small_df['real_average']-small_df['predict_average']) > delta
    ]
    debugP(f"({delta}, inf]: {this.shape[0]}")
    report["(5, inf]"].append(this.shape[0])
    print_line()

def plot_histogram(ax, xs, labels, key, name, xlim, ylim):
    n_bins = 1000
    colors = ["g", "b", "r", "y"]
    kwargs = dict(bins=n_bins, alpha=0.5)

    for k in range(len(xs)):
        ax.hist(xs[k], **kwargs, color=colors[k], label=labels[k])
    ax.set_title(f"{key}")
    plt.ylabel("Density")
    plt.xlabel(key)
    plt.ylim(*ylim)
    plt.xlim(*xlim)
    plt.legend()

def plot_scatter(ax, xs, ys, labels, xlabel, ylabel, xlim, ylim):
    colors = ["g", "b", "r", "y"]
    kwargs = dict(marker='.', alpha=0.5)

    for k in range(len(xs)):
        ax.scatter(x=xs[k], y=ys[k], color=colors[k], label=labels[k], **kwargs)
    ax.set_title(f'{xlabel} on {ylabel}')
    plt.ylabel(ylabel)
    plt.xlabel(xlabel)
    plt.ylim(*ylim)
    plt.xlim(*xlim)
    plt.legend()

def get_figure(dfs, labels, timestamp):
    plt.clf()
    fig = plt.figure(figsize=(18,12))

    ax1 = fig.add_subplot(221)
    xs = list(map(
        lambda df: np.abs(df["real_average"].values-df["predict_average"].values),
        dfs
    ))
    plot_histogram(ax1, xs, labels, "predict_error", timestamp, (0, 1000), (0, 1000))

    ax2 = fig.add_subplot(222)
    ys = list(map(
        lambda df: df["real_average"].values-df["predict_average"].values,
        dfs
    ))
    xs = list(map(
        lambda df: df["real_average"].values,
        dfs
    ))
    plot_scatter(ax2, xs, ys, labels, "real_measure", "predict_error", (0, 1000), (-100, 100))

    ax3 = fig.add_subplot(223)
    xs = list(map(
        lambda df: df["predict_average"].values,
        dfs
    ))
    plot_histogram(ax3, xs, labels, "predict_measure", timestamp, (0, 1000), (0, 1000))

    ax4 = fig.add_subplot(224)
    xs = list(map(
        lambda df: df["real_average"].values,
        dfs
    ))
    plot_histogram(ax4, xs, labels, "real_measure", timestamp, (0, 1000), (0, 1000)) 

    path = SCRIPT_DIR / f"{timestamp}.png"
    plt.savefig(path.resolve())

def executor(timestamp, input_path, output_dir, groundtruth):
    output_path = output_dir / f"{timestamp}.csv"
    df = get_dataframe(input_path / f'{timestamp}.csv')
    attrs, leaf_info = get_attributes(df)
    normal, abnormal, rc = split_dataframe(leaf_info, groundtruth)
    print_line()

    report = {
        "data": [],
        "total": [],
        "v_small": [],
        "[0, 0]": [],
        "(0, 1]": [],
        "(1, 2]": [],
        "(2, 3]": [],
        "(3, 4]": [],
        "(4, 5]": [],
        "(5, inf]": [],
    }
    get_report(normal, report, "normal")
    for i in range(len(abnormal)):
        get_report(abnormal[i], report, rc[i])
    report = pd.DataFrame.from_dict(report)
    debugP(report)
    # report.to_csv(output_path.resolve(), index=False)

    get_figure([normal]+abnormal, ["normal"]+rc, timestamp)
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
    groundtruth = get_groundtruth(injection_info)
    # timestamps = sorted(injection_info['timestamp'])
    timestamps = ['1538349900']
    # make_dir(output_dir.resolve(), remove_flag=1)
    print("inspect for {name} {setting}")

    Parallel(n_jobs=int(num_workers), backend="multiprocessing", verbose=100)(
        delayed(executor)(timestamp, input_path, output_dir, groundtruth(timestamp)) for timestamp in timestamps)

if __name__ == "__main__":
    data_inspect(*sys.argv[1:])
