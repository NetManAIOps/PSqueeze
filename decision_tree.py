#!/usr/bin/env python3
#-*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import graphviz
import json
import os 
import re
import itertools
from functools import reduce
from sklearn import tree
from pathlib import Path

SCRIPT_DIR = Path(os.path.dirname(os.path.abspath(__file__)))
INPUT_DIR = SCRIPT_DIR
OUTPUT_DIR = SCRIPT_DIR
GT_DIR = SCRIPT_DIR / "data"


# load results
paths = [
    [INPUT_DIR],
    [f"psq_E{i+1}_result" for i in range(3)],
    [f"B{i}" for i in range(5)],
    ["/".join([f"B_cuboid_layer_{(i//3)+1}_n_ele_{(i%3)+1}"]*2)+".json" for i in range(9)],
]

paths = list(map(
    lambda x: reduce(lambda a, b: a/b, x),
    itertools.product(*paths)
))

df = pd.DataFrame.from_records(list(itertools.chain.from_iterable([
        sorted([dict(j["info_collect"],
                **{
                    "ep": j["ep"],
                    "n_rc": len(j["root_cause"].split(";")),
                    "n_ele": len(re.split("[&;]", j["root_cause"])),
                    "timestamp": j['timestamp'],
                }
            ) for j in json.load(open(i, "r"))],
            key=lambda x: x['timestamp']
        )
    for i in paths
])), exclude=["scores", "ranks", "n_eles", "layers"])


# load labels
paths = [
    [GT_DIR],
    [f"E{i+1}" for i in range(3)],
    [f"B{i}" for i in range(5)],
    [f"B_cuboid_layer_{(i//3)+1}_n_ele_{(i%3)+1}/injection_info.csv" for i in range(9)],
]

paths = list(map(
    lambda x: reduce(lambda a, b: a/b, x),
    itertools.product(*paths)
))

gts = [pd.read_csv(i) for i in paths]
labels = [
    (j["ex_rc_dim"].astype(str) != "nan").values.astype(int)
    for j in gts
]
idxs = [np.argsort(j["timestamp"]) for j in gts]
labels = [labels[i][idxs[i]] for i in range(len(labels))]
labels = list(itertools.chain.from_iterable(labels))

df["label"] = labels
df = df[["n_rc", "scores_min", "ranks_min", "ep", "label"]]

df = df.loc[~np.logical_or.reduce(
    [df[i].astype(str) == 'nan' for i in df.columns]
)]


# decision tree
clf = tree.DecisionTreeClassifier(
    criterion='gini',
    max_depth=3,
)
clf = clf.fit(df.iloc[:, 0:-1], df.label)

dot_data = tree.export_graphviz(
    clf, out_file=None,
    feature_names=df.columns[:-1],
    class_names=['No', 'Yes'],
    filled=True, rounded=True,
    special_characters=True,
    max_depth=3,
)

graph = graphviz.Source(dot_data)
graph.render('decision_tree.gv', cleanup=True, directory=OUTPUT_DIR, view=False)
