#!/usr/bin/env python3
#-*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import json
import os 
import re
import itertools
from pathlib import Path
from sklearn import tree
from post_process import get_threshold

SCRIPT_DIR = Path(os.path.dirname(os.path.abspath(__file__)))
INPUT_DIR = SCRIPT_DIR
OUTPUT_DIR = SCRIPT_DIR
GT_DIR = SCRIPT_DIR / "data"

Es = [f"E{i}" for i in range(1,4)]
Bs = [f"B{i}" for i in range(5)]
As = [f"{2*i+1}{2*i+2}" for i in range(4)]

fig = plt.figure(figsize=(18,18))

# B dataset
for e in Es:
    for b in Bs:
        # data
        settings = list(itertools.product([1,2,3], [1,2,3]))
        paths = map(
            lambda x: INPUT_DIR / f"psq_{e}_result" / b / \
                ("/".join([f"B_cuboid_layer_{x[0]}_n_ele_{x[1]}"]*2)+".json"),
            settings
        )
        dfs_map = dict(zip(
            settings,
            map(
                lambda x: pd.DataFrame.from_records([dict(i["info_collect"], **{
                        "ep": i["ep"],
                        "n_rc": len(i["root_cause"].split(";")),
                        "n_ele": len(re.split("[&;]", i["root_cause"])),
                        "timestamp": i['timestamp'],
                    })
                    for i in json.load(open(x, 'r'))]
                ).sort_values(by="timestamp"),
                paths
            )
        ))

        # plot
        fig.clf()
        fig.suptitle(f"{e}-{b}")
        for idx, data in enumerate(dfs_map.items()):
            setting, df = data

            label_path = GT_DIR / e / b / \
                f"B_cuboid_layer_{setting[0]}_n_ele_{setting[1]}" / "injection_info.csv"
            df["label"] = (pd.read_csv(label_path).sort_values(by="timestamp")["ex_rc_dim"].astype(str) != "nan").values.astype(int)
            df = df[["scores_min", "label"]]
            df = df.loc[~np.logical_or.reduce(
                [df[i].astype(str) == 'nan' for i in df.columns]
            )]
            clf = tree.DecisionTreeClassifier(
                criterion='gini',
                max_depth=1,
            )
            clf = clf.fit(df.iloc[:, 0:-1], df.label)
            x = df.scores_min.values.astype(float)
            ax = fig.add_subplot(f"33{idx+1}")
            ax.set_title(f"{setting}")
            ax.set_xlim((-0.2, 1.2))
            ax.hist(x, bins=50)
            ax.axvline(clf.tree_.threshold[0], color="red", linestyle="--", linewidth=2, alpha=0.5)
            ax.set_xticks([clf.tree_.threshold[0], 0.00, 1.00])

            th = get_threshold(df["scores_min"])
            ax.axvline(th, color="g", linestyle="-", linewidth=2, alpha=0.8)

        fig.tight_layout()
        fig.savefig(OUTPUT_DIR / f"{e}_{b}_scores_dist.png", bbox_inches='tight')


# A dataset
for e in Es:
    for a in As:
        # data
        settings = list(itertools.product([1,2,3], [1,2,3]))
        paths = map(
            lambda x: INPUT_DIR / f"psq_{e}_result" / "A" / \
                ("/".join([f"new_dataset_A_week_{a}_n_elements_{x[0]}_layers_{x[1]}"]*2)+".json"),
            settings
        )
        dfs_map = dict(zip(
            settings,
            map(
                lambda x: pd.DataFrame.from_records([dict(i["info_collect"], **{
                        "ep": i["ep"],
                        "n_rc": len(i["root_cause"].split(";")),
                        "n_ele": len(re.split("[&;]", i["root_cause"])),
                        "timestamp": i['timestamp'],
                    })
                    for i in json.load(open(x, 'r'))]
                ).sort_values(by="timestamp"),
                paths
            )
        ))

        # plot
        fig.clf()
        fig.suptitle(f"{e}-{a}")
        for idx, data in enumerate(dfs_map.items()):
            setting, df = data

            label_path = GT_DIR / e / "A" / \
                f"new_dataset_A_week_{a}_n_elements_{setting[0]}_layers_{setting[1]}" / \
                "injection_info.csv"
            df["label"] = (pd.read_csv(label_path).sort_values(by="timestamp")["ex_rc_dim"].astype(str) != "nan").values.astype(int)
            df = df[["scores_min", "label"]]
            df = df.loc[~np.logical_or.reduce(
                [df[i].astype(str) == 'nan' for i in df.columns]
            )]
            clf = tree.DecisionTreeClassifier(
                criterion='gini',
                max_depth=1,
            )
            clf = clf.fit(df.iloc[:, 0:-1], df.label)

            x = df.scores_min.values.astype(float)
            ax = fig.add_subplot(f"33{idx+1}")
            ax.set_title(f"{setting}")
            ax.set_xlim((-0.2, 1.2))
            ax.hist(x, bins=50)
            ax.axvline(clf.tree_.threshold[0], color="red", linestyle="--", linewidth=2, alpha=0.5)
            ax.set_xticks([clf.tree_.threshold[0], 0.00, 1.00])

            th = get_threshold(df["scores_min"])
            ax.axvline(th, color="g", linestyle="-", linewidth=2, alpha=0.8)

        fig.tight_layout()
        fig.savefig(OUTPUT_DIR / f"{e}_A_week_{a}_scores_dist.png", bbox_inches='tight')
