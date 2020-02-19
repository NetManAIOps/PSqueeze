#!/usr/bin/env python3
#-*- coding: utf-8 -*-

import json 
import os
import itertools

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

def post_process(result):
    result["external_rc"] = False
    n_rc = len([i for i in result["root_cause"].split(";") if i])
    if n_rc == 0:
        result["external_rc"] = True
    else:
        assert "scores_min" in result["info_collect"], f"{result}"
        if result["info_collect"]["scores_min"] < 0.75:
            result["external_rc"] = True
        else:
            result["external_rc"] = bool((result["ep"] < 0.65) and (n_rc > 3))

def change_result():
    paths = [
        ["E1", "E2", "E3"],
        ["B0", "B1", "B2", "B3", "B4"],
        [1, 2, 3],
        [1, 2, 3],
    ]
    paths = list(map(
        lambda x: f"{SCRIPT_DIR}/psq_{x[0]}_result/{x[1]}/B_cuboid_layer_{x[2]}_n_ele_{x[3]}/B_cuboid_layer_{x[2]}_n_ele_{x[3]}.json",
        itertools.product(*paths)
    ))
    for p in paths:
        with open(p, 'r') as f:
            results = json.load(f)
        for r in results:
            post_process(r)
        with open(p, "w+") as f:
            json.dump(results, f, indent=4)

if __name__ == "__main__":
    change_result()
