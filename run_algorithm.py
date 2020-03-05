#!/usr/bin/env python3
#-*- coding: utf-8 -*-
import sys
import time
from pathlib import Path
import click
from functools import reduce
from typing import Dict, List
import json
import numpy as np
# from run_apriori import run
from joblib import Parallel, delayed
# noinspection PyProtectedMember
from loguru._defaults import LOGURU_FORMAT

from utility import AC, AttributeCombination
from post_process import post_process
from squeeze import Squeeze, SqueezeOption
import pandas as pd
from loguru import logger

import os


@click.command('Runner')
@click.option("--name", default="", help="name of this setting")
@click.option("--input-path", help="will read data from {input_path}/{name}")
@click.option("--output-path", help="if {output_path} is a dir, save to {output_path}/{name}.json; \
otherwise save to {output_path}")
@click.option("--num-workers", default=1, help="num of processes")
@click.option("--injection_info", default="", help="path to injection info")
@click.option("--derived", is_flag=True, help="means we should read {timestamp}.a.csv and {timestamp}.b.csv")
@click.option("--toint", is_flag=True, help="round measure values to integer")
def main(name, input_path, output_path, num_workers, **kwargs):
    """
    :param name:
    :param input_path:
    :param output_path:
    :param num_workers:
    :param kwargs:
    :return:
    """
    logger.remove()
    logger.add(
        sys.stdout, level="INFO",
        format="[<green>{time}</green>, <blue>{level}</blue>] <white>{message}</white>"
    )
    dervied = kwargs.pop('derived')
    injection_info = kwargs.pop('injection_info')

    input_path = Path(input_path)
    assert input_path.exists(), f"{input_path} does not exist"
    output_path = Path(output_path)
    logger.info(f"read data from {input_path / name}")
    if output_path.is_dir():
        output_path = output_path / f"{name}.json"
    elif not output_path.exists():
        logger.info(f"create {output_path}")
        output_path.mkdir()
        output_path = output_path / f"{name}.json"
    logger.info(f"save to {output_path}")
    if not injection_info: injection_info = input_path / name / 'injection_info.csv'
    injection_info = pd.read_csv(injection_info, engine='c')
    timestamps = sorted(injection_info['timestamp'])
    # timestamps = ['1451118000'] # NOTE: for debug
    injection_info = injection_info.set_index(['timestamp'])
    if not dervied:
        results = Parallel(n_jobs=num_workers, backend="multiprocessing", verbose=100)(
            delayed(executor)(file_path, output_path.parent, injection_info, **kwargs)
            for file_path in map(lambda x: input_path / name / f'{x}.csv', timestamps))
    else:
        results = Parallel(n_jobs=num_workers, backend="multiprocessing", verbose=100)(
            delayed(executor_derived)(file_path_list, output_path.parent, **kwargs)
            for file_path_list in map(
                lambda x: [input_path / name / f'{x}.a.csv', input_path / name / f'{x}.b.csv'],
                timestamps
            )
        )
    post_process(results)
    with open(str(output_path.resolve()), "w+") as f:
        json.dump(results, f, indent=4)
    logger.info(results)


def load_data(file_path: Path, injection_info, toint=False):
    df = pd.read_csv(file_path.resolve(), engine='python', dtype='str', delimiter=r"\s*,\s*")
    if "ex_rc_dim" in injection_info.columns:
        ex_rc_dim = str(injection_info.loc[int(file_path.stem), "ex_rc_dim"])
        if not ex_rc_dim == "nan":
            df = df.drop(ex_rc_dim.split(";"), axis=1)
    df['real'] = df['real'].astype(float)
    df['predict'] = df['predict'].astype(float)
    if toint:
        df['real'] = df['real'].astype(int)
        df['predict'] = df['predict'].astype(int)
    return df


def executor(file_path: Path, output_path: Path, injection_info, **kwargs) -> Dict:
    debug = kwargs.pop('debug', False)
    toint = kwargs.pop('toint', False)
    logger.remove()
    logger.add(
        sys.stdout, level='DEBUG',
        format=f"<yellow>{file_path.name}</yellow> - {LOGURU_FORMAT}",
        backtrace=True
    )
    logger.info(f"running squeeze for {file_path}")
    df = load_data(file_path, injection_info, toint)
    try:
        timestamp = int(file_path.name.rstrip('.csv'))
    except ValueError:
        timestamp = file_path.name.rstrip('.csv')
        logger.warning(f"Unresolved timestamp: {timestamp}")
    tic = time.time()

    psqueezeOption = SqueezeOption(
        psqueeze = True,
        debug=debug,
        fig_save_path=f"{output_path.resolve()}/{timestamp}" + "{suffix}" + ".pdf",
        density_estimation_method="histogram_prob", 
        bias=1,
        score_measure="ps", # NOTE: "ps" or "ji"
        dis_norm=True,
        # max_bins=100, # NOTE here
        **kwargs,
    )
    squeezeOption = SqueezeOption(
        psqueeze = False,
        debug=debug,
        fig_save_path=f"{output_path.resolve()}/{timestamp}" + "{suffix}" + ".pdf",
        score_measure="ps", # NOTE: "ps" or "ji"
        dis_norm=True,
        # max_bins=100, # NOTE here
        # histogram_bar_width = 0.01,
        **kwargs,
    )

    model = Squeeze(
        data_list=[df],
        op=lambda x: x,
        option=psqueezeOption,
    )
    model.run()

    logger.info("\n" + model.report)
    try:
        root_cause = AC.batch_to_string(
            frozenset(reduce(lambda x, y: x.union(y), model.root_cause, set())))  # type:
    except IndexError:
        root_cause = ""

    toc = time.time()
    elapsed_time = toc - tic

    ep = explanatory_power(model.derived_data, root_cause)
    result = {
        'timestamp': timestamp,
        'elapsed_time': elapsed_time,
        'root_cause': root_cause,
        'ep': ep, 
        'info_collect': model.info_collect,
    }

    return result

def executor_derived(file_path_list: List[Path], output_path: Path, **kwargs) -> Dict:
    debug = kwargs.pop('debug', False)
    logger.remove()
    ts = file_path_list[0].name.rstrip('.a.csv')
    logger.add(
        sys.stdout, level='DEBUG',
        format=f"<yellow>{ts}</yellow> - {LOGURU_FORMAT}",
        backtrace=True
    )
    logger.info(f"running squeeze for {ts}")
    dfa = pd.read_csv(file_path_list[0].resolve(), engine='python', dtype='str', delimiter=r"\s*,\s*")
    dfa['real'] = dfa['real'].astype(float)
    dfa['predict'] = dfa['predict'].astype(float)
    dfb = pd.read_csv(file_path_list[1].resolve(), engine='python', dtype='str', delimiter=r"\s*,\s*")
    dfb['real'] = dfb['real'].astype(float)
    dfb['predict'] = dfb['predict'].astype(float)
    zero_index = (dfa.real == 0) & (dfa.predict == 0) & (dfb.real == 0) & (dfb.predict == 0)
    dfa = dfa[~zero_index]
    dfb = dfb[~zero_index]
    try:
        timestamp = int(ts)
    except ValueError:
        timestamp = ts
        logger.warning(f"Unresolved timestamp: {timestamp}")
    tic = time.time()

    divide = lambda x, y: np.divide(x, y, out=np.zeros_like(x), where=y != 0)
    psqueezeOption = SqueezeOption(
        psqueeze = True,
        debug=debug,
        fig_save_path=f"{output_path.resolve()}/{timestamp}" + "{suffix}" + ".pdf",
        density_estimation_method="histogram_prob", 
        # max_bins=100,
        enable_filter=True,
        **kwargs,
    )
    squeezeOption = SqueezeOption(
        psqueeze = False,
        debug=debug,
        fig_save_path=f"{output_path.resolve()}/{timestamp}" + "{suffix}" + ".pdf",
        enable_filter=True,
        **kwargs,
    )

    model = Squeeze(
        data_list=[dfa, dfb],
        op=divide,
        option=squeezeOption
    )
    model.run()
    logger.info("\n" + model.report)
    try:
        root_cause = AC.batch_to_string(
            frozenset(reduce(lambda x, y: x.union(y), model.root_cause, set())))  # type:
    except IndexError:
        root_cause = ""

    toc = time.time()
    elapsed_time = toc - tic
    return {
        'timestamp': timestamp,
        'elapsed_time': elapsed_time,
        'root_cause': root_cause,
        'external_rc': False,
    }

def explanatory_power(df, rc_str):
    if not rc_str: return 0.0
    delta = df["real"].values.sum() - df["predict"].values.sum()
    rc_list = [dict(map(lambda x: x.split("="), i.split("&"))) for i in rc_str.split(";")]
    cover = df.loc[np.logical_or.reduce([
        np.logical_and.reduce([df[k] == v for k,v in i.items()])
        for i in rc_list
    ])]
    return (cover["real"].values.sum() - cover["predict"].values.sum()) / delta

if __name__ == '__main__':
    main()
