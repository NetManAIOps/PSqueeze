#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import json
import sys
import time
from functools import reduce
from pathlib import Path
from typing import Dict, List

import click
import numpy as np
import pandas as pd
# from run_apriori import run
from joblib import Parallel, delayed
from loguru import logger
# noinspection PyProtectedMember
from loguru._defaults import LOGURU_FORMAT

from Adtributor import Adtributor
from Apriori import AprioriRCA
from ImpAPTr import ImpAPTr
from MID import MID
from hotspotpy import HotSpot
from post_process import post_process
from squeeze import Squeeze, SqueezeOption
from utility import AC


@click.command('Runner')
@click.option("--name", default="", help="name of this setting")
@click.option("--input-path", "-i", help="will read data from {input_path}/{name}")
@click.option("--output-path", "-o", help="if {output_path} is a dir, save to {output_path}/{name}.json; \
otherwise save to {output_path}")
@click.option("--num-workers", "-j", default=1, help="num of processes")
@click.option(
    "--injection_info", default="",
    help="path to injection info, if empty, {input_path}/{name}/injection_info.csv will be used"
)
@click.option(
    "--algorithm", default="psqueeze", help="algorithm name",
)
@click.option("--derived", is_flag=True, help="means we should read {timestamp}.a.csv and {timestamp}.b.csv")
@click.option("--toint", is_flag=True, help="round measure values to integer")
@click.option("--n-ele", default=1, help="for ImpAPTr")
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
        sys.stdout, level="DEBUG",
        format="[<green>{time}</green>, <blue>{level}</blue>] <white>{message}</white>"
    )
    input_path = Path(input_path)
    assert input_path.exists(), f"{input_path} does not exist"
    logger.info(f"read data from {input_path / name}")

    output_path = Path(output_path)
    if output_path.is_dir():
        output_path = output_path / f"{name}.json"
    elif not output_path.exists():
        logger.info(f"create {output_path}")
        output_path.mkdir(parents=True)
        output_path = output_path / f"{name}.json"
    logger.info(f"save to {output_path}")

    injection_info: str = kwargs.pop('injection_info')
    if not injection_info:
        injection_info = str(input_path / name / "injection_info.csv")
    injection_info: pd.DataFrame = pd.read_csv(injection_info, engine='c')
    timestamps = sorted(injection_info['timestamp'])
    # timestamps = ['1451188800'] # NOTE: for debug
    injection_info: pd.DataFrame = injection_info.set_index(['timestamp'])

    derived: bool = kwargs.pop('derived')
    if not derived:
        results = Parallel(n_jobs=num_workers, backend="multiprocessing", verbose=100)(
            delayed(executor)(file_path, output_path.parent, injection_info, **kwargs)
            for file_path in map(lambda x: input_path / name / f'{x}.csv', timestamps))
    else:
        results = Parallel(n_jobs=num_workers, backend="multiprocessing", verbose=100)(
            delayed(executor_derived)(file_path_list, output_path.parent, injection_info, **kwargs)
            for file_path_list in map(
                lambda x: [input_path / name / f'{x}.a.csv', input_path / name / f'{x}.b.csv'],
                timestamps
            )
        )
    if kwargs['algorithm'] in {'psq', 'psqueeze'}:
        post_process(results)
    with open(str(output_path.resolve()), "w+") as f:
        json.dump(results, f, indent=4)
    logger.info(json.dumps(results, indent=4))


def load_data(file_path: Path, injection_info, toint=False):
    df = pd.read_csv(file_path.resolve(), dtype='str', delimiter=r",")
    if "ex_rc_dim" in injection_info.columns:
        ex_rc_dim = str(injection_info.loc[int(file_path.stem), "ex_rc_dim"])
        if not ex_rc_dim == "nan":
            # TODO? wrong. need groupby
            df = df.drop(ex_rc_dim.split(";"), axis=1)
    df['real'] = df['real'].astype(float)
    df['predict'] = df['predict'].astype(float)
    if toint:
        df['real'] = df['real'].astype(int)
        df['predict'] = df['predict'].astype(int)
    return df


def executor(file_path: Path, output_path: Path, injection_info: pd.DataFrame, **kwargs) -> Dict:
    from loguru import logger
    debug = kwargs.pop('debug', False)
    toint = kwargs.pop('toint', False)
    algorithm = kwargs.pop("algorithm", "psqueeze")
    logger.remove()
    logger.add(
        sys.stdout, level='INFO',
        format=f"<yellow>{file_path.name}</yellow> - {LOGURU_FORMAT}",
        backtrace=True
    )
    logger.info(f"running {algorithm} for {file_path}")
    df = load_data(file_path, injection_info, toint)
    try:
        timestamp = int(file_path.name.rstrip('.csv'))
    except ValueError:
        timestamp = file_path.name.rstrip('.csv')
        logger.warning(f"Unresolved timestamp: {timestamp}")
    tic = time.time()

    if algorithm.lower() in {'psqueeze', 'psq'}:
        model = get_model_psq(
            [df], op=lambda x: x,
            debug=debug, output_path=output_path, timestamp=timestamp, derived=False,
            **kwargs
        )
    elif algorithm.lower() in {'squeeze', 'sq'}:
        model = get_model_squeeze(
            [df], op=lambda x: x,
            debug=debug, output_path=output_path, timestamp=timestamp, derived=False,
            **kwargs
        )
    elif algorithm.lower() in {'mid'}:
        model = MID(data_list=[df], **kwargs)
    elif algorithm.lower() in {"impaptr", 'iap'}:
        model = ImpAPTr(data_list=[df], **kwargs)
    elif algorithm.lower() in {"apriori", 'apr'}:
        model = AprioriRCA(data_list=[df], **kwargs)
    elif algorithm.lower() in {"adt", "adtributor", 'rad', 'r-adtributor', 'recursive_adtributor'}:
        model = Adtributor([df], algorithm, derived=False)
    elif algorithm.lower() in {"hs", "hotspot"}:
        model = HotSpot(data_list=[df], max_steps=100,
                        ps_upper_threshold=0.95,
                        ps_lower_threshold=0.05,
                        )
    else:
        raise RuntimeError(f"unknown algorithm name: {algorithm=}")
    model.run()

    logger.info("\n" + model.report)
    try:
        root_cause = AC.batch_to_string(
            frozenset(reduce(lambda x, y: x.union(y), model.root_cause, set())))
    except IndexError:
        root_cause = ""

    toc = time.time()
    elapsed_time = toc - tic

    try:
        ground_truth = injection_info.loc[int(file_path.stem), 'set'] if int(file_path.stem) in injection_info.index else None
    except ValueError:
        ground_truth = None

    result = {
        'timestamp': timestamp,
        'elapsed_time': elapsed_time,
        'root_cause': root_cause,
        # 'ep': explanatory_power(model.derived_data, root_cause) if algorithm in {'psq', 'psqueeze'} else None,
        'ground_truth': ground_truth,
        'info_collect': model.info_collect,
    }

    return result


def executor_derived(file_path_list: List[Path], output_path: Path, injection_info: pd.DataFrame, **kwargs) -> Dict:
    from loguru import logger
    toint = kwargs.pop('toint', False)
    debug = kwargs.pop('debug', False)
    logger.remove()
    ts = file_path_list[0].name.rstrip('.a.csv')
    logger.add(
        sys.stdout, level='INFO',
        format=f"<yellow>{ts}</yellow> - {LOGURU_FORMAT}",
        backtrace=True
    )
    algorithm = kwargs.pop("algorithm", "psqueeze")
    logger.info(f"running {algorithm} for {ts}")
    dfa = load_data(file_path_list[0].resolve(), injection_info, toint=toint)
    dfb = load_data(file_path_list[1].resolve(), injection_info, toint=toint)
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

    if algorithm.lower() in {'psqueeze', 'psq'}:
        model = get_model_psq(
            [dfa, dfb], op=divide,
            debug=debug, output_path=output_path, timestamp=timestamp, derived=True,
            **kwargs
        )
    elif algorithm.lower() in {'squeeze', 'sq'}:
        model = get_model_squeeze(
            [dfa, dfb], op=divide,
            debug=debug, output_path=output_path, timestamp=timestamp, derived=True,
            **kwargs
        )
    elif algorithm.lower() in {'mid'}:
        model = MID(data_list=[dfa, dfb], op=divide, **kwargs)
    elif algorithm.lower() in {"impaptr", 'iap'}:
        model = ImpAPTr(data_list=[dfa, dfb], op=divide, **kwargs)
    elif algorithm.lower() in {"apriori", 'apr'}:
        model = AprioriRCA(data_list=[dfa, dfb], op=divide, **kwargs)
    elif algorithm.lower() in {"adt", "adtributor", 'rad', 'r-adtributor', 'recursive_adtributor'}:
        model = Adtributor([dfa, dfb], algorithm, derived=True)
    elif algorithm.lower() in {"hs", "hotspot"}:
        model = HotSpot(data_list=[dfa, dfb], op=divide, max_steps=100,
                        ps_upper_threshold=0.95,
                        ps_lower_threshold=0.05, )
    else:
        raise RuntimeError(f"unknown algorithm name: {algorithm=}")
    model.run()
    logger.info("\n" + model.report)
    try:
        root_cause = AC.batch_to_string(
            frozenset(reduce(lambda x, y: x.union(y), model.root_cause, set())))  # type:
    except IndexError:
        root_cause = ""

    toc = time.time()
    elapsed_time = toc - tic
    try:
        ground_truth = injection_info.loc[int(timestamp), 'set'] if int(timestamp) in injection_info.index else None
    except ValueError:
        ground_truth = None
    result = {
        'timestamp': timestamp,
        'elapsed_time': elapsed_time,
        'root_cause': root_cause,
        # 'ep': ep,
        'ground_truth': ground_truth,
        'info_collect': model.info_collect,
    }
    return result


def get_model_squeeze(data_list, op, debug: bool, output_path: Path, timestamp: int, derived: bool, **kwargs):
    option = SqueezeOption(
        psqueeze=False,
        debug=debug,
        fig_save_path=f"{output_path.resolve()}/{timestamp}" + "{suffix}" + ".pdf",
        **kwargs,
    )
    return Squeeze(data_list=data_list, op=op, option=option)


def get_model_psq(data_list, op, debug: bool, output_path: Path, timestamp: int, derived: bool, **kwargs):
    option = SqueezeOption(
        psqueeze=True,
        debug=debug,
        fig_save_path=f"{output_path.resolve()}/{timestamp}" + "{suffix}" + ".pdf",
        density_estimation_method="histogram_prob",
        bias=1 if not derived else 0,
        **kwargs,
    )
    return Squeeze(data_list=data_list, op=op, option=option)


def explanatory_power(df, rc_str):
    if not rc_str: return 0.0
    delta = df["real"].values.sum() - df["predict"].values.sum()
    rc_list = [dict(map(lambda x: x.split("="), i.split("&"))) for i in rc_str.split(";")]
    cover = df.loc[np.logical_or.reduce([
        np.logical_and.reduce([df[k] == v for k, v in i.items()])
        for i in rc_list
    ])]
    return (cover["real"].values.sum() - cover["predict"].values.sum()) / delta


if __name__ == '__main__':
    main()
