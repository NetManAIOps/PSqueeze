import warnings
from concurrent.futures import ThreadPoolExecutor
from datetime import timedelta, datetime
from functools import partial
from pathlib import Path
from typing import Optional
import os
import click
import numpy as np
import pandas as pd
from dateutil.parser import parse as dt_parse
from loguru import logger


@click.command("Prepare inputs for the algorithms")
@click.option(
    '--anomaly-time', '-a', help="The minute to be analyzed", type=str, required=True,
)
@click.option(
    "--output-metric", type=str, help="the name of output metric"
)
@click.option(
    "--derived", type=str, default="", help="operands1,operation,operands2",
)
@click.option(
    '--timestamp-column', '-t', help="the column name of timestamp", default='timestamp', type=str
)
@click.option(
    '--metric-columns', '-m', help="a comma separated list of the column names of metrics", type=str,
    default="count,total_cost,total_proc,bussc_count,syssc_count,proc,cost,bussc_rate,syssc_rate"
)
@click.option(
    "--column-names", type=str,
    default="initnd,bc,apgrp,sndnd,rcvnd,ap,ret,svct,tc"
            ",count,total_cost,total_proc,bussc_count,syssc_count,proc,cost,bussc_rate,syssc_rate"
            ",timestamp",
    required=False,
    help="A comma separated list of the column names of the input files. If None, use the header line",
)
@click.option(
    "--window-length", type=int, default=5,
    help="how many timestamp before the anomaly time to use for prediction"
)
@click.option(
    '--granularity-minutes', type=int, default=1
)
@click.option("--output-dir", "-o", type=click.Path(exists=True, dir_okay=True, file_okay=False), default=".")
@click.option("--fill-na", is_flag=True)
@click.argument("input-files", nargs=-1, type=str)
def main(
        anomaly_time: str, timestamp_column: str, metric_columns: str, input_files: list[str],
        window_length: int, granularity_minutes: int, output_dir: Path, column_names: str or None,
        fill_na: bool, derived: str, output_metric: str
):
    anomaly_time: datetime = dt_parse(anomaly_time)
    output_dir: Path = Path(output_dir)
    if column_names is not None:
        with ThreadPoolExecutor() as executor:
            df = pd.concat(
                list(map(partial(pd.read_csv, index_col=None, names=column_names.split(",")), input_files)))
    else:
        with ThreadPoolExecutor() as executor:
            df = pd.concat(list(map(partial(pd.read_csv, index_col=None, header=0), input_files)))

    df['timestamp'] = df[timestamp_column].map(dt_parse).map(lambda x: x.replace(second=0, microsecond=0))
    if timestamp_column != "timestamp":
        df.drop(columns=[timestamp_column], inplace=True)
    metric_columns = metric_columns.split(",")
    logger.info(f"{metric_columns=}")
    attr_columns = list(set(df.columns) - set(metric_columns) - {'timestamp'})
    logger.info(f"{attr_columns=}")

    if output_metric not in metric_columns:
        assert derived != "", "derived should be set when metric is not in metric_columns"

    if derived != "":
        derived = tuple(derived.split(","))
        assert len(derived) == 3, derived
        assert derived[0] in metric_columns, derived
        assert derived[2] in metric_columns, derived
        assert derived[1] in {'divide', 'plus', 'minus', 'multiply'}, derived
    else:
        derived = None

    ac_df = df.drop(
        columns=['timestamp'] + metric_columns
    ).drop_duplicates()  # which contain all attribute combinations
    logger.info(f"#attribute combinations: {len(ac_df)}")

    granularity = timedelta(minutes=granularity_minutes)

    abnormal_dt_list = np.arange(anomaly_time - window_length * granularity, anomaly_time + granularity, granularity)
    filled_df_list = []
    for dt in abnormal_dt_list:
        __the_df = df[df.timestamp == dt]
        logger.info(f"#attribute combinations (assert no duplications) in {dt}: {len(__the_df)}")
        filled_df_list.append(
            pd.merge(
                ac_df,
                __the_df.drop(columns=['timestamp']),
                how='left', on=attr_columns, sort=True
            )
        )
        if fill_na:
            filled_df_list[-1].fillna(0, inplace=True)

    output_filename_prefix = f"{anomaly_time.timestamp():.0f}"
    if len(input_files) == 1:
        output_filename_prefix += f".{os.path.basename(input_files[0])}"
    output_filename_prefix += f".{output_metric}"
    if derived:
        ret_df = get_ret_df_for_metric(filled_df_list, derived[0], attr_columns)
        ret_df.to_csv(output_dir / f"{output_filename_prefix}_a.csv", index=False)
        ret_df = get_ret_df_for_metric(filled_df_list, derived[2], attr_columns)
        ret_df.to_csv(output_dir / f"{output_filename_prefix}_b.csv", index=False)
    else:
        assert output_metric in metric_columns
        # Fundamental measure
        ret_df = get_ret_df_for_metric(filled_df_list, output_metric, attr_columns)
        ret_df.to_csv(output_dir / f"{output_filename_prefix}.csv", index=False)


def get_ret_df_for_metric(filled_df_list: list[pd.DataFrame], metric: str, attr_columns: list[str]):
    ret_df = filled_df_list[-1].copy().fillna(0)
    ret_df['real'] = ret_df[metric]
    historical_data = np.vstack([_[metric].values for _ in filled_df_list[:-1]])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        ret_df['predict'] = np.nan_to_num(np.nanmean(historical_data, axis=0), nan=0)
    logger.info(f"|residual|_1={np.mean(np.abs(ret_df.real - ret_df.predict))}")
    logger.info(f"|residual|_2={np.sqrt(np.mean(np.square(ret_df.real - ret_df.predict)))}")
    logger.info(f"|residual|_inf={np.max(ret_df.real - ret_df.predict)}")
    ret_df.drop(columns=list(set(ret_df.columns) - set(attr_columns) - {'real', 'predict'}), inplace=True)
    return ret_df


if __name__ == '__main__':
    main()
