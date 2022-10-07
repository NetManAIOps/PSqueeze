import os
import warnings
from datetime import timedelta, datetime
from functools import partial
from pathlib import Path
from typing import Optional
from pytz import timezone
import click
import numpy as np
import pandas as pd
from dateutil.parser import parse as dt_parse
from loguru import logger


TZ = timezone("Asia/Shanghai")


@click.command("Prepare inputs for the algorithms")
@click.option(
    '--anomaly-time', '-a', help="The minute to be analyzed", type=str, default="",
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
    "--timestamp-column-type", type=str, default="datetime", help="datetime or timestamp",
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
    "--window-length", type=int, default=20,
    help="how many timestamp before the anomaly time to use for prediction"
)
@click.option(
    '--granularity-minutes', type=int, default=1
)
@click.option(
    '--given-predict', type=float, default=None, help="Use it as the predict rather than calculated value by averaging"
)
@click.option("--query", type=str, default="", help="pd.DataFrame.query")
@click.option("--output-dir", "-o", type=click.Path(exists=True, dir_okay=True, file_okay=False), default=".")
@click.option("--fill-na", is_flag=True)
@click.option("--extra-eval-columns", type=str, default="", help="e.g., total_cost=cost * count;")
@click.option("--read-pickle", is_flag=True)
@click.option("--drop-columns", type=str, default="")
@click.argument("input-files", nargs=-1, type=str)
def main(
        anomaly_time: str, timestamp_column: str, metric_columns: str, input_files: list[str],
        window_length: int, granularity_minutes: int, output_dir: Path, column_names: str or None,
        fill_na: bool, derived: str, output_metric: str, given_predict: Optional[float],
        extra_eval_columns: str, read_pickle: bool, timestamp_column_type: str, drop_columns: str,
        query: str,
):
    # parse anomaly time
    if anomaly_time != "":
        anomaly_timestamp: int = int(dt_parse(anomaly_time, ignoretz=True).astimezone(TZ).timestamp())
    else:
        logger.debug(f"Parse anomaly time from the filename: {input_files[0]=}")
        anomaly_timestamp: int = int(dt_parse(
            os.path.basename(input_files[0]).split('.')[0], ignoretz=True
        ).astimezone(TZ).timestamp())
    logger.info(f"{anomaly_time=} {datetime.fromtimestamp(anomaly_timestamp)=} {anomaly_timestamp=}")
    del anomaly_time

    output_dir: Path = Path(output_dir)
    logger.info(f"{output_dir=}")
    if column_names == "" or column_names == "None":
        column_names = None
    logger.info(f"{column_names=}")

    if read_pickle:
        logger.info("reading pickles")
        df = pd.concat(
            list(map(pd.read_pickle, input_files))
        )
    else:
        logger.info("reading CSVs")
        if column_names is not None:
            df = pd.concat(
                list(map(partial(pd.read_csv, index_col=None, names=column_names.split(",")), input_files))
            )
        else:
            df = pd.concat(list(map(partial(pd.read_csv, index_col=None, header=0), input_files)))
    logger.info("read input dataframes finished")

    logger.info(f"{timestamp_column=} {timestamp_column_type=}")
    if timestamp_column_type == "datetime":
        df['timestamp'] = df[timestamp_column].map(partial(dt_parse, ignoretz=True)).map(
            lambda x: int(x.replace(second=0, microsecond=0).astimezone(TZ).timestamp())
        )
    else:
        df['timestamp'] = df[timestamp_column] // 60 * 60
    if timestamp_column != "timestamp":
        df.drop(columns=[timestamp_column], inplace=True)

    if query != "":
        df = df.query(query)

    extra_metric_columns = []
    for extra_eval_column in extra_eval_columns.split(";"):
        __column_name, __expr = extra_eval_column.split("=")
        extra_metric_columns.append(__column_name)
        logger.info(f"{__column_name=} {__expr=}")
        df[__column_name] = df.eval(__expr)

    metric_columns = metric_columns.split(",") + extra_metric_columns
    logger.info(f"{metric_columns=}")
    drop_columns = drop_columns.split(",")
    logger.info(f"{drop_columns=}")
    attr_columns = list(set(df.columns) - set(metric_columns) - {'timestamp'} - set(drop_columns))
    logger.info(f"{attr_columns=}")
    assert set(metric_columns) < set(df.columns), f"{df.columns=}, {metric_columns=}"
    assert set(attr_columns) < set(df.columns), f"{df.columns=}, {attr_columns=}"
    assert set(attr_columns) & set(metric_columns) == set(), f"{attr_columns=}, {metric_columns=}"

    if output_metric not in metric_columns:
        assert derived != "", "derived should be set when metric is not in metric_columns"
    logger.info(f"{output_metric=}")

    # drop the drop_columns in df
    df = df.groupby(attr_columns + ["timestamp"], observed=True)[metric_columns].sum().reset_index()

    if derived != "":
        derived = tuple(derived.split(","))
        assert len(derived) == 3, derived
        assert derived[0] in metric_columns, derived
        assert derived[2] in metric_columns, derived
        assert derived[1] in {'divide', 'plus', 'minus', 'multiply'}, derived
    else:
        derived = None
    logger.info(f"{derived=}")

    ac_df = df.drop(
        columns=list(set(df.columns) - set(attr_columns))
    ).drop_duplicates()  # which contain all attribute combinations
    logger.info(f"#attribute combinations: {len(ac_df)}, {ac_df.columns=}")

    granularity = timedelta(minutes=granularity_minutes).total_seconds()

    abnormal_dt_list = np.arange(
        anomaly_timestamp - window_length * granularity, anomaly_timestamp + granularity, granularity
    )
    filled_df_list = []
    for his_timestamp in abnormal_dt_list:
        __the_df = df[df.timestamp == int(his_timestamp)]
        logger.info(
            f"#attribute combinations (assert no duplications) in "
            f"{his_timestamp} {datetime.fromtimestamp(his_timestamp)}: "
            f"{len(__the_df)}"
        )
        filled_df_list.append(
            pd.merge(
                ac_df,
                __the_df.drop(columns=['timestamp']),
                how='left', on=attr_columns, sort=True
            )
        )
        if fill_na:
            filled_df_list[-1].fillna(0, inplace=True)

    output_filename_prefix = f"{anomaly_timestamp:.0f}"
    if len(input_files) == 1:
        output_filename_prefix += f".{os.path.basename(input_files[0])}"
    output_filename_prefix += f".{output_metric}"
    if derived:
        if derived[1] == 'divide':
            ret_df_2 = get_ret_df_for_metric(filled_df_list, derived[2], attr_columns, predict=given_predict is None)
            if given_predict is not None:
                ret_df_2['predict'] = ret_df_2['real']
            ret_df_2.to_csv(output_dir / f"{output_filename_prefix}.b.csv", index=False)
            ret_df_1 = get_ret_df_for_metric(filled_df_list, derived[0], attr_columns, predict=given_predict is None)
            if given_predict is not None:
                ret_df_1['predict'] = ret_df_2['real'] * given_predict
            ret_df_1.to_csv(output_dir / f"{output_filename_prefix}.a.csv", index=False)
        else:
            raise NotImplementedError(f"{derived=}")
    else:
        assert output_metric in metric_columns
        # Fundamental measure
        ret_df = get_ret_df_for_metric(filled_df_list, output_metric, attr_columns, predict=given_predict is None)
        if given_predict is not None:
            ret_df['predict'] = given_predict
        ret_df.to_csv(output_dir / f"{output_filename_prefix}.csv", index=False)


def get_ret_df_for_metric(filled_df_list: list[pd.DataFrame], metric: str, attr_columns: list[str],
                          predict: bool = True):
    ret_df = filled_df_list[-1].copy().fillna(0)
    ret_df['real'] = ret_df[metric]
    if predict:
        historical_data = np.vstack([_[metric].values for _ in filled_df_list[:-2]])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            ret_df['predict'] = np.nan_to_num(np.nanmean(historical_data, axis=0), nan=0)
        logger.info(f"|residual|_1={np.mean(np.abs(ret_df.real - ret_df.predict))}")
        logger.info(f"|residual|_2={np.sqrt(np.mean(np.square(ret_df.real - ret_df.predict)))}")
        logger.info(f"|residual|_inf={np.max(ret_df.real - ret_df.predict)}")
    else:
        logger.info("no predict calculated")
    ret_df.drop(columns=list(set(ret_df.columns) - set(attr_columns) - {'real', 'predict'}), inplace=True)
    return ret_df


if __name__ == '__main__':
    main()
