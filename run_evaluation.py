#!/usr/bin/env python3
#-*- coding: utf-8 -*-
import click
import pandas as pd
import json
from utility import AttributeCombination as AC
import numpy as np
import os


@click.command()
@click.option("--injection-info", '-i', help='injection_info.csv file')
@click.option("--predict", '-p', help='output json file')
@click.option("--config", '-c', help='config json file')
@click.option("--output-path", '-o', help="output path", default="")
@click.option("--groundtruth-dir", '-g', help="groundtruth data directory")
def main(*args, **kwargs):
    evaluate(*args, **kwargs)

def evaluate(injection_info, predict, config, output_path, groundtruth_dir, verbose=True, return_detail=False):
    injection_info = pd.read_csv(injection_info)
    with open(predict, 'r') as f:
        predict = json.load(f)
    with open(config, 'r') as f:
        config = json.load(f)
    injection_info.set_index(['timestamp'], inplace=True)
    if "ex_rc_dim" in injection_info.columns:
        evaluate_ex_rc(injection_info, predict, config, output_path, verbose, return_detail, groundtruth_dir)
    else:
        evaluate_non_ex_rc(injection_info, predict, config, output_path, verbose, return_detail)

def evaluate_ex_rc(injection_info, predict, config, output_path, verbose, return_detail, gt_dir):
    for idx, item in enumerate(predict):
        try:
            rc_gt, ex_rc_label = injection_info.loc[int(item['timestamp']), ['set', 'ex_rc_dim']]
            label = predict[idx]['label'] = AC.batch_from_string(
                rc_gt,
                attribute_names=config['columns']
            )
            if not type(ex_rc_label) == str:
                ex_rc_label = None

            predict[idx]['ext_dim_label'] = ex_rc_label
            try:
                ret = AC.batch_from_string(
                    item['root_cause'].replace('|', ';'),
                    attribute_names=config['columns']
                )
                pred = predict[idx]['pred'] = ret
                ex_rc_pred = predict[idx]['external_rc']
            except Exception as e:
                print(item, e)
                continue
            _fn = len(label)
            _tp, _fp = 0, 0

            if not ex_rc_label == None:
                if ex_rc_pred:
                    _fn, _tp = 0, 1
                else:
                    path = os.path.join(gt_dir, f"{item['timestamp']}.csv")
                    df = pd.read_csv(path)
                    for rc_pred in pred:
                        positive = False
                        for rc_label in label:
                            if get_jaccard_index(df, rc_label, rc_pred) > 0.8:
                                positive = True
                                break
                        if positive:
                            _fn -= 1
                            _tp += 1
                        else:
                            _fp += 1
            else:
                if ex_rc_pred:
                    _fp = 1
                else:
                    for rc_item in pred:
                        if rc_item in label:
                            _fn -= 1
                            _tp += 1
                        else:
                            _fp += 1
        except KeyError:
            continue

        predict[idx]['tp'] = _tp
        predict[idx]['fp'] = _fp
        predict[idx]['fn'] = _fn
        predict[idx]['cuboid_layer'] = len(list(label)[0].non_any_values)
        predict[idx]['num_elements'] = len(label)
        predict[idx]['significance'] = injection_info.loc(axis=0)[int(item['timestamp']), 'significance']
        if verbose:
            print("========================================")
            print(f"timestamp:{item['timestamp']}")
            print(f"label_rc        :{AC.batch_to_string(label)}")
            print(f"label_ext_rc_dim:{ex_rc_label}")
            print(f"pred_rc         :{AC.batch_to_string(pred)}")
            print(f"pred_external_rc:{predict[idx]['external_rc']}")
            if 'ep' in predict[idx]:
                print(f"ep:{predict[idx]['ep']}")
            print(f"tp: {_tp}, fp: {_fp}, fn: {_fn}")
        del predict[idx]['root_cause']
    df = pd.DataFrame.from_records(predict)
    total_fscore = 2 * np.sum(df.tp) / (2 * np.sum(df.tp) + np.sum(df.fp) + np.sum(df.fn))
    total_precision = np.sum(df.tp) / (np.sum(df.tp) + np.sum(df.fp))
    total_recall = np.sum(df.tp) / (np.sum(df.tp) + np.sum(df.fn))
    df_total = pd.DataFrame.from_dict(
        {"tp": [np.sum(df.tp)],
         "fp": [np.sum(df.fp)],
         "fn": [np.sum(df.fn)],
         "F1-Score": [total_fscore],
         "Precision": [total_precision],
         "Recall": [total_recall],
         'Time Cost (s)': [np.mean(df['elapsed_time'])],
         'time_std': [np.std(df['elapsed_time'])],
         'Total Time Cost (s)': [np.sum(df['elapsed_time'])],
         'length': len(predict),
         # 'time_list': df['elapsed_time'].values,
         }
    )
    if verbose:
        print(df_total)
    if not output_path == "":
        df_total.to_csv(output_path, index=False)
    if verbose:
        print(f"{total_fscore:.4f} {total_precision:.4f} {total_recall:.4f}")
    if return_detail:
        return df
    return df_total
    

def evaluate_non_ex_rc(injection_info, predict, config, output_path, verbose, return_detail):
    for idx, item in enumerate(predict):
        try:
            label = predict[idx]['label'] = AC.batch_from_string(
                injection_info.loc(axis=0)[int(item['timestamp']), 'set'],
                attribute_names=config['columns']
            )
            try:
                ret = AC.batch_from_string(
                    item['root_cause'].replace('|', ';'),
                    attribute_names=config['columns']
                )
                pred = predict[idx]['pred'] = ret
            except Exception as e:
                print(item, e)
                continue
            _fn = len(label)
            _tp, _fp = 0, 0
            for rc_item in pred:
                if rc_item in label:
                    _fn -= 1
                    _tp += 1
                else:
                    _fp += 1
        except KeyError:
            continue
        predict[idx]['tp'] = _tp
        predict[idx]['fp'] = _fp
        predict[idx]['fn'] = _fn
        predict[idx]['cuboid_layer'] = len(list(label)[0].non_any_values)
        predict[idx]['num_elements'] = len(label)
        predict[idx]['significance'] = injection_info.loc(axis=0)[int(item['timestamp']), 'significance']
        if verbose:
            print("========================================")
            print(f"timestamp:{item['timestamp']}")
            print(f"label:{AC.batch_to_string(label)}")
            print(f"pred :{AC.batch_to_string(pred)}")
            print(f"tp: {_tp}, fp: {_fp}, fn: {_fn}")
        del predict[idx]['root_cause']
    df = pd.DataFrame.from_records(predict)
    total_fscore = 2 * np.sum(df.tp) / (2 * np.sum(df.tp) + np.sum(df.fp) + np.sum(df.fn))
    total_precision = np.sum(df.tp) / (np.sum(df.tp) + np.sum(df.fp))
    total_recall = np.sum(df.tp) / (np.sum(df.tp) + np.sum(df.fn))
    df_total = pd.DataFrame.from_dict(
        {"tp": [np.sum(df.tp)],
         "fp": [np.sum(df.fp)],
         "fn": [np.sum(df.fn)],
         "F1-Score": [total_fscore],
         "Precision": [total_precision],
         "Recall": [total_recall],
         'Time Cost (s)': [np.mean(df['elapsed_time'])],
         'time_std': [np.std(df['elapsed_time'])],
         'Total Time Cost (s)': [np.sum(df['elapsed_time'])],
         'length': len(predict),
         # 'time_list': df['elapsed_time'].values,
         }
    )
    if verbose:
        print(df_total)
    if not output_path == "":
        df_total.to_csv(output_path, index=False)
    if verbose:
        print(f"{total_fscore:.4f} {total_precision:.4f} {total_recall:.4f}")
    if return_detail:
        return df
    return df_total

def get_ac_from_df(df, ac):
    for k, v in ac.items():
        if v != "__ANY__":
            df = df.loc[df[k] == v]
    return df

def get_jaccard_index(df, ac1, ac2):
    df1 = get_ac_from_df(df, ac1)
    df2 = get_ac_from_df(df, ac2)
    df1_and_df2 = get_ac_from_df(df1, ac2)
    ji = df1_and_df2.size / (df1.size + df2.size - df1_and_df2.size)
    return ji
    

if __name__ == '__main__':
    main()


