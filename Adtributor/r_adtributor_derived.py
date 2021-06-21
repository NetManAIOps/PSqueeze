#!/usr/bin/env python
# coding: utf-8

import json
import math
import os
import time
# In[ ]:
from typing import List, FrozenSet

import loguru
import pandas as pd

# In[ ]:
from utility import AC


def append_list_in_a_list(list_list, list_item):
    list_list.append([])
    ll = len(list_list) - 1
    for ii in range(len(list_item)):
        list_list[ll].append(list_item[ii])


# In[ ]:


def surprise(f, a, f_all, a_all):
    if (a_all != 0) and (f_all != 0):
        pp = a / a_all
        qq = f / f_all
        if (pp != 0) and (qq != 0):
            ss = 0.5 * (pp * math.log(2 * pp / (pp + qq), 10) + qq * math.log(2 * qq / (pp + qq), 10))
        elif pp == 0:
            ss = 0.5 * qq * math.log(2, 10)
        else:
            ss = 0.5 * pp * math.log(2, 10)
    elif (a_all == 0) and (f_all == 0):
        ss = 0
    elif (a_all == 0) and (f_all != 0):
        qq = f / f_all
        ss = 0.5 * qq * math.log(2, 10)
    else:
        pp = a / a_all
        ss = 0.5 * pp * math.log(2, 10)
    return ss


# In[ ]:


def adtributor_one_layer(dt_cube: pd.DataFrame, t_eep):
    attribute_names = list(sorted(set(dt_cube.columns) - {'real_1', 'predict_1', 'real_2', 'predict_2'}))
    f_all_1 = sum(dt_cube['predict_1'])
    a_all_1 = sum(dt_cube['real_1'])
    f_all_2 = sum(dt_cube['predict_2'])
    a_all_2 = sum(dt_cube['real_2'])
    candi_set = []
    exp_set = []
    for attr in attribute_names:
        ss = 0
        candi_set.clear()
        gp_ = dt_cube[['predict_1', 'real_1', 'predict_2', 'real_2']].groupby(dt_cube[attr]).sum()
        gp = pd.DataFrame(gp_)
        gp['EP'] = (gp['real_1'] - gp['predict_1']) * f_all_2 - (gp['real_2'] - gp['predict_2']) * f_all_1
        gp['surprise'] = 0
        for jj in range(gp.shape[0]):
            xx = list(gp.index)[jj]
            if gp.loc[xx, 'EP'] != 0:
                if f_all_2 * (f_all_2 + gp.loc[xx, 'real_2'] - gp.loc[xx, 'predict_2']) != 0:
                    gp.loc[xx, 'EP'] = gp.loc[xx, 'EP'] / f_all_2
                    gp.loc[xx, 'EP'] = gp.loc[xx, 'EP'] / (f_all_2 + gp.loc[xx, 'real_2'] - gp.loc[xx, 'predict_2'])
                elif f_all_2 == 0:
                    if gp.loc[xx, 'real_2'] != 0:
                        gp.loc[xx, 'EP'] = (gp.loc[xx, 'real_1'] - gp.loc[xx, 'real_2']) / gp.loc[xx, 'real_2']
                    else:
                        gp.loc[xx, 'EP'] = 0
                else:
                    gp.loc[xx, 'EP'] = 0
            ss1 = surprise(list(gp['predict_1'])[jj], list(gp['real_1'])[jj], f_all_1, a_all_1)
            ss2 = surprise(list(gp['predict_2'])[jj], list(gp['real_2'])[jj], f_all_2, a_all_2)
            gp.loc[xx, 'surprise'] = ss1 + ss2
        gp = gp.sort_values(by=['surprise'], axis=0, ascending=False)
        ep_sum = sum(list(gp['EP']))
        for jj in range(gp.shape[0]):
            if list(gp['predict_2'])[jj] != 0:
                ff = list(gp['predict_1'])[jj] / list(gp['predict_2'])[jj]
            else:
                ff = 0
            if list(gp['real_2'])[jj] != 0:
                aa = list(gp['real_1'])[jj] / list(gp['real_2'])[jj]
            else:
                aa = 0
            outside = (aa > 1.05 * ff) or (aa < 0.95 * ff)
            if (list(gp['EP'])[jj] > (t_eep * ep_sum)) and True:
                candi_set.append(str(list(gp.index)[jj]))
                ss += list(gp['surprise'])[jj]
        if (len(candi_set) < gp.shape[0]) and (len(candi_set) > 0):
            exp_set.append([])
            exp_set[len(exp_set) - 1].append(attr)
            append_list_in_a_list(exp_set[len(exp_set) - 1], candi_set)
            exp_set[len(exp_set) - 1].append(ss)
    if len(exp_set) > 0:
        exp_set.sort(key=lambda ca_set: ca_set[2], reverse=True)
    for ii in range(3, len(exp_set)):
        exp_set.remove(exp_set[len(exp_set) - 1])
    return exp_set


# In[ ]:


def r_adtributor(current_dt_cube: pd.DataFrame, f_all_1, a_all_1, f_all_2, a_all_2, t_eep, layer):
    exp_set = adtributor_one_layer(current_dt_cube, t_eep)
    exp_node = []
    if layer >= 1:
        for candi_set in exp_set:
            for xx in candi_set[1]:
                sub_dt_cube = current_dt_cube[current_dt_cube[candi_set[0]] == xx]
                exp_node.append([])
                ll = len(exp_node) - 1
                exp_node[ll].append(candi_set[0])
                exp_node[ll].append(xx)
                f = sum(list(sub_dt_cube["predict_1"]))
                a = sum(list(sub_dt_cube["real_1"]))
                ss = surprise(f, a, f_all_1, a_all_1)
                f = sum(list(sub_dt_cube["predict_2"]))
                a = sum(list(sub_dt_cube["real_2"]))
                ss += surprise(f, a, f_all_2, a_all_2)
                exp_node[ll].append(ss)
                layer -= 1
                exp_node[ll].append(r_adtributor(sub_dt_cube, f_all_1, a_all_1, f_all_2, a_all_2, t_eep, layer))
    return exp_node


# exp_node:
# [
# 0: dimension name
# 1: element name
# 2: sub-cube surprise
# 3: sub-node list [sub-node1(exp_node), sub-node2, sub-node3]
# ]


# In[ ]:


def tree_to_list(root_list: list, dim_list):
    result_lt = list()
    current_node_dict = dict()
    for root_node in root_list:
        for dd in dim_list:
            current_node_dict[dd] = ""
        current_node_dict[root_node[0]] = root_node[1]
        current_node_dict["surprise"] = root_node[2]
        result_lt.append(current_node_dict)
        sub_result_list = tree_to_list(root_node[3], dim_list)
        for sub_node_dict in sub_result_list:
            sub_node_dict[root_node[0]] = root_node[1]
            result_lt.append(sub_node_dict)
    return result_lt


# In[ ]:


def tree_to_root_cause(tree_list: list, dim_list: list, rt_cause_nm):
    rt_cause = ""
    if len(tree_list) >= 1:
        tree_list.sort(key=lambda dicts: dicts["surprise"], reverse=True)
        start_flag = True
        for dd in dim_list:
            if tree_list[0][dd] != "":
                if start_flag:
                    start_flag = False
                else:
                    rt_cause += '&'
                rt_cause += dd + '=' + tree_list[0][dd]
        # if rt_cause_nm >= 2:
        #     rt_cause += ';'
        rt_cause_count = 1
        tree_list_count = 1
        while (rt_cause_count < rt_cause_nm) and (tree_list_count < len(tree_list)):
            start_flag = False
            for dd in dim_list:
                if tree_list[tree_list_count][dd] != tree_list[tree_list_count - 1][dd]:
                    start_flag = True
            if start_flag:
                for dd in dim_list:
                    if tree_list[tree_list_count][dd] != "":
                        if start_flag:
                            start_flag = False
                            rt_cause += ';'
                        else:
                            rt_cause += '&'
                        rt_cause += dd + '=' + tree_list[tree_list_count][dd]
                rt_cause_count += 1
                # if rt_cause_count < rt_cause_nm:
                #     rt_cause += ';'
            tree_list_count += 1
    return rt_cause


def r_adtributor_derived(
        df_list: List[pd.DataFrame], T_EEP=0.1, deepest_layer=3, root_cause_num=5
) -> List[FrozenSet[AC]]:
    data_cube = df_list[0]
    data_cube_2 = df_list[1]
    attribute_names = list(sorted(set(data_cube.columns) - {'real', 'predict'}))
    forecast_str_1 = "predict"
    actual_str_1 = "real"
    forecast_str_2 = "predict"
    actual_str_2 = "real"
    data_cube.rename(columns={forecast_str_1: "predict_1", actual_str_1: "real_1"}, inplace=True)
    data_cube['predict_2'] = data_cube_2[forecast_str_2]
    data_cube['real_2'] = data_cube_2[actual_str_2]
    forecast_1 = sum(list(data_cube["predict_1"]))
    real_1 = sum(list(data_cube["real_1"]))
    forecast_2 = sum(list(data_cube["predict_2"]))
    real_2 = sum(list(data_cube["real_2"]))
    explain_node = r_adtributor(data_cube, forecast_1, real_1, forecast_2, real_2, T_EEP, deepest_layer)
    rt_cause_list = tree_to_list(explain_node, attribute_names)
    root_cause = tree_to_root_cause(rt_cause_list, attribute_names, root_cause_num)
    return [AC.batch_from_string(root_cause, attribute_names)]
