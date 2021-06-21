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
    pp = a / a_all
    qq = f / f_all
    if (pp != 0) and (qq != 0):
        ss = 0.5 * (pp * math.log(2 * pp / (pp + qq), 10) + qq * math.log(2 * qq / (pp + qq), 10))
    elif pp == 0:
        ss = 0.5 * qq * math.log(2, 10)
    else:
        ss = 0.5 * pp * math.log(2, 10)
    return ss


# In[ ]:


def adtributor_one_layer(dt_cube: pd.DataFrame, t_eep, t_ep):
    attribute_names = list(sorted(set(dt_cube.columns) - {'real_1', 'predict_1', 'real_2', 'predict_2'}))
    f_all_1 = sum(dt_cube['predict_1'])
    a_all_1 = sum(dt_cube['real_1'])
    f_all_2 = sum(dt_cube['predict_2'])
    a_all_2 = sum(dt_cube['real_2'])
    candi_set = []
    exp_set = []
    for attr in attribute_names:
        ss = 0
        ep = 0
        candi_set.clear()
        gp_ = dt_cube[['predict_1', 'real_1', 'predict_2', 'real_2']].groupby(dt_cube[attr]).sum()
        gp = pd.DataFrame(gp_)
        gp['EP'] = (gp['real_1'] - gp['predict_1']) * f_all_2 - (gp['real_2'] - gp['predict_2']) * f_all_1
        gp['EP'] = (gp['EP'] / f_all_2) / (f_all_2 + gp['real_2'] - gp['predict_2'])  # calculate EP
        gp['EP'] = gp['EP'] / sum(list(gp['EP']))  # normalization
        gp['surprise'] = 0
        for jj in range(gp.shape[0]):
            ss1 = surprise(list(gp['predict_1'])[jj], list(gp['real_1'])[jj], f_all_1, a_all_1)
            ss2 = surprise(list(gp['predict_2'])[jj], list(gp['real_2'])[jj], f_all_2, a_all_2)
            gp.loc[list(gp.index)[jj], 'surprise'] = ss1 + ss2
        gp = gp.sort_values(by=['surprise'], axis=0, ascending=False)
        for jj in range(gp.shape[0]):
            if list(gp['EP'])[jj] > t_eep:
                candi_set.append(str(list(gp.index)[jj]))
                ss += list(gp['surprise'])[jj]
                ep += list(gp['EP'])[jj]
                if ep > t_ep:
                    break
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


def adtributor_derived(df_list: List[pd.DataFrame], T_EP=0.67, T_EEP=0.1) -> List[FrozenSet[AC]]:
    data_cube_1 = df_list[0]
    data_cube_2 = df_list[1]
    attribute_names = list(sorted(set(data_cube_1.columns) - {'real', 'predict'}))
    data_cube_1.rename(columns={"predict": 'predict_1', "real": 'real_1'}, inplace=True)
    data_cube_1['predict_2'] = data_cube_2["predict"]
    data_cube_1['real_2'] = data_cube_2["real"]
    explain_set = adtributor_one_layer(data_cube_1, T_EEP, T_EP)
    root_cause = ""
    for i in range(len(explain_set)):
        for j in range(len(explain_set[i][1])):
            root_cause += explain_set[i][0] + '=' + explain_set[i][1][j]
            if (j != len(explain_set[i][1]) - 1) or (i != len(explain_set) - 1):
                root_cause += ';'
    return [AC.batch_from_string(root_cause, attribute_names)]
