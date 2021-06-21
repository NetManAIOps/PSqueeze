import math
from typing import FrozenSet, List

import pandas as pd

from utility import AC


def append_list_in_a_list(list_list, list_item):
    list_list.append([])
    ll = len(list_list) - 1
    for ii in range(len(list_item)):
        list_list[ll].append(list_item[ii])


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


def adtributor_one_layer(dt_cube: pd.DataFrame, t_eep, t_ep):
    attribute_names = list(sorted(set(dt_cube.columns) - {'real', 'predict'}))
    f_str = "predict"
    a_str = "real"
    f_all = sum(dt_cube[f_str])
    a_all = sum(dt_cube[a_str])
    candi_set = []
    exp_set = []
    for attr in attribute_names:
        ss = 0
        ep = 0
        candi_set.clear()
        gp = pd.DataFrame(dt_cube[[f_str, a_str]].groupby(dt_cube[attr]).sum())
        gp['EP'] = (gp[a_str] - gp[f_str]) / (a_all - f_all)
        gp['surprise'] = 0
        for jj in range(gp.shape[0]):
            gp.loc[list(gp.index)[jj], 'surprise'] = surprise(list(gp[f_str])[jj], list(gp[a_str])[jj], f_all, a_all)
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


def adtributor(df: pd.DataFrame, T_EP=0.67, T_EEP=0.1) -> List[FrozenSet[AC]]:
    data_cube = df
    explain_set = adtributor_one_layer(data_cube, T_EEP, T_EP)
    attribute_names = list(sorted(set(df.columns) - {'real', 'predict'}))
    root_cause = ""
    for i in range(len(explain_set)):
        for j in range(len(explain_set[i][1])):
            root_cause += explain_set[i][0] + '=' + explain_set[i][1][j]
            if j != len(explain_set[i][1]) - 1:
                root_cause += ';'
            elif i != len(explain_set) - 1:
                root_cause += ';'
            else:
                rubbish = 0
    return [AC.batch_from_string(root_cause, attribute_names)]
