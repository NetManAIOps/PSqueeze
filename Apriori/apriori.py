# coding=utf-8
import time
from typing import List, Callable, FrozenSet

import kneed
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from utility import AC
from .Apriori_mining_alg import AprioriMining


def get_kneed(bias, bandwidth):
    cdf = np.zeros(int(1 / bandwidth) + 1)
    for i in range(0, len(bias)):
        cdf[int(bias[i] * 1 / bandwidth)] += 1 / len(bias)
    cdf = np.cumsum(cdf)
    print("cdf ", cdf)
    plt.plot(np.arange(0, 1, 1 / (1 / bandwidth + 1)), cdf)
    plt.savefig("/home/v-lcy19/1.jpg")
    return kneed.KneeLocator(x=np.arange(0, 1, 1 / (1 / bandwidth + 1)), y=cdf, interp_method="polynomial").find_knee()[
        2]


def get_abnormal_points(bias, bandwidth=0.01):
    # kneed = get_kneed(bias, bandwidth=bandwidth)
    kneed = 20
    print("pivot:", kneed * bandwidth)
    return [i for i in range(0, len(bias)) if bias[i] >= kneed * bandwidth]


def run_directly(df):
    tic = time.time()
    # 第一步，净化这个数据集，去掉全部为0的列

    df = df.drop([i for i in range(0, len(df.values)) if df.values[i][-1] * df.values[i][-2] == 0], axis=0)
    attribute_names = list(sorted(set(df.columns) - {'real', 'predict'}))
    for attr in attribute_names:
        df[attr] = df[attr].map(lambda _: f"{attr}={_}")
    print("attribute names ", attribute_names)
    real_value = np.array(df["real"])
    predict_value = np.array(df["predict"])
    bias = abs(predict_value - real_value) / real_value
    # max_value = np.max(bias)
    # bias = np.array([float(i)/max_value for i in bias])
    # print("len of bias",sum(np.where(bias==0,0,1)))
    # print("len of total ", len(bias))
    abnormal_index = get_abnormal_points(bias)
    print("len of abnormal_index ", len(abnormal_index))
    df_abnormal = df.iloc[abnormal_index][attribute_names].values
    df_total = df.iloc[:][attribute_names].values
    answer = AprioriMining(df_abnormal, df_total, abnormal_index).get_ans()

    # we have add `key=` to the values before
    root_cause = ";".join("&".join([str(i) for i in abnormal_cubiod]) for abnormal_cubiod in answer)
    toc = time.time()
    elapsed_time = toc - tic
    return root_cause


class AprioriRCA:
    def __init__(self, data_list: List[pd.DataFrame], op=lambda x: x, **kwargs):
        self.derived_data: pd.DataFrame = self.__get_derived_dataframe(
            data_list,
            op,
        )
        self._attribute_names = list(sorted(set(self.derived_data.columns) - {'real', 'predict'}))
        self._root_causes = []

    def run(self):
        self._root_causes = [
            AC.batch_from_string(run_directly(self.derived_data), attribute_names=self._attribute_names)]

    @property
    def report(self):
        return ""

    @property
    def info_collect(self):
        return {
        }

    @property
    def root_cause(self) -> List[FrozenSet[AC]]:
        return self._root_causes

    @staticmethod
    def __get_derived_dataframe(
            data_list: List[pd.DataFrame],
            op: Callable,
    ) -> pd.DataFrame:
        values = op(*[_data[['real', 'predict']].values for _data in data_list])
        ret = data_list[0].copy(deep=True)
        ret[['real', 'predict']] = values
        return ret

# run(r"C:\Users\惠普\PycharmProjects\hotspot-plus\tests\test_data\test_case8.csv","")
