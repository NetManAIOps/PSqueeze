import random
import time
from itertools import product

import numpy as np
from typing import List, Callable, FrozenSet, Set, Dict, Optional, Container
import pandas as pd
from loguru import logger
from utility import AC
from functools import lru_cache
from sklearn.cluster import DBSCAN
from scipy.spatial.distance import cosine


class ImpAPTr:
    def __init__(
            self, data_list: List[pd.DataFrame], op=lambda x: x, n_ele: int=1
    ):
        """
        :param data_list:
        :param op:
        """
        self.n_ele = n_ele
        self._op = op
        self._data_list = data_list
        self._derived_data: pd.DataFrame = self.__get_derived_dataframe(
            self._data_list,
            self._op
        )
        self._derived_data['diff'] = np.abs(self.derived_data['real'] - self.derived_data['predict'])
        self._root_causes: List[FrozenSet[AC]] = []

        self.attribute_names = list(sorted(set(self._derived_data.columns) - {'real', 'predict', 'diff'}))
        logger.debug(f"available attributes: {self.attribute_names}")

        idx = self._derived_data['predict'] > 0
        for i in range(len(self._data_list)):
            self._data_list[i] = self._data_list[i][idx]
        self._derived_data = self._derived_data[idx]
        self._derived_data['loc'] = np.arange(len(self.derived_data))

        self.attribute_values: Dict = {
            name: set(self._derived_data[name]) for name in self.attribute_names
        }

        # 异常是降低就是1， 是升高就是-1
        self.__direction = 1 if self.derived_data.real.sum() < self.derived_data.predict.sum() else -1

        self.__finished = False

    def run(self):
        if self.__finished:
            return
        root_cause_candidates = []
        root = AC(**{_: AC.ANY for _ in self.attribute_names})
        self._bfs_search(self._get_children(root), root_cause_candidates)
        rank_cp = self._get_rank(root_cause_candidates, key=self.contribution_power, reverse=False)
        rank_df = self._get_rank(root_cause_candidates, key=self.diversity_factor, reverse=True)
        ranks = rank_cp + rank_df
        # ranks = self._get_rank(root_cause_candidates, key=lambda x: self.contribution_power(x) * self.di)
        idx = np.argsort(ranks)
        root_cause_candidates = np.asarray(root_cause_candidates)[idx]
        for rc in root_cause_candidates[:self.n_ele]:
            self._root_causes.append({rc})
        self.__finished = True

    def _bfs_search(self, current_layer: Set[AC], candidates: List[AC]):
        if len(current_layer) <= 0:
            return
        logger.info(f"current layer contains {len(current_layer)} nodes")
        next_layer = set()
        for ac in current_layer:
            if self.contribution_power(ac) < 0:
                candidates.append(ac)
            if self._data_list[-1].loc[self.get_index_by_ac(ac), 'real'].sum() < \
                    0.05 * self._data_list[-1]['real'].sum():
                continue
            if self.impact_factor(ac, after=True) >= 0:
                continue
            next_layer |= self._get_children(ac)
        self._bfs_search(next_layer, candidates)

    def _get_rank(self, ac_list: List[AC], key=None, reverse=None):
        idx = sorted(np.arange(len(ac_list)), key=lambda _: key(ac_list[_]), reverse=reverse)
        rank = np.zeros_like(idx)
        rank[idx] = np.arange(len(ac_list))
        return rank

    @lru_cache
    def _get_children(self, ac: AC) -> Set[AC]:
        ret = set()
        for attribute in set(self.attribute_names) - set(ac.non_any_keys):
            for value in self.attribute_values[attribute]:
                ret.add(ac.add(attribute, value))
        return ret

    @property
    def report(self) -> str:
        report_df = pd.DataFrame(columns=['root_cause', 'score'])
        for ac in self.root_cause:
            ac = list(ac)[0]
            report_df = report_df.append(
                {"root_cause": str(ac), "score": self.impact_factor(ac, after=True)},
                ignore_index=True
            )
        return report_df.to_csv(index=False)

    @property
    def derived_data(self):
        return self._derived_data

    @property
    def info_collect(self):
        try:
            return {
                "scores_min": min(map(
                    lambda _: self.impact_factor(list(_)[0], after=True),
                    self.root_cause
                ))
            }
        except ValueError:
            return {}

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

    def get_derived_measure(self, idx, aggregation="sum"):
        values = [_data.loc[idx, ['real', 'predict']].values for _data in self._data_list]
        if aggregation == "sum":
            values = [np.sum(_, axis=0) for _ in values]
        else:
            values = values
        return self._op(*values)

    @lru_cache
    def get_index_by_ac(self, ac: AC):
        if len(ac.non_any_keys) > 0:
            return ac.index_dataframe(self._get_indexed_dataframe(ac.non_any_keys))
        else:
            return np.ones((len(self._derived_data)), dtype=bool)

    @lru_cache
    def _get_indexed_dataframe(self, keys):
        # 不能乱排序，需要保证loc和self.derived_data是一样的
        return self._derived_data.set_index(list(keys)).sort_index()

    @lru_cache
    def impact_factor(self, ac: AC, after=False):
        column = 0 if after else 1
        e1_idx = ~self.get_index_by_ac(ac)
        e2_idx = np.ones_like(e1_idx)
        imp_f = (
            self.get_derived_measure(e2_idx)[column] - self.get_derived_measure(e1_idx)
        )[column] * self.__direction
        return imp_f

    @lru_cache
    def contribution_power(self, ac: AC):
        return self.impact_factor(ac, after=True) - self.impact_factor(ac, after=False)

    @lru_cache
    def diversity_factor(self, ac: AC):
        idx = self.get_index_by_ac(ac)
        p, q = self.get_derived_measure(idx)
        return 0.5 * (p * np.log(2 * p / (p + q + 1e-4)) + q * np.log(2 * q / (p + q + 1e-4)) + 1e-4)
