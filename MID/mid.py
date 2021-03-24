import random
import time
from itertools import product
import pylru

import numpy as np
from typing import List, Callable, FrozenSet, Set, Dict, Optional
import pandas as pd
from loguru import logger
from utility import AC
from functools import lru_cache
from sklearn.cluster import DBSCAN
from scipy.spatial.distance import cosine


class MID:
    def __init__(
            self, data_list: List[pd.DataFrame], op=lambda x: x,
            p: float = 0.2,
            d: float = 0.5,
            max_seconds: Optional[float] = 60,
            max_iterations: Optional[int] = 1e5,
    ):
        """
        :param data_list:
        :param op:
        :param p: the probability of random update
        :param d: the distance threshold for DBSCAN clustering
        :param max_seconds: maximum running time
        :param max_iterations: maximum iterations
        """
        self.max_iterations = max_iterations
        self.max_seconds = max_seconds
        self.d = d
        self.p = p
        self._op = op
        self._data_list = data_list
        self._derived_data: pd.DataFrame = self.__get_derived_dataframe(
            self._data_list,
            self._op
        )
        self._derived_data['diff'] = np.abs(self.derived_data['real'] - self.derived_data['predict'])
        self._derived_data = self._derived_data[self._derived_data.predict > 0]
        self._root_causes: List[FrozenSet[AC]] = []

        self.attribute_names = list(sorted(set(self._derived_data.columns) - {'real', 'predict', 'diff'}))
        logger.debug(f"available attributes: {self.attribute_names}")

        self._derived_data['loc'] = np.arange(len(self.derived_data))

        self.attribute_values: Dict = {
            name: set(self._derived_data[name]) for name in self.attribute_names
        }

        self.__finished = False

    def run(self):
        def terminable() -> bool:
            if self.max_seconds is not None and (time.time() - tic) > self.max_seconds:
                return True
            if self.max_iterations is not None and iter_cnt > self.max_iterations:
                return True
            if unchanged_cnt > 100:
                return True
            return False
        if self.__finished:
            logger.warning(f"try to rerun {self}")
            return self
        c = AC(**{a: AC.ANY for a in self.attribute_names})
        ec: Set[AC] = set()
        min_score = float('inf')
        forbidden = pylru.lrucache(size=100)
        unchanged_cnt = 0
        iter_cnt = 0
        tic = time.time()
        while not terminable():
            logger.debug(f"{c=}")
            # logger.debug(
            #     f"{c=} {unchanged_cnt=} {iter_cnt=} {elapsed_time=}"
            # )
            # logger.debug(f"{ec=}")
            forbidden[c] = True
            if len(ec) == 0 or self.objective_function(c) > min_score:
                min_score = min(self.objective_function(c), min_score)
                ec.add(c)
            if random.random() < self.p:
                logger.debug("in random mode")
                new_c = self.random_update(c)
            else:
                logger.debug("in greedy mode")
                new_c = self.greedy_update(c, forbidden=forbidden)
            if c == new_c:
                unchanged_cnt += 1
                logger.debug("unchanged")
            else:
                unchanged_cnt = 0
            iter_cnt += 1
            c = new_c
        logger.info(f"EC size: {len(ec)}")
        ec: List[AC] = sorted(list(ec), key=lambda _: self.objective_function(_), reverse=True)[:10]
        logger.info(f"get EC: {ec[:10]}...")
        logger.info(f"calculating distance")
        distances = np.zeros((len(ec), len(ec)), dtype=float)
        for i in range(len(ec)):
            for j in range(i):
                distances[i, j] = self.ac_distance(ec[i], ec[j])
                distances[j, i] = distances[i, j]
        logger.info(f"clustering")
        labels = DBSCAN(
            eps=self.d, metric="precomputed",
        ).fit_predict(distances)
        clusters = {}
        for label, ac in zip(labels, ec):
            if label not in clusters:
                clusters[label] = set()
            clusters[label].add(ac)
        for cluster in clusters.values():
            self._root_causes.append(frozenset({
                max(cluster, key=self.objective_function)
            }))
        self.__finished = True

    @property
    def report(self) -> str:
        report_df = pd.DataFrame(columns=['root_cause', 'score'])
        for ac in self.root_cause:
            ac = list(ac)[0]
            report_df = report_df.append(
                {"root_cause": str(ac), "score": self.objective_function(ac)},
                ignore_index=True
            )
        return report_df.to_csv(index=False)

    @property
    def derived_data(self):
        return self._derived_data

    @property
    def info_collect(self):
        return {
            "scores_min": min(map(
                lambda _: self.objective_function(list(_)[0]),
                self.root_cause
            ))
        }

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
    def ac_distance(self, a: AC, b: AC) -> float:
        idx_a = self.get_index_by_ac(a)
        idx_b = self.get_index_by_ac(b)
        cos = cosine(
            np.sum(self._derived_data.loc[idx_a, ['predict', 'real']].values, axis=0),
            np.sum(self._derived_data.loc[idx_b, ['predict', 'real']].values, axis=0),
        )
        jaccard = np.count_nonzero(
            np.logical_and(idx_a, idx_b)
        ) / np.count_nonzero(
            np.logical_or(idx_a, idx_b)
        )
        return (jaccard + cos) / 2

    @staticmethod
    def __get_derived_dataframe(
            data_list: List[pd.DataFrame],
            op: Callable,
    ) -> pd.DataFrame:
        values = op(*[_data[['real', 'predict']].values for _data in data_list])
        ret = data_list[0].copy(deep=True)
        ret[['real', 'predict']] = values
        return ret

    @property
    def root_cause(self) -> List[FrozenSet[AC]]:
        return self._root_causes

    @staticmethod
    def get_random_op(c: AC) -> 'str':
        if c.is_terminal():
            ops = ['swap_value', 'delete']
        elif len(c.non_any_keys) == 0:
            ops = ['add']
        else:
            ops = ['add', 'swap_value', "swap_tuple", 'delete']
        op = random.choice(ops)
        logger.debug(f"{op=} {ops=}")
        return op

    def random_update(self, c: AC) -> AC:
        op = self.get_random_op(c)
        if op == 'add':
            a = random.choice(list(filter(
                lambda _: c[_] == AC.ANY,
                self.attribute_names
            )))
            v = random.choice(list(self.attribute_values[a]))
            logger.debug(f"add ({a}={v})")
            return c.add(
                attribute=a, value=v
            )
        elif op == 'swap_value':
            a = random.choice(c.non_any_keys)
            v = random.choice(list(set(self.attribute_values[a]) - {c[a]}))
            logger.debug(f"swap value ({a}={v})")
            return c.swap(a, a, v)
        elif op == 'swap_tuple':
            a_old = random.choice(c.non_any_keys)
            a_new = random.choice(list(set(self.attribute_names) - {a_old}))
            v = random.choice(list(self.attribute_values[a_new]))
            logger.debug(f"swap tuple ({a_old}, _) to ({a_new}={v})")
            return c.swap(a_old, a_new, v)
        elif op == 'delete':
            a = random.choice(c.non_any_keys)
            logger.debug(f"delete ({a}, _)")
            return c.delete(a)
        else:
            raise RuntimeError(f"unknown op: {op}")

    def greedy_update(self, c: AC, forbidden: Set[AC]) -> AC:
        op = self.get_random_op(c)
        if op == 'add':
            a_candidates = list(filter(
                lambda _: c[_] == AC.ANY,
                self.attribute_names
            ))
            for a, v in self.max_entropy_tuples(
                    sum([
                        [(_a, _v) for _v in self.attribute_values[_a]]
                        for _a in a_candidates
                    ], [])
            ):
                ret = c.add(
                    attribute=a, value=v
                )
                if ret not in forbidden:
                    logger.debug(f"add ({a}={v})")
                    return ret
                else:
                    # logger.debug(f"skip add ({a}={v}) since {ret} in forbidden")
                    pass
            else:
                return c
        elif op == 'swap_value':
            for a, _ in self.max_entropy_tuples([
                (_, c[_]) for _ in c.non_any_keys
            ]):
                for _, v in self.max_entropy_tuples([
                    (a, _) for _ in list(set(self.attribute_values[a]) - {c[a]})
                ]):
                    ret = c.swap(a, a, v)
                    if ret not in forbidden:
                        logger.debug(f"swap value ({a}={v})")
                        return ret
                    else:
                        # logger.debug(f"skip swap value ({a}={v}) since {ret} in forbidden")
                        pass
            else:
                return c
        elif op == 'swap_tuple':
            for a_old, _ in self.max_entropy_tuples([
                (_, c[_]) for _ in c.non_any_keys
            ]):
                for a_new, v_new in self.max_entropy_tuples(sum([
                    [(_a, _v) for _v in self.attribute_values[_a]]
                    for _a in list(set(self.attribute_names) - {a_old})
                ], [])):
                    ret = c.swap(a_old, a_new, v_new)
                    if ret not in forbidden:
                        logger.debug(f"swap tuple ({a_old}, _) to ({a_new}={v_new})")
                        return ret
                    else:
                        # logger.debug(f"skip swap tuple ({a_old}, _) to ({a_new}={v_new}) since {ret} in forbidden")
                        pass
            else:
                return c
        elif op == 'delete':
            for a, _ in self.max_entropy_tuples(
                    sum([
                        [(_a, _v) for _v in self.attribute_values[_a]]
                        for _a in c.non_any_keys
                    ], [])
            ):
                ret = c.delete(a)
                logger.debug(f"delete ({a}, _)")
                return ret
        else:
            raise RuntimeError(f"unknown op: {op}")

    @lru_cache
    def objective_function(self, ac: AC):
        def dis(_a, _b):
            _ret = np.abs(_a - _b)
            if len(_ret) <= 0:
                return 0.
            else:
                return np.mean(_ret)
        idx_p = self.get_index_by_ac(ac)
        idx_n = np.logical_not(idx_p)
        v1 = self._derived_data.loc[idx_p, 'real'].values
        v2 = self._derived_data.loc[idx_n, 'real'].values
        f1 = self._derived_data.loc[idx_p, 'predict'].values
        f2 = self._derived_data.loc[idx_n, 'predict'].values
        a1 = f1 * np.sum(v1) / np.sum(f1)
        ps = 1 - (
            dis(v1, a1) + dis(v2, f2)
        ) / (
            dis(v1, f1) + dis(v2, f2) + 1e-4
        )
        cuboid_layer = len(ac.non_any_keys)
        weight = 0.01
        return ps - weight * cuboid_layer * cuboid_layer

        # EP
        # delta = (
        #                 self._derived_data["real"] - self._derived_data["predict"]
        #         ).sum() + 1e-4
        # idx = self.get_index_by_ac(ac)
        # cover = (
        #                 self._derived_data[idx]["real"] - self._derived_data[idx]["predict"]
        #         ).sum() + 1e-4
        # return cover / delta

        # fisher
        # idx = self.get_index_by_ac(ac)
        # pa = self._derived_data.loc[idx, 'real'].sum() / self._derived_data['real'].sum() + 1e-4
        # pb = self._derived_data.loc[idx, 'predict'].sum() / self._derived_data['predict'].sum() + 1e-4
        # p = self._derived_data.loc[idx, 'diff'].sum() / self._derived_data['diff'].sum() + 1e-4
        # # p = abs((self._derived_data.loc[idx, 'real'].sum() - self._derived_data.loc[idx, 'predict'].sum())
        # #         / (self._derived_data['real'].sum() - self._derived_data['predict'].sum()))
        # return np.abs(p * np.log(pa / pb))

    @lru_cache
    def tuple_entropy(self, attribute, value):
        p = np.count_nonzero(
            self.get_index_by_ac(AC(**{attribute: value}))
        ) / len(self._derived_data) + 1e-4
        return -p * np.log(p)

    @lru_cache
    def attribute_entropy(self, attribute):
        return sum([
            self.tuple_entropy(attribute=attribute, value=_)
            for _ in self.attribute_values[attribute]
        ])

    def max_entropy_tuples(self, candidates):
        return sorted(
            candidates,
            key=lambda _: self.attribute_entropy(_[0]) + self.tuple_entropy(_[0], _[1]),
            reverse=True
        )
