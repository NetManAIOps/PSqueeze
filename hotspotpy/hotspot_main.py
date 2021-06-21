from heapq import merge
from typing import List, FrozenSet
import typing
import pandas as pd
import numpy as np
from itertools import combinations
from memory_profiler import profile
from utility import AC
from .hotspot_cuboid import Cuboid
from loguru import logger


K = 3


class HotSpot:
    def __init__(self, data_list: typing.List[pd.DataFrame], op: typing.Callable = lambda x: x,
                 max_steps=None, max_time=None, ps_upper_threshold=1.0, ps_lower_threshold=0.0):
        """
        :param data_list:  each line must match exactly TODO: check this
        :param op:
        :param max_steps:
        :param max_time:
        :param ps_upper_threshold:
        """
        self.ps_lower_threshold = ps_lower_threshold if ps_lower_threshold is not None else float('-inf')
        self.max_time = max_time
        self.max_steps = max_steps
        self.ps_upper_threshold = ps_upper_threshold if ps_upper_threshold is not None else float('+inf')
        self.op = op
        self.data_list = data_list

        def _get(_data):
            _ = list(set(_data.columns) - {'real', 'predict'})
            _attribute_values = {}
            for column in _:
                _attribute_values[column] = list(set(_data[column]))
            return _, _attribute_values

        for idx, data in enumerate(data_list):
            data.real = data.real.astype('float64')
            data.predict = data.predict.astype('float64')
            assert 'real' in data.columns, f'data {idx} should contain column \'real\', {data.columns}'
            assert 'predict' in data.columns, f'data {idx} should contain column \'predict\', {data.columns}'
        ret_list = list(_get(_) for _ in data_list)
        assert all(_ == ret_list[0] for _ in ret_list), f'attributes are not the same, {ret_list}'
        self.attribute_names = ret_list[0][0]  # type: typing.List[str]
        self.attribute_values = ret_list[0][1]  # type: typing.Dict[str, list]
        assert len(self.attribute_names) > 0, f"there is no available attributes, {self.attribute_names}"
        # logger.info(f'available attribute names: {self.attribute_names}')
        self._best_nodes = list()  # type: typing.List['HotSpotNode']
        self._finished = False
        self.cuboids = [[]]  # type: List[List[Cuboid]]

    @profile
    def run(self):
        if self._finished:
            logger.warning("try to rerun HotSpot")
            return
        for idx_layer in np.arange(len(self.attribute_names)) + 1:  # from 1 to #{attribute_names}
            self.cuboids.append([])
            for cuboid in combinations(self.attribute_names, idx_layer):
                cuboid = Cuboid(self, cuboid,
                                max_steps=self.max_steps,
                                max_time=self.max_time,
                                ps_upper_threshold=self.ps_upper_threshold,
                                ps_lower_threshold=self.ps_lower_threshold)
                cuboid.run()
                self.cuboids[idx_layer].append(cuboid)
            for cuboid in self.cuboids[idx_layer]:
                logger.debug(f'{cuboid} best nodes:')
                for node in cuboid.best_nodes:
                    logger.debug(f'{node}')
                self._best_nodes = list(merge(self._best_nodes, cuboid.best_nodes,
                                              key=lambda x: x.result_score, reverse=True))[:K]
                if len(self._best_nodes) > 0 and self._best_nodes[0].potential_score >= self.ps_upper_threshold:
                    break
            else:  # iter all cuboids, then continue
                continue
            break  # inner loop is broken, so break
        self._best_nodes = sorted(self._best_nodes, key=lambda x: x.result_score, reverse=True)
        self._finished = True

    @property
    def best_nodes(self):
        assert self._finished, 'HotSpot has not run.'
        return self._best_nodes

    @property
    def report(self):
        return ""

    @property
    def info_collect(self):
        return {
        }

    @property
    def root_cause(self) -> List[FrozenSet[AC]]:
        try:
            node = self.best_nodes[0]  # type: 'HotSpotNode'
            root_cause = ";".join(str(_) for _ in node.attribute_combinations)
        except IndexError:
            root_cause = ""
        return [AC.batch_from_string(root_cause, self.attribute_names)]
