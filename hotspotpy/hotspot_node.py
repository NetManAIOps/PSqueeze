import typing
from functools import lru_cache
import numpy as np
from scipy.spatial.distance import euclidean
import hotspotpy
from utility.attribute_combination import AttributeCombination


class HotSpotNode:
    def __init__(self, hotspot: 'hotspotpy.HotSpot', cuboid: 'hotspotpy.Cuboid',
                 parent: typing.Union['HotSpotNode', None],
                 attribute_combinations: typing.FrozenSet[AttributeCombination]):
        self.parent = parent
        self.attribute_combinations = attribute_combinations
        self.cuboid = cuboid
        self.hotspot = hotspot
        self.n_visits = 0
        self.children = set()
        self.wasted_children = set()
        self.is_fully_expanded = False
        self._is_wasted = False
        self.has_next_action = True
        self._reward = float('-inf')

        _data_index = np.asarray(list(
            list(AttributeCombination.index_dataframe(
                attribute_combination,
                data
            ) for data in self.cuboid.indexed_data_list)
            for attribute_combination in self.attribute_combinations
        ))
        _complement_data_index = np.logical_not(np.logical_or.reduce(_data_index, axis=0))

        _data_list = list(
            list(data[index]
                 for data, index in zip(self.cuboid.indexed_data_list, index_list))
            for index_list in _data_index
        )
        _complement_data_list = list(
            data[index]
            for data, index in zip(self.cuboid.indexed_data_list, _complement_data_index)
        )
        leaf_v_list, non_leaf_v = self._get_column_value('real', self.hotspot.op, _data_list, _complement_data_list)
        leaf_f_list, non_leaf_f = self._get_column_value('predict', self.hotspot.op, _data_list, _complement_data_list)
        # set_f, _ = self._get_column_value('predict', reduction=np.sum)
        # set_v, _ = self._get_column_value('real', reduction=np.sum)
        set_f = list(map(np.sum, leaf_f_list))
        set_v = list(map(np.sum, leaf_v_list))
        leaf_a_list = list(__f * (1. - (__set_f - __set_v) / np.maximum(__set_f, 1e-4))
                           for __f, __set_f, __set_v in zip(leaf_f_list, set_f, set_v))
        v = np.concatenate(leaf_v_list + [non_leaf_v])
        f = np.concatenate(leaf_f_list + [non_leaf_f])
        a = np.concatenate(leaf_a_list + [non_leaf_f])
        # self._v = v
        # self._f = f
        # self._a = a
        ps = np.maximum(1. - euclidean(v, a) / np.maximum(euclidean(v, f), 1e-4), 0.)
        # ps = 1. - euclidean(v, a) / np.maximum(euclidean(v, f), 1e-4)
        self._potential_score = ps

        # self._v = None
        # self._f = None
        # self._a = None

    @property
    # @lru_cache()
    def potential_score(self):
        return self._potential_score

    @staticmethod
    def _get_column_value(column, op, data_list, complement_data_list, reduction=None):
        if reduction is None:
            reduction = lambda x: x
        leaf_value_list = list(
            op(*(reduction(_.loc[:, column].values) for _ in attr_data_list))
            for attr_data_list in data_list
        )
        non_leaf_value = op(*(reduction(_.loc[:, column].values) for _ in complement_data_list))
        return leaf_value_list, non_leaf_value

    @property
    def reward(self):
        return self._reward

    @reward.setter
    def reward(self, value):
        self._reward = value

    def get_next_action(self):
        for ele in self.cuboid.elements:
            if ele.attribute_combination in self.attribute_combinations:
                continue
            yield ele.attribute_combination

    def take_action(self, action):
        attribute_combination = set(self.attribute_combinations)
        attribute_combination.add(action)
        return HotSpotNode(self.hotspot, self.cuboid, self, frozenset(attribute_combination))

    @property
    def wasted(self):
        return self._is_wasted

    @wasted.setter
    def wasted(self, wasted):
        if wasted and self.parent is not None and not self.wasted:
            # remove self from parent's children
            self.parent.children.remove(self)
            self.parent.wasted_children.add(self)
        self._is_wasted = wasted

    @lru_cache()
    def is_terminal(self):
        return self.wasted or len(self.attribute_combinations) >= len(self.cuboid.elements)

    def get_best_child_by_ucb(self, exploration_rate=1.414):
        if len(self.children) == 0:
            return None
        return max(self.children,
                   key=lambda x: x.reward + exploration_rate * np.sqrt(np.log(x.parent.n_visits) / x.n_visits))

    def back_propagate(self):
        node = self
        ps = node.potential_score
        while node is not None:
            node.n_visits += 1
            node.reward = max(node.reward, ps)
            node = node.parent

    @property
    def result_score(self):
        return self.potential_score \
               + 0.01 * (-len(self.cuboid.attribute_names)) \
               + 0.02 * (-len(self.attribute_combinations))

    def __str__(self):
        return f"{AttributeCombination.to_iops_2019_format(self.attribute_combinations)} ps:{self.potential_score:.3f}"

    def __eq__(self, other):
        return self.attribute_combinations == other.attribute_combinations and self.cuboid == other.cuboid

    def __hash__(self):
        ac = hash(self.attribute_combinations) if hasattr(self, 'attribute_combinations') else 0
        hotspot = id(self.hotspot) if hasattr(self, 'hotspot') else 0
        cuboid = id(self.cuboid) if hasattr(self, 'cuboid') else 0
        return ac + hotspot * 113 + cuboid * 11


class HotSpotElement(HotSpotNode):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        assert len(self.attribute_combinations) == 1, f'This Node is not Element, {self}'
        self.attribute_combination = next(iter(self.attribute_combinations))

        _data_index = list(AttributeCombination.index_dataframe(
            self.attribute_combination,
            data
        ) for data in self.cuboid.indexed_data_list)
        _complement_data_index = list(
            np.logical_not(index) for index in _data_index
        )
        _data_list = list(
            data[index]
            for data, index in zip(self.cuboid.indexed_data_list, _data_index)
        )
        _complement_data_list = list(
            data[index]
            for data, index in zip(self.cuboid.indexed_data_list, _complement_data_index)
        )
        # Potential Score
        v1, v2 = self._get_column_values('real', self.hotspot.op, _data_list, _complement_data_list)
        f1, f2 = self._get_column_values('predict', self.hotspot.op, _data_list, _complement_data_list)
        v = np.concatenate([v1, v2])
        f = np.concatenate([f1, f2])
        f_, v_ = np.sum(f1), np.sum(v1)
        # f_, v_ = self._get_column_values('predict', np.sum)[0], self._get_column_values('real', np.sum)[0]
        a1 = f1 - (f_ - v_) * f1 / np.maximum(f_, 1e-4)
        a = np.concatenate([a1, f2])
        ps = np.maximum(1. - euclidean(v, a) / np.maximum(euclidean(v, f), 1e-4), 0.)

        self._potential_score = ps
        
    @property
    def potential_score(self):
        return self._potential_score

    @staticmethod
    def _get_column_values(column, op, data_list, complement_data_list, reduction=None):
        if reduction is None:
            reduction = lambda x: x
        ret = (op(*(reduction(_.loc[:, column].values) for _ in data_list)),
               op(*(reduction(_.loc[:, column].values) for _ in complement_data_list)))
        return ret
