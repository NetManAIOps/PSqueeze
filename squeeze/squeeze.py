from functools import lru_cache
from itertools import combinations
import pandas as pd
from typing import List, FrozenSet, Dict, Union
from loguru import logger
from scipy.stats import entropy, norm
from sklearn.metrics import log_loss
from typing import Tuple
from utility import AttributeCombination as AC, AttributeCombination
from bidict import bidict
import numpy as np
from squeeze.anomaly_amount_fileter import KPIFilter
from squeeze.squeeze_option import SqueezeOption
from squeeze.clustering import cluster_factory
from squeeze.clustering.cluster_plot import plot_cluster
from scipy.spatial.distance import cityblock, euclidean
from scipy.special import factorial


class Squeeze:
    def __init__(self, data_list: List[pd.DataFrame], op=lambda x: x, option: SqueezeOption = SqueezeOption()):
        """
        :param data_list: dataframe without index,
            must have 'real' and 'predict' columns, other columns are considered as attributes
            all elements in this list must have exactly the same attribute combinations in the same order
        """
        self.option = option
        self.info_collect = {
            "scores": [],
            "ranks": [],
            "n_eles": [],
            "layers": [],
        }

        self.one_dim_cluster = cluster_factory(self.option)  # DensityBased1dCluster(option)
        self.cluster_list = []  # type: List[np.ndarray]

        # valid_idx = np.logical_or.reduce(
        #     [(_.predict > 0) | (_.real > 0) for _ in data_list],
        # )
        valid_idx = (data_list[-1].predict > 0) | (data_list[-1].real > 0)

        self.data_list = list(_[valid_idx] for _ in data_list)
        self.op = op
        self.derived_data = self.get_derived_dataframe(None)  # type: pd.DataFrame
        # There is an error in injection
        self.derived_data.real -= min(np.min(self.derived_data.real), 0)

        self.attribute_names = list(sorted(set(self.derived_data.columns) - {'real', 'predict'}))
        logger.debug(f"available attributes: {self.attribute_names}")

        self.data_list = list(map(lambda x: x.sort_values(by=self.attribute_names), self.data_list))
        self.derived_data.sort_values(by=self.attribute_names, inplace=True)
        self.derived_data['loc'] = np.arange(len(self.derived_data))

        self.attribute_values = list(list(set(self.derived_data.loc[:, name].values)) for name in self.attribute_names)
        logger.debug(f"available values: {self.attribute_values}")

        self.ac_array = np.asarray(
            [AC(**record) for record in self.derived_data[self.attribute_names].to_dict(orient='records')])

        self._v = self.derived_data['real'].values
        self._f = self.derived_data['predict'].values
        assert all(self._v >= 0) and all(self._f >= 0), \
            f"currently we assume that KPIs are non-negative, {self.derived_data[~(self._f >= 0)]}"

        self.__finished = False
        self._root_cause = []

        self.filtered_indices = None

    @property
    @lru_cache()
    def root_cause(self):
        return self._root_cause

    @property
    @lru_cache()
    def report(self) -> str:
        cluster_impacts = [
            np.sum(np.abs(self._f[idx[0]] - self._v[idx[0]])) for idx in self.cluster_list
        ]
        unique_root_cause, rc_indies = np.unique(self.root_cause, return_index=True)
        cluster_impacts = [
            np.sum(cluster_impacts[idx]) for idx in rc_indies
        ]
        logger.debug(f"{unique_root_cause}, {cluster_impacts}")
        report_df = pd.DataFrame(columns=['root_cause', 'impact'])
        report_df['root_cause'] = list(AC.batch_to_string(_) for _ in unique_root_cause)
        report_df['impact'] = cluster_impacts
        report_df.sort_values(by=['impact'], inplace=True, ascending=False)
        return report_df.to_csv(index=False)

    @lru_cache()
    def get_cuboid_ac_array(self, cuboid: Tuple[str, ...]):
        return np.asarray(list(map(lambda x: x.mask(cuboid), self.ac_array)))

    @lru_cache()
    def get_indexed_data(self, cuboid: Tuple[str, ...]):
        return self.derived_data.set_index(list(cuboid)).sort_index()

    @property
    @lru_cache()
    def normal_indices(self):
        if self.option.psqueeze:
            # NOTE: for PSqueeze
            abnormal = np.concatenate(self.cluster_list)
            idx = np.argsort(
                np.abs(
                    self.choose_from_2darray(
                        self.leaf_deviation_score_with_variance,
                        abnormal[:, 0],
                        abnormal[:, 1]
                    )
                )
            )
            abnormal = abnormal[idx]
            upper_bound = np.abs(self.leaf_deviation_score_with_variance[abnormal[0][0]][abnormal[0][1]])
            normal = np.where(np.max(np.abs(self.leaf_deviation_score_with_variance), axis=1) < upper_bound)[0]
            return normal
        else:
            abnormal = np.sort(np.concatenate(self.cluster_list))
            idx = np.argsort(np.abs(self.leaf_deviation_score[abnormal]))
            abnormal = abnormal[idx]
            normal = np.where(np.abs(self.leaf_deviation_score) < np.abs(self.leaf_deviation_score[abnormal[0]]))[0]
            # normal = np.setdiff1d(np.arange(len(self.derived_data)), abnormal, assume_unique=True)
            # return np.intersect1d(normal, self.filtered_indices, assume_unique=True)
            return normal

    def run(self):
        if self.__finished:
            logger.warning(f"try to rerun {self}")
            return self
        if self.option.enable_filter:
            kpi_filter = KPIFilter(self._v, self._f)
            self.filtered_indices = kpi_filter.filtered_indices
            if self.option.psqueeze:
                # NOTE: for PSqueeze
                cluster_list, plot_kwargs = self.one_dim_cluster(
                    self.leaf_deviation_score_with_variance[self.filtered_indices],
                    self.leaf_deviation_weights_with_variance[self.filtered_indices]
                )
                cluster_list = list(
                    [np.array([
                        kpi_filter.inverse_map((_ / 3).astype(int)),
                        _ % 3
                    ]).T
                     for _ in cluster_list]  # NOTE: for PSqueeze, each cluster is np.ndarray([[index, bias]...])
                )
                cluster_list = list(
                    [np.array(list(
                        filter(lambda x:
                               np.min(
                                   self.choose_from_2darray(self.leaf_deviation_score_with_variance, _[:, 0], _[:, 1]))
                               <= self.leaf_deviation_score_with_variance[int(x / 3), x % 3] <=
                               np.max(
                                   self.choose_from_2darray(self.leaf_deviation_score_with_variance, _[:, 0], _[:, 1])),
                               np.arange(len(self.leaf_deviation_score_with_variance.flatten())))
                    ))
                        for _ in cluster_list]
                )
                # NOTE: for PSqueeze: each cluster is np.ndarray([[index, bias]...])
                cluster_list = list([
                    np.array([(_ / 3).astype(int), _ % 3]).T
                    for _ in cluster_list])
            else:
                cluster_list, plot_kwargs = self.one_dim_cluster(self.leaf_deviation_score[self.filtered_indices])
                cluster_list = list([kpi_filter.inverse_map(_) for _ in cluster_list])
                cluster_list = list(
                    [list(
                        filter(lambda x: np.min(self.leaf_deviation_score[_]) <= self.leaf_deviation_score[x] <= np.max(
                            self.leaf_deviation_score[_]), np.arange(len(self._f)))
                    )
                        for _ in cluster_list]
                )

            self.cluster_list = cluster_list
        else:
            self.filtered_indices = np.ones(len(self._v), dtype=bool)
            self.cluster_list, plot_kwargs = self.one_dim_cluster(self.leaf_deviation_score)

        if self.option.debug:
            plot_kwargs.update({
                "save_path": self.option.fig_save_path.format(suffix="_density_cluster"),
                "f_values": self._f,
                "v_values": self._v,
                "clusters": self.cluster_list,
            })
            plot_cluster(**plot_kwargs)

        self.locate_root_cause()
        self.__finished = True
        self._root_cause = self._root_cause
        return self

    def _locate_in_cuboid(self, cuboid, indices, **params) -> Tuple[FrozenSet[AC], float]:
        """
        :param cuboid: try to find root cause in this cuboid
        :param indices: anomaly leaf nodes' indices
        :return: root causes and their score
        """
        variance_map = {}

        def array2map(x):
            if x[0] in variance_map:
                variance_map[x[0]].append(x[1])
            else:
                variance_map[x[0]] = [x[1]]

        if type(indices) == np.ndarray:  # NOTE: for PSqueeze
            np.apply_along_axis(func1d=array2map, arr=indices, axis=1)
            indices = np.unique(indices[:, 0]).tolist()
        # assert len(self.data_list) == 1

        data_cuboid_indexed = self.get_indexed_data(cuboid)
        logger.debug(f"current cuboid: {cuboid}")

        abnormal_cuboid_ac_arr = self.get_cuboid_ac_array(cuboid)[indices]

        if variance_map:  # NOTE: for PSqueeze
            elements_count = {}
            for i in range(len(indices)):
                idx = indices[i]
                ele = abnormal_cuboid_ac_arr[i]
                # weight = self.leaf_deviation_weights_with_variance[idx][variance_map[idx]].max() # NOTE: max or sum?
                weight = np.sum(self.leaf_deviation_weights_with_variance[idx][variance_map[idx]])
                if ele in elements_count:
                    elements_count[ele] += weight
                else:
                    elements_count[ele] = weight
            elements = np.array(list(elements_count.keys()))
            num_elements = np.array(list(elements_count.values()))
            del elements_count
        else:
            elements, num_elements = np.unique(abnormal_cuboid_ac_arr, return_counts=True)

        num_ele_descents = np.asarray(list(
            np.count_nonzero(
                _.index_dataframe(data_cuboid_indexed),
            ) for _ in elements
        ))

        # sort reversely by descent score
        descent_score = num_elements / np.maximum(num_ele_descents, 1e-4)
        idx = np.argsort(descent_score)[::-1]
        elements = elements[idx]
        num_ele_descents = num_ele_descents[idx]
        num_elements = num_elements[idx]

        # descent_score = descent_score[idx]
        del descent_score

        logger.debug(f"elements: {';'.join(str(_) for _ in elements)}")

        def _root_cause_score(partition: int, sm='ps') -> float:
            dis_f = cityblock
            data_p, data_n = self.get_derived_dataframe(
                frozenset(elements[:partition]), cuboid=cuboid,
                reduction=lambda x: x, return_complement=True,
                subset_indices=np.concatenate([indices, self.normal_indices]))
            # logger.info(data_p.shape)
            # logger.info(data_n.shape)
            assert len(data_p) + len(data_n) == len(indices) + len(self.normal_indices), \
                f'{len(data_n)} {len(data_p)} {len(indices)} {len(self.normal_indices)}'

            # dp = self.__deviation_score(data_p['real'].values, data_p['predict'].values)
            # dn = self.__deviation_score(data_n['real'].values, data_n['predict'].values) if len(data_n) else []
            # log_ll = np.mean(norm.pdf(dp, loc=mu, scale=sigma)) \
            #          + np.mean(norm.pdf(dn, loc=0, scale=self.option.normal_deviation_std))
            # _abnormal_descent_score = np.sum(num_elements[:partition]) / np.sum(num_ele_descents[:partition])
            # _normal_descent_score = 1 - np.sum(num_elements[partition:] / np.sum(num_ele_descents[partition:]))
            # _ds = _normal_descent_score * _abnormal_descent_score
            # succinct = partition + len(cuboid) * len(cuboid)

            def ps(se=lambda x: 1):
                _v1, _v2 = data_p.real.values, data_n.real.values
                _pv, _pf = np.sum(_v1), np.sum(data_p.predict.values)
                _f1, _f2 = data_p.predict.values, data_n.predict.values

                reduced_data_p, _ = self.get_derived_dataframe(
                    frozenset(elements[:partition]), cuboid=cuboid,
                    reduction="sum", return_complement=True,
                    subset_indices=np.concatenate([indices, self.normal_indices]))
                if len(reduced_data_p):
                    if abs(reduced_data_p.predict.item()) > 1e-4:
                        _a1, _a2 = data_p.predict.values * (
                                reduced_data_p.real.item() / reduced_data_p.predict.item()
                        ), data_n.predict.values
                    else:
                        _a1, _a2 = data_p.predict.values + reduced_data_p.real.item(), data_n.predict.values

                else:
                    # print(elements[:partition], data_p, reduced_data_p)
                    assert len(data_p) == 0
                    _a1 = 0
                    _a2 = data_n.predict.values

                if self.option.dis_norm:
                    deno_1 = np.maximum(_v1, _f1).clip(1e-6)
                    deno_2 = np.maximum(_v2, _f2).clip(1e-6)
                    _v1, _a1, _f1 = _v1 / deno_1, _a1 / deno_1, _f1 / deno_1
                    _v2, _a2, _f2 = _v2 / deno_2, _a2 / deno_2, _f2 / deno_2

                divide = lambda x, y: x / y if y > 0 else (0 if x == 0 else float('inf'))
                return 1 - (divide(dis_f(_v1, _a1), len(_v1)) * se(_v1) + divide(dis_f(_v2, _f2), len(_v2)) * se(_v2)) \
                       / (divide(dis_f(_v1, _f1), len(_v1)) * se(_v1) + divide(dis_f(_v2, _f2), len(_v2)) * se(_v2))

            def ji():
                cluster_data = self.derived_data.iloc[indices]
                data_p_in_cluster = pd.merge(cluster_data, data_p, how='inner')
                return data_p_in_cluster.size / (cluster_data.size + data_p.size - data_p_in_cluster.size)

            # ps, ji = ps(), ji()
            # partition_gain = np.log(partition+1) / partition
            # partition_gain = 1 / partition ** 2
            # size_log = np.log(len(indices))
            # size_gain = np.log(size_log+1) / size_log if size_log > 0 else 1
            # size_gain = 1 / (size_log + 1)
            # pjavg = (ji ** (size_gain*partition_gain)) * (ps ** (1-size_gain*partition_gain)) if ps > 0 else 0.
            # pjavg = ji * (size_gain * partition_gain) + ps * (1 - size_gain * partition_gain)

            if sm == 'auto':
                return [
                    ("ps", ps()),
                    ("ji", ji()),
                    ("pps", ps(lambda x: np.log(max(len(x), 1)))),
                ]
            elif sm == 'ps':
                _ps = ps()
            elif sm == 'ji':
                _ps = ji()
            elif sm == 'pjavg':
                _ps = (ps() * ji()) ** 0.5 if ps() > 0 else 0
            elif sm == 'pps':
                _ps = ps(lambda x: np.log(max(len(x), 1)))
            else:
                raise RuntimeError("bad score measure")

            logger.debug(
                f"partition:{partition} "
                # f"log_ll:{log_ll} "
                # f"impact: {impact_score} "
                # f"succinct: {succinct} "
                f"ps: {_ps}"
            )
            # return _p * self.option.score_weight / (-succinct)
            return _ps

        partitions = np.arange(
            min(
                len(elements),
                self.option.max_num_elements_single_cluster,
                len(set(self.get_indexed_data(cuboid).index.values)) - 1
            )
        ) + 1
        if len(partitions) <= 0:
            return elements, float('-inf')

        if self.option.score_measure == 'auto':
            sm = sorted(_root_cause_score(1, 'auto'), key=lambda x: x[1])[-1][0]
        else:
            sm = self.option.score_measure

        logger.debug(f"score measure: {sm}")
        rc_scores = np.asarray(list(map(lambda x: _root_cause_score(x, sm=sm), partitions)))
        idx = np.argsort(rc_scores)[::-1]
        partitions = partitions[idx]
        rc_scores = rc_scores[idx]

        score = rc_scores[0]
        rc = elements[:partitions[0].item()]
        logger.debug(f"cuboid {cuboid} gives root cause {AC.batch_to_string(rc)} with score {score}")
        return rc.tolist(), score

    def _locate_in_cluster(self, indices: np.ndarray):
        """
        :param indices:  indices of leaf nodes in this cluster (list or np.ndarray)
        :return: None
        """
        if self.option.psqueeze:  # NOTE: for PSqueeze
            non_variance = indices[np.where(indices[:, 1] == 1)[0]]
            # too many variance data
            if non_variance.shape[0] * self.option.non_var_split_ratio < indices.shape[0]:
                logger.info(f"too many variance data, {non_variance.shape[0]} in {indices.shape[0]}")
                if non_variance.size == 0:
                    logger.info("no non-variance cluster")
                    return
                indices = non_variance
            del non_variance

            score_samples = self.choose_from_2darray(
                self.leaf_deviation_score_with_variance,
                indices[:, 0],
                indices[:, 1]
            )
            mu = np.mean(score_samples)
            sigma = np.maximum(np.std(score_samples), 1e-4)
            logger.debug(f"locate in cluster: {mu}(+-{sigma})")
            logger.debug(f"cluster indices: {indices.shape}")
            # logger.debug(f"cluster indices: {indices}")

            # def detail_print(idx):
            #     print(f"data at line {idx}:")
            #     print(f"\t_v: {self._v[idx]}, _f: {self._f[idx]}")
            #     print("\tdeviation score:", end="\t")
            #     print(self.leaf_deviation_score_with_variance[idx][0], end="\t")
            #     print(self.leaf_deviation_score_with_variance[idx][1], end="\t")
            #     print(self.leaf_deviation_score_with_variance[idx][2])
            #     print("\tdeviation weights:", end="\t")
            #     print(self.leaf_deviation_weights_with_variance[idx][0], end="\t")
            #     print(self.leaf_deviation_weights_with_variance[idx][1], end="\t")
            #     print(self.leaf_deviation_weights_with_variance[idx][2])
            # detail_print(809)
            # input()

            max_cuboid_layer = len(self.attribute_names)
            ret_lists = []
            for cuboid_layer in np.arange(max_cuboid_layer) + 1:
                layer_ret_lists = list(map(
                    lambda x, _i=indices, _mu=mu, _sigma=sigma: self._locate_in_cuboid(x, indices=_i, mu=_mu,
                                                                                       sigma=_sigma),
                    combinations(self.attribute_names, cuboid_layer)
                ))
                if self.option.psqueeze:
                    ret_lists.extend([
                        {
                            'rc': x[0], 'score': x[1], 'n_ele': len(x[0]), 'layer': cuboid_layer,
                            'rank': x[1] * self.option.score_weight - len(x[0]) * cuboid_layer ** 2
                        } for x in layer_ret_lists
                    ])
                else:
                    ret_lists.extend([
                        {
                            'rc': x[0], 'score': x[1], 'n_ele': len(x[0]), 'layer': cuboid_layer,
                            'rank': x[1] * self.option.score_weight - len(x[0]) * cuboid_layer
                        } for x in layer_ret_lists
                    ])
                if len(list(filter(lambda x: x['score'] > self.option.ps_upper_bound, ret_lists))):
                    break
        else:
            mu = np.mean(self.leaf_deviation_score[indices])
            sigma = np.maximum(np.std(self.leaf_deviation_score[indices]), 1e-4)
            logger.debug(f"locate in cluster: {mu}(+-{sigma})")
            logger.debug(f"cluster indices: {len(indices)}")
            # logger.debug(f"cluster indices: {indices}")
            max_cuboid_layer = len(self.attribute_names)
            ret_lists = []
            for cuboid_layer in np.arange(max_cuboid_layer) + 1:
                layer_ret_lists = list(map(
                    lambda x, _i=indices, _mu=mu, _sigma=sigma: self._locate_in_cuboid(x, indices=_i, mu=_mu,
                                                                                       sigma=_sigma),
                    combinations(self.attribute_names, cuboid_layer)
                ))
                ret_lists.extend([
                    {
                        'rc': x[0], 'score': x[1], 'n_ele': len(x[0]), 'layer': cuboid_layer,
                        'rank': x[1] * self.option.score_weight - len(x[0]) * cuboid_layer
                    } for x in layer_ret_lists
                ])
                if len(list(filter(lambda x: x['score'] > self.option.ps_upper_bound, ret_lists))):
                    break
        ret_lists = list(sorted(
            ret_lists,
            key=lambda x: x['rank'],
            reverse=True)
        )
        if ret_lists:
            ret = ret_lists[0]['rc']
            logger.debug(
                f"find root cause: {AC.batch_to_string(ret)}, rank: {ret_lists[0]['rank']}, score: {ret_lists[0]['score']}")
            logger.debug(
                f"candidate: {list(map(lambda x: (AC.batch_to_string(x['rc']), x['score'], x['rank']), ret_lists[:min(3, len(ret_lists))]))}")
            self._root_cause.append(frozenset(ret))
            self.info_collect["scores"].append(ret_lists[0]['score'])
            self.info_collect["ranks"].append(ret_lists[0]['rank'])
            self.info_collect["n_eles"].append(ret_lists[0]['n_ele'])
            self.info_collect["layers"].append(ret_lists[0]['layer'])
        else:
            logger.info("failed to find root cause")

    def update_info_collect(self):
        def update_info(d, k):
            if len(d[k]) == 0:
                d[f"{k}_mean"] = float("nan")
                d[f"{k}_std"] = float("nan")
                d[f"{k}_max"] = float("nan")
                d[f"{k}_min"] = float("nan")
            else:
                d[f"{k}_mean"] = float(np.mean(d[k]))
                d[f"{k}_std"] = float(np.std(d[k]))
                d[f"{k}_max"] = float(np.max(d[k]))
                d[f"{k}_min"] = float(np.min(d[k]))
            del d[k]

        update_info(self.info_collect, "scores")
        update_info(self.info_collect, "ranks")
        update_info(self.info_collect, "n_eles")
        update_info(self.info_collect, "layers")

    def locate_root_cause(self):
        if not self.cluster_list:
            logger.info("We do not have abnormal points")
            self.update_info_collect()
            return
        if self.option.score_weight == 'auto':
            if self.option.psqueeze:
                num_sample = len(self._f) * 3

                n_cluster = len(self.cluster_list)
                n_attr = sum(len(_) for _ in self.attribute_values)
                cover_rate = sum(len(_) for _ in self.cluster_list) / num_sample

                n_cluster_gain = np.log(n_cluster + 1) / n_cluster if n_cluster > 0 else 1
                cover_rate_gain = - np.log(min(max(cover_rate, 1e-9), 1 - 1e-9))
                n_attr_gain = n_attr / np.log(n_attr + 1) if n_attr > 0 else 1

                self.option.score_weight = n_cluster_gain * cover_rate_gain * n_attr_gain

                logger.debug(
                    f"auto score weight:"
                    f" {self.option.score_weight} = {n_cluster_gain} * {cover_rate_gain} * {n_attr_gain}")
                logger.debug(f"n_cluster: {n_cluster}")
                logger.debug(f"n_attr: {n_attr}")
                logger.debug(f"cover_rate: {cover_rate}")
            else:
                self.option.score_weight = - np.log(
                    len(self.cluster_list) *
                    sum(len(_) for _ in self.cluster_list) / len(self._f)) / np.log(
                    sum(len(_) for _ in self.attribute_values)) * sum(len(_) for _ in self.attribute_values)
                logger.debug(f"auto score weight: {self.option.score_weight}")

            self.info_collect["score_weight"] = self.option.score_weight
        for indices in self.cluster_list:
            self._locate_in_cluster(indices)
        self.update_info_collect()

    @property
    @lru_cache()
    def leaf_deviation_score_with_variance(self, min=0) -> np.ndarray:
        '''
        Return: numpy.array([[D(-1), D(0), D(+1)],...])
        '''
        # NOTE: for PSqueeze
        with np.errstate(divide='ignore', invalid='ignore'):
            _minus = self.__deviation_score((self._v - self.option.bias).clip(min=min), self._f)
            _origin = self.__deviation_score(self._v, self._f)
            _plus = self.__deviation_score(self._v + self.option.bias, self._f)
            deviation_scores = np.array((_minus, _origin, _plus)).T
            del _minus, _origin, _plus
        assert np.shape(deviation_scores)[0] == np.shape(self._v)[0] == np.shape(self._f)[0], \
            f"bad deviation score shape {np.shape(deviation_scores)}"
        assert np.sum(np.isnan(deviation_scores)) == 0, \
            f"there are nan in deviation score {np.where(np.isnan(deviation_scores))}"
        assert np.sum(~np.isfinite(deviation_scores)) == 0, \
            f"there are infinity in deviation score {np.where(~np.isfinite(deviation_scores))}"
        logger.debug(f"anomaly ratio ranges in [{np.min(deviation_scores)}, {np.max(deviation_scores)}]")
        logger.debug(f"anomaly ratio ranges in [{np.min(deviation_scores[:, 1])}, {np.max(deviation_scores[:, 1])}]")
        return deviation_scores

    @property
    @lru_cache()
    def leaf_deviation_weights_with_variance(self) -> np.ndarray:
        '''
        Return: numpy.array([[W(-1), W(0), W(+1)],...])
        '''
        # NOTE: for PSqueeze
        histogram_weights = self.__variance_weights(self._v, self._f, self.option.bias, self.option.bias)
        assert np.shape(histogram_weights)[0] == np.shape(self._v)[0] == np.shape(self._f)[0], \
            f"bad histogram weights shape {np.shape(histogram_weights)}"
        assert np.sum(np.isnan(histogram_weights)) == 0, \
            f"there are nan in histogram weights {np.where(np.isnan(histogram_weights))}"
        assert np.sum(~np.isfinite(histogram_weights)) == 0, \
            f"there are infinity in histogram weights {np.where(~np.isfinite(histogram_weights))}"
        return histogram_weights

    @property
    @lru_cache()
    def leaf_deviation_score(self):
        with np.errstate(divide='ignore', invalid='ignore'):
            deviation_scores = self.__deviation_score(self._v, self._f)
        assert np.shape(deviation_scores) == np.shape(self._v) == np.shape(self._f)
        assert np.sum(np.isnan(deviation_scores)) == 0, \
            f"there are nan in deviation score {np.where(np.isnan(deviation_scores))}"
        assert np.sum(~np.isfinite(deviation_scores)) == 0, \
            f"there are infinity in deviation score {np.where(~np.isfinite(deviation_scores))}"
        logger.debug(f"anomaly ratio ranges in [{np.min(deviation_scores)}, {np.max(deviation_scores)}]")
        return deviation_scores

    def get_derived_dataframe(self, ac_set: Union[FrozenSet[AC], None], cuboid: Tuple[str] = None,
                              reduction=None, return_complement=False, subset_indices=None):
        subset = np.zeros(len(self.data_list[0]), dtype=np.bool)
        if subset_indices is not None:
            subset[subset_indices] = True
        else:
            subset[:] = True

        if reduction == "sum":
            reduce = lambda x, _axis=0: np.sum(x, axis=_axis, keepdims=True)
        else:
            reduce = lambda x: x

        if ac_set is None:
            idx = np.ones(shape=(len(self.data_list[0]),), dtype=np.bool)
        else:
            idx = AC.batch_index_dataframe(ac_set, self.get_indexed_data(cuboid))

        def _get_ret(_data_list):
            if len(_data_list[0]) == 0:
                return pd.DataFrame(data=[], columns=['real', 'predict'])
            _values = self.op(*[reduce(_data[["real", "predict"]].values) for _data in _data_list])
            if np.size(_values) == 0:
                _values = []
            if reduction == 'sum':
                _ret = pd.DataFrame(data=_values, columns=['real', 'predict'])
            else:
                _ret = _data_list[0].copy(deep=True)
                _ret[['real', 'predict']] = _values
            return _ret

        data_list = list(_[idx & subset] for _ in self.data_list)
        if not return_complement:
            return _get_ret(data_list)
        complement_data_list = list(_[(~idx) & subset] for _ in self.data_list)
        return _get_ret(data_list), _get_ret(complement_data_list)

    @staticmethod
    def __deviation_score(v, f):
        n = 1
        with np.errstate(divide='ignore'):
            ret = n * (f - v) / (n * f + v)
            # ret = np.log(np.maximum(v, 1e-10)) - np.log(np.maximum(f, 1e-10))
            # ret = (2 * sigmoid(1 - v / f) - 1)
            # k = np.log(np.maximum(v, 1e-100)) - np.log(np.maximum(f, 1e-100))
            # ret = (1 - k) / (1 + k)
        ret[np.isnan(ret)] = 0.
        return ret

    @staticmethod
    def __variance_weights(v, f, bias, min):
        # NOTE: for PSqueeze
        with np.errstate(divide='ignore', invalid='ignore'):
            _v = (v + 0.5).astype(int).clip(min=min)  # round to integer
            _variance_v = np.array((_v - bias, _v, _v + bias)).T
            possion_prob = lambda x: (x[1] ** x) * (np.math.e ** (-x[1])) / factorial(x)
            ret = np.apply_along_axis(func1d=possion_prob, axis=1, arr=_variance_v)
            # ret = np.apply_along_axis(func1d=lambda x: x/x[1], axis=1, arr=ret) # normalization on each line
            ret = np.apply_along_axis(func1d=lambda x: x / np.sum(x), axis=1, arr=ret)  # normalization on each line
        # NOTE: error strategy here
        ret[:, 1][np.isnan(ret[:, 1])] = 1.
        ret[:, 1][np.isinf(ret[:, 1])] = 1.
        ret[np.isnan(ret)] = 0.
        ret[np.isinf(ret)] = 0.
        return ret

    @staticmethod
    def choose_from_2darray(source: np.ndarray, x_index_list, y_index_list):
        return source[x_index_list, y_index_list]
