import copy
import random
from utility.attribute_combination import AttributeCombination
from .hotspot_node import HotSpotNode, HotSpotElement
import typing
from loguru import logger
import time
import hotspotpy
# from memory_profiler import profile


K = 3


class Cuboid(object):
    def __init__(self, hotspot: "hotspotpy.HotSpot", attribute_names: typing.Tuple[str], max_steps=None, max_time=None,
                 ps_upper_threshold=None, ps_lower_threshold=None, exploration_rate=1.414, ):
        self.ps_lower_threshold = ps_lower_threshold if ps_lower_threshold else float('-inf')
        self.exploration_rate = exploration_rate
        self.ps_upper_threshold = ps_upper_threshold if ps_upper_threshold else float('+inf')
        self.max_time = max_time
        self.max_steps = max_steps
        self.attribute_names = sorted(attribute_names)
        self.hotspot = hotspot
        self._finished = False
        self.elements = False  # type: typing.List['HotSpotElement']
        self.root = None
        self._best_nodes = False  # type: typing.List['HotSpotNode']
        self._visited_nodes = set()  # type: typing.Set['HotSpotNode']
        self.step_cnt = 0
        self.begin_time = None
        self._layer = len(self.attribute_names)
        self.indexed_data_list = list(_.set_index(self.attribute_names).sort_index() for _ in self.hotspot.data_list)

    # @profile
    def run(self):
        if self._finished:
            logger.warning(f"try to rerun {self}")
            return
        logger.debug(f"{self} start run")
        self._get_sorted_elements_and_root()  # get and sort elements
        self.begin_time = time.time()
        if self.max_steps is not None and self.max_time is not None:
            should_terminate = lambda: (self.step_cnt >= self.max_steps) \
                                       or ((time.time() - self.begin_time) > self.max_time)
        elif self.max_steps is not None:
            should_terminate = lambda: self.step_cnt >= self.max_steps
        elif self.max_time is not None:
            should_terminate = lambda: (time.time() - self.begin_time > self.max_time)
        else:
            raise RuntimeError('one of max_steps and max_time should not be None')

        while not should_terminate():
            # step_tic = time.time()
            self.step_cnt += 1
            if not self.run_step():
                break
            # logger.debug(f"step {self.step_cnt}, time: {time.time() - step_tic}s")
        self._best_nodes = sorted(self._visited_nodes, key=lambda x: x.result_score, reverse=True)[:K]
        # lo = self._filter_sorted_element(self._best_nodes, self.ps_lower_threshold)
        # self._best_nodes = self._best_nodes[:lo]
        # self._best_nodes = list(filter(lambda x: x.potential_score >= self.ps_lower_threshold, self._best_nodes))
        self._finished = True
        # delete unused attributes to save memory
        self._visited_nodes = None
        self.indexed_data_list = None
        # self._visited_nodes = None
        logger.debug(f"{self} finished")
        return self

    @property
    def best_nodes(self) -> typing.List['HotSpotNode']:
        assert self._finished, f'{self} has not run.'
        return self._best_nodes

    def __str__(self):
        return f"Cuboid {self.attribute_names}"

    # @profile
    def run_step(self):
        if self.root.wasted:
            logger.debug('MCTS exited because all nodes searched')
            return False
        node = self.select_node(self.root)
        # logger.debug(f"ret: {node}")
        if node == self.root:
            return True
        self._visited_nodes.add(node)
        node.back_propagate()
        if node.potential_score >= self.ps_upper_threshold:
            return False
        return True

    def select_node(self, node):
        # logger.debug("New Step")
        while not node.is_terminal():
            # logger.debug(f'search: {node}')
            best_node = node.get_best_child_by_ucb(exploration_rate=self.exploration_rate)
            best_reward = -1 if best_node is None else best_node.reward
            if node.is_fully_expanded or random.uniform(0, 1) <= best_reward:
                # logger.debug(f'try to find best children: best_node: {best_node}')
                if best_node is not None:
                    node = best_node
                else:
                    assert node.is_fully_expanded
                    node.wasted = True  # node is fully expanded, and all children are wasted, so it is wasted
                    return node  # all children are pruned, the same as this node is terminal
            else:
                new_node = self.expand_node_or_fail(node)  # may not be new, new_node = None if expansion fails
                if new_node is None:  # expansion fails, and now node is fully expanded
                    continue
                if new_node.is_terminal():
                    new_node.wasted = True
                return new_node
        return node

    def expand_node_or_fail(self, node):
        assert not node.is_fully_expanded, f"node {node} should not be fully expanded"
        assert node.has_next_action, f'node {node} has no actions'
        for action in node.get_next_action():
            if action not in node.children:
                new_node = node.take_action(action)
                if new_node in self._visited_nodes:
                    continue
                node.children.add(new_node)
                return new_node
        # return None indicates that this node is leaf node, can't be expanded
        node.is_fully_expanded = True
        # logger.debug(f"node is fully expand {node}")
        # logger.debug(f"children: {' '.join(str(_) for _ in node.children)}")
        return None

    def _get_sorted_elements_and_root(self):
        root_attribute_combination = AttributeCombination.get_root_attribute_combination(self.hotspot.attribute_names)
        self.root = HotSpotNode(
            self.hotspot, self, None, frozenset({root_attribute_combination}))
        if self.layer == 1:
            values = self.hotspot.attribute_values[self.attribute_names[0]]
            self.elements = [
                HotSpotElement(
                    self.hotspot, self, self.root,
                    frozenset({root_attribute_combination.copy_and_update(
                        {self.attribute_names[0]: value}
                    )})
                ) for value in values]
        else:
            self.elements = []
            for cuboid in self.hotspot.cuboids[self.layer - 1]:  # cuboid in last layer
                key = (set(self.attribute_names) - set(cuboid.attribute_names))
                if len(key) == 1:
                    key = key.pop()
                else:
                    continue
                values = self.hotspot.attribute_values[key]
                for element in cuboid.elements:
                    if element.potential_score < self.ps_lower_threshold:
                        continue
                    logger.debug(f"expand element from {element}")
                    self.elements.extend([
                        HotSpotElement(
                            self.hotspot, self, self.root,
                            frozenset({element.attribute_combination.copy_and_update(
                                {key: value}
                            )})
                        )
                        for value in values])
        self.elements = sorted(list(set(self.elements)), key=lambda x: x.potential_score, reverse=True)
        logger.debug(f"number of element after  filter: {len(self.elements)}")
        self._visited_nodes = set(self.elements)
        self.root.children = copy.copy(self._visited_nodes)
        self.root.is_fully_expanded = True
        self.root.has_next_action = False
        for ele in self.elements:
            ele.back_propagate()
            logger.debug(f"{ele}")

    @staticmethod
    def _filter_sorted_element(nodes_array, lower_bound):
        lo = 0
        hi = len(nodes_array)
        while lo < hi:
            m = (lo + hi) // 2
            if nodes_array[m].potential_score > lower_bound:
                lo = m + 1
            else:
                hi = m
        return lo

    def __eq__(self, other):
        return frozenset(self.attribute_names) == frozenset(other.attribute_names) \
               and id(self.hotspot) == id(other.hotspot)

    @property
    def layer(self):
        return self._layer
