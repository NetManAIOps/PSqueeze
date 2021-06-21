from typing import List, FrozenSet

import pandas as pd

from utility import AC
from .adtributor import adtributor
from .adtributor_derived import adtributor_derived
from .r_adtributor import r_adtributor
from .r_adtributor_derived import r_adtributor_derived


class Adtributor:
    def __init__(self, df_list: List[pd.DataFrame], algorithm: str, derived: bool = False):
        self._root_cause = None
        if algorithm.lower() in {'adt', 'adtributor'}:
            if not derived:
                self._root_causes = adtributor(df_list[0])
            else:
                self._root_causes = adtributor_derived(df_list)
        elif algorithm.lower() in {'rad', 'recursive_adtributor', 'r_adtributor', 'radtributor', 'r-adtributor'}:
            if not derived:
                self._root_causes = r_adtributor(df_list[0])
            else:
                self._root_causes = r_adtributor_derived(df_list)
        else:
            raise NotImplementedError(f"algorithm={algorithm}")

    def run(self):
        pass

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
