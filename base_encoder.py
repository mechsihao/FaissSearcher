from typing import List
from numpy import array


class BaseEncoder(object):
    """
    传入FaissIndex的encoder模型基类
    """
    def encode(self, items: List[str], verbose: int = 1) -> array:
        raise NotImplemented
