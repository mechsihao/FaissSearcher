"""
创建人：MECH
创建时间：2022/02/01
功能描述：faiss索引构建检索系统的全流程
"""
import time

from typing import List, Union, Dict
import numpy as np
import pandas as pd
import faiss

from pandas import DataFrame
from numpy import array
from base_encoder import BaseEncoder
from bert_encoder import BertEncoder
import pickle


class FaissSearcher:
    """
    faiss索引构建检索系统的全流程
    """
    def __init__(self, encoder, items: DataFrame, index_param: str, measurement: str, norm_vec: bool = False, **kwargs):
        if isinstance(encoder, BaseEncoder):
            self.encoder = encoder
        else:
            # 兼容通用encoder
            self.encoder = BertEncoder(dict_path=kwargs['dict_path'], encoder=encoder)
        self.index_param = index_param
        self.items = items
        self.norm_vec = True if measurement == 'cos' else norm_vec
        self.metric = self.set_measure_metric(measurement)
        self.measurement = measurement
        self.vec_dim = self.get_vecs(items[items.columns[0]][:1].to_list(), verbose=0).shape[1]
        self.vecs = None
        self.index = None
        self.kwargs = kwargs
        self.doc_feature_sep = kwargs['doc_feature_sep'] if 'doc_feature_sep' in self.kwargs else None
        self.query_feature_sep = kwargs['query_feature_sep'] if 'query_feature_sep' in self.kwargs else None

    def get_vecs(self, items: List[str], verbose: int = 1) -> array:
        vecs = self.encoder.encode(items, verbose=verbose)
        if self.norm_vec:
            return self.__tofloat32__(self.__normvec__(vecs))
        else:
            return self.__tofloat32__(vecs)

    @staticmethod
    def set_measure_metric(measurement):
        metric_dict = {'cos': faiss.METRIC_INNER_PRODUCT,
                       'l1': faiss.METRIC_L1,
                       'l2': faiss.METRIC_L2,
                       'l_inf': faiss.METRIC_Linf,
                       'brayCurtis': faiss.METRIC_BrayCurtis,
                       'l_p': faiss.METRIC_Lp,
                       'canberra': faiss.METRIC_Canberra,
                       'jensen_shannon': faiss.METRIC_JensenShannon}
        if measurement in metric_dict:
            return metric_dict[measurement]
        else:
            raise Exception(f"Do not support measurement: '{measurement}', support measurement is [{', '.join(list(metric_dict.keys()))}]")

    @staticmethod
    def __tofloat32__(vecs):
        return vecs.astype(np.float32)

    @staticmethod
    def __normvec__(vecs):
        return vecs / (vecs ** 2).sum(axis=1, keepdims=True) ** 0.5

    def __build_faiss_index(self):
        if 'hnsw' in self.index_param.lower() and ',' not in self.index_param:
            self.index = faiss.IndexHNSWFlat(self.vec_dim, int(self.index_param.split('HNSW')[-1]), self.metric)
        else:
            self.index = faiss.index_factory(self.vec_dim, self.index_param, self.metric)
        self.index.verbose = True
        self.index.do_polysemous_training = False
        return self

    def load_index(self, index_path):
        print(f"Load index...")
        self.index = faiss.read_index(index_path)
        assert self.index.ntotal == len(self.items), f"Index sample nums {self.index.ntotal} != Items length {len(self.items)}"
        assert self.index.d == self.vec_dim, f"Index dim {self.index.d} != Vecs dim {self.vec_dim}"
        assert self.index.is_trained, "Index dose not trained"

    def train(self):
        print(f"Encode items start...")
        self.vecs = self.get_vecs(self.items[self.items.columns[0]].to_list())
        start_time = time.time()
        vecs = self.__tofloat32__(self.vecs)
        print(f"Train index start...")
        self.__build_faiss_index()
        self.index.train(vecs)
        self.index.add(vecs)
        print(f"Train index cost time: {time.time() - start_time}")

    def search_items(self, target: List[str], indexes: List[List[int]], directories: List[List[float]]) -> DataFrame:
        """
        返回df列名：["source_item", "sim_item", "sim_val" + 原items除第一列外的剩下的列]
        """
        start_time = time.time()
        target = pd.DataFrame(target, columns=['source_item'])
        target['sim_ind'] = list(indexes)
        target['sim_val'] = list(directories)
        target['pair'] = target.apply(lambda x: [[cont[0], cont[1]] for cont in zip(x['sim_ind'], x['sim_val'])], axis=1)
        target = target.drop(columns=['sim_ind', 'sim_val'])
        target = target.explode('pair').reset_index(drop=True)
        target[['sim_ind', 'sim_val']] = pd.DataFrame(target['pair'].to_list(), columns=['sim_ind', 'sim_val'])
        target['sim_val'] = target['sim_val'].values.astype(np.float32)
        sim_item = self.items.iloc[target['sim_ind']].reset_index(drop=True)
        sim_item.columns = ['sim_item'] + list(sim_item.columns[1:])
        target = target.drop(columns=['pair', 'sim_ind'])

        if self.query_feature_sep:
            print(f"[Warning] Out Put Query Will Be Split By '{self.query_feature_sep}'")
            target["source_item"] = target["source_item"].apply(lambda x: str(x).split(self.query_feature_sep)[0])

        if self.doc_feature_sep:
            print(f"[Warning] Out Put Document Will Be Split By '{self.doc_feature_sep}'")
            sim_item["sim_item"] = sim_item["sim_item"].apply(lambda x: str(x).split(self.doc_feature_sep)[0])

        print(f"Find items cost time: {time.time() - start_time}")
        return pd.concat([target, sim_item], axis=1)

    def search(self, target: List[str], topK: Union[int, List[int]]) -> Union[DataFrame, Dict[int, DataFrame]]:
        if self.index:
            target_vec = self.get_vecs(target)
            if isinstance(topK, int):
                start_time = time.time()
                directories, indexes = self.index.search(target_vec, topK)
                print(f"Search index cost time: {time.time() - start_time}")
                return self.search_items(target, indexes, directories)
            elif isinstance(topK, List):
                start_time = time.time()
                res = {}
                for i, k in enumerate(topK):
                    print(f"{i+1}.topK={k}:")
                    directories, indexes = self.index.search(target_vec, k)
                    print(f"Search index cost time: {time.time() - start_time}")
                    res[k] = self.search_items(target, indexes, directories)
                return res
            else:
                raise TypeError(f"TopK dose not support type: {type(topK)}")
        else:
            raise Exception("Faiss dose not train, please use train method before search or load a trained index...")

    def save_index(self, index_save_path):
        faiss.write_index(self.index, index_save_path)

    def cal_sim(self, item1: str, items2: List[str]):
        vec1 = self.encoder.encode([item1], verbose=0)
        vecs2 = self.encoder.encode(items2, verbose=0)
        sim_score_list = list(vec1.dot(vecs2.T))
        sim_df = pd.DataFrame([items2], columns=['item'])
        sim_df['score'] = sim_score_list
        return sim_df.sort_values(by="score", ascending=False)

    def save_searcher(self, path):
        file = open(path, "wb")
        pickle.dump(self, file)
        file.close()

    @staticmethod
    def load_searcher(path):
        file = open(path, "rb")
        return pickle.load(file)
