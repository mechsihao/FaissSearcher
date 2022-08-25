"""
创建人：MECH
创建时间：2022/02/01
功能描述：向量白化操作，用于解决向量空间坍缩问题，可以向量提升相似度的准确性
"""

import pandas as pd
import numpy as np


class VecsWhitening(object):
    """
    向量whitening实现，用于调整向量基底，使其近似等于标准正交基，
    并且还可以做到降维，一般来说可以提升句向量的表达效果
    """

    def __init__(self, n_components):
        """和sklearn中的pca的用法一样
        """
        self.n_components = n_components
        self.kernel = None
        self.bias = None
        self.origin_dim = None

    def compute_kernel_bias(self, vecs):
        """计算kernel和bias
        vecs.shape = [num_samples, embedding_size]，
        最后的变换：y = (x + bias).dot(kernel)
        """
        mu = vecs.mean(axis=0, keepdims=True)
        cov = np.cov(vecs.T)
        u, s, _ = np.linalg.svd(cov)
        W = np.dot(u, np.diag(1 / np.sqrt(s)))
        return W[:, :self.n_components], -mu

    def fit(self, vecs):
        self.origin_dim = vecs.shape[1]
        if self.origin_dim >= self.n_components:
            self.kernel, self.bias = self.compute_kernel_bias(vecs)
        else:
            raise Exception("n_components must smaller than vecs original dim")
        return self

    def transform(self, vecs):
        if self.kernel is not None and self.bias is not None:
            if vecs.shape[1] == self.kernel.shape[0]:
                return (vecs + self.bias).dot(self.kernel)
            else:
                raise Exception("original dim dose not match bert whiten model kernel")
        else:
            raise Exception("bert writen model must fit first when transform vectors")

    def fit_transform(self, vecs):
        if self.kernel is None and self.bias is None:
            self.fit(vecs)
        return self.transform(vecs)

    def save_bw_model(self, model_save_path):
        """用pandas的pickle接口保存
        """
        df = pd.DataFrame({'kernel': [self.kernel],
                           'bias': [self.bias],
                           'n_components': self.n_components,
                           'origin_dim': self.origin_dim})
        df.to_pickle(model_save_path)

    def load_bw_model(self, bw_model_path):
        df = pd.read_pickle(bw_model_path)
        self.kernel = df.kernel[0]
        self.bias = df.bias[0]
        self.origin_dim = df.origin_dim
        if self.n_components != df.n_components[0]:
            raise Exception("Vecs Whitening Model Load Error, n_components dose not match")
