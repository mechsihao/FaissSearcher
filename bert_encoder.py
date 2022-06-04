#! -*- coding: utf-8 -*-
import os
import time
from typing import List

os.environ['TF_KERAS'] = '1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

import logging
from tensorflow.python.keras import Input, Model, layers
from bert4keras.models import build_transformer_model
from bert4keras.tokenizers import Tokenizer
from bert4keras.snippets import sequence_padding
from bert4keras.snippets import DataGenerator
from base_encoder import BaseEncoder
from vecs_whitening import VecsWhitening


logger = logging.getLogger("ENCODER")


class EncodeDataGenerator(DataGenerator):
    """
    数据生成器，将文本数据 灌入的时候 构造为如下格式： 
    原文本：
    ['我是小鲨鱼', '小鲨鱼爱吃小兔子']
    构造为:
    [
        [
          [187, 3445, 6786, 23567, 85858, 0, 0, 0], 
          [6786, 23567, 85858, 1123, 75763, 8574, 10293, 243]
        ],
        [
          [1, 1, 1, 1, 1, 0, 0, 0], 
          [1, 1, 1, 1, 1, 1, 1, 1]
        ]
    ]
    """
    def __init__(self, data, dict_path, maxlen=32, batch_size=32, buffer_size=None):
        super(EncodeDataGenerator, self).__init__(data, batch_size=batch_size, buffer_size=buffer_size)
        self.maxlen = maxlen
        self.dict_path = dict_path

    def __iter__(self, random=False):
        tokenizer = Tokenizer(self.dict_path, do_lower_case=True)
        batch_token_ids, batch_segment_ids = [], []
        for is_end, text in self.sample(random):
            token_id, segment_id = tokenizer.encode(str(text), maxlen=self.maxlen)
            batch_token_ids.append(token_id)
            batch_segment_ids.append(segment_id)
            if len(batch_token_ids) == self.batch_size or is_end:
                batch_token_ids = sequence_padding(batch_token_ids)
                batch_segment_ids = sequence_padding(batch_segment_ids)
                yield [batch_token_ids, batch_segment_ids]
                batch_token_ids, batch_segment_ids = [], []

    def forpred(self):
        while True:
            for d in self.__iter__():
                yield d


class BertEncoder(BaseEncoder):
    """
    利用bert encode文本数据
    """

    def __init__(self, 
                 config_path: str, 
                 checkpoint_path: str, 
                 dict_path: str, 
                 maxlen: int = 16, 
                 batch_size: int = 32,
                 is_whitening: bool = False,
                 **kwargs):
        self.config_path = config_path
        self.checkpoint_path = checkpoint_path
        self.dict_path = dict_path
        self.maxlen = maxlen
        self.batch_size = batch_size
        self.is_whitening = is_whitening  # 是否需要向量whitening，whitening技术详情可见：https://spaces.ac.cn/archives/8069
        self.whitening_dim = kwargs['whitening_dim'] if 'whitening_dim' in kwargs else 768  # 不指定whitening_dim则默认不降维
        self.whitening_model = self.init_whitening_model(kwargs)
        self.encoder = self.__init_encoder__()

    def __init_encoder__(self):
        token_input = Input(shape=(None,))
        segment_input = Input(shape=(None,))
        bert = build_transformer_model(self.config_path, self.checkpoint_path, maxlen=self.maxlen)
        x = bert([token_input, segment_input])
        vecs = layers.Lambda(lambda tensor: tensor[:, 0])(x)
        encoder = Model(inputs=[token_input, segment_input], outputs=vecs)
        return encoder

    def init_whitening_model(self, kwargs):
        if self.is_whitening:
            whitening_model = VecsWhitening(n_components=self.whitening_dim)
            if 'whitening_path' in kwargs:
                whitening_model_path = kwargs['whitening_path']
            else:
                raise IOError("Arg 'whitening_path' not found, whitening_path must given when is_whitening is True")
            whitening_model.load_bw_model(whitening_model_path)
            return whitening_model
        else:
            return None

    def whitening_vecs(self, vecs):
        return self.whitening_model.transform(vecs)

    def encode(self, text_list: List[str], verbose: int = 1):
        start_time = time.time()
        data_gen = EncodeDataGenerator(data=text_list, dict_path=self.dict_path, batch_size=self.batch_size, maxlen=self.maxlen)
        vecs = self.encoder.predict(data_gen.forpred(), steps=len(data_gen), verbose=verbose)
        logger.info(f"Encode item cost time: {time.time() - start_time}")
        if self.is_whitening:
            start_time = time.time()
            vecs = self.whitening_vecs(vecs)
            logger.info(f"Whitening vecs cost time: {time.time() - start_time}")
        else:
            pass
        return vecs

    @property
    def info(self):
        info = f'Max_len is [{self.maxlen}]\n' \
               f'Output_dim is [{self.whitening_dim}]\n' \
               f'Whitening vecs [{self.is_whitening}]\n'
        print(info)
