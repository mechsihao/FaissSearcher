"""
创建人：MECH
创建时间：2022/02/01
功能描述：encoder类，用bert编码文本，融合了whitening方法，可以有效的提高bert的检索能力
    支持使用bert_service，利用远端gpu能力encode bert
功能更新：支持自动whitening，支持指定embedding输出位置和层
"""

import os
import time
from typing import List

os.environ['TF_KERAS'] = '1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

import pandas as pd
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers, models
from bert4keras.models import build_transformer_model
from base_encoder import BaseEncoder
from encoder_utils import timeout, EncodeDataGenerator, InteractDataGenerator, merge, print_args_info
from vecs_whitening import VecsWhitening
from bert_serving.client import BertClient
from concurrent import futures

executor = futures.ThreadPoolExecutor(1)


class BertEncoder(BaseEncoder):
    """
    利用bert encode文本数据
    """
    def __init__(
            self,
            config_path: str = '',
            checkpoint_path: str = '',
            dict_path: str = '',
            max_len: int = 16,
            batch_size: int = 32,
            model_name: str = 'base',
            use_remote_service_first: bool = False,
            **kwargs
    ):
        self.kwargs = kwargs
        self.params = self.__init_support_params()
        self.__check_kwargs()
        self.config_path = config_path
        self.checkpoint_path = checkpoint_path
        self.dict_path = dict_path
        self.max_len = max_len
        self.model_name = model_name
        self.batch_size = batch_size
        self.is_whitening = kwargs['is_whitening'] if 'is_whitening' in kwargs else False
        self.whitening_dim = kwargs['whitening_dim'] if 'whitening_dim' in kwargs else 768
        self.whitening_vec_nums = kwargs['whitening_vec_nums'] if 'whitening_vec_nums' in kwargs else None
        self.whitening_model = self.__init_whitening_model(**kwargs)
        self.__init_encoder(kwargs)
        self.bert_client = self.__init_bert_service(kwargs) if use_remote_service_first else None

    @staticmethod
    def __init_support_params():
        return {
            "is_whitening": "[bool] Whitening is an efficient way to deal with vector collapse.",
            "whitening_dim": "[int] Dimension after whitening",
            "whitening_path": "[str] Whitening Model Path.",
            "whitening_vec_nums": "[int] Whitening model fit dataset nums",
            "encoder": "[tf-model] A tf model which can encode text by 'predict' method.",
            "pool_pos": "[int|str] Output embedding position of bert model, it also can be 'avg' or 'max'",
            "out_layer": "[int] Output embedding layer of bert model",
            "service_config": "[dict] Bert service config, like {'ip': '192.168.22.101', 'port': '1314', 'port_out': '1315'}",
            "model_weights_path": "[str] You can also input model save path and given model name to load encoder"
        }

    @timeout(5)
    def __connect_to_bert_service(self):
        """连接bert_service，如果5s连接不上bert_service就抛出TimeOutError
        """
        return BertClient(ip=self.ip, port=self.port, port_out=self.port_out)

    def __init_bert_service(self, kwargs):
        try:
            service_config = kwargs['service_config']

            self.ip = service_config['ip']
            self.port = int(service_config['port']) if 'port' in service_config else 5555
            self.port_out = int(service_config['port_out']) if 'port_out' in service_config else 5556

            print(f"Connecting bert service...\nip={self.ip}, port={self.port}, port_out={self.port_out}")
            bert_client = self.__connect_to_bert_service()
        except Exception as e:
            if str(e) == "":
                print(f"Connect bert service time out...")
            else:
                print(f"Failed to connect bert service: {str(e)}")
            bert_client = None
        return bert_client

    @timeout(0.02)
    def __test_service_encode(self, text: str):
        # 拿一条数据测试，设置超时时间20ms，如果超时则不使用bert_service
        self.bert_client.encode([text])

    def test_service_encode(self, text: str):
        try:
            self.__test_service_encode(text)
            return True
        except Exception as e:
            print(f"Bert service encode failed, use local devices. details: {e}")
            return False

    def __init_encoder(self, kwargs):
        if "encoder" in kwargs:
            self.encoder, self.model = kwargs["encoder"], None
            if self.dict_path == "":
                raise Exception("外部输入Encoder，必须输入对应的vocab dict文件")
        else:
            self.encoder, self.model = load_encoder(self.model_name, self.config_path, self.checkpoint_path, **kwargs)

    def __init_whitening_model(self, **kwargs):
        if self.is_whitening:
            whitening_model = VecsWhitening(n_components=self.whitening_dim)
            if 'whitening_path' in kwargs:
                whitening_model_path = kwargs['whitening_path']
                whitening_model.load_bw_model(whitening_model_path)
            elif 'whitening_source_vecs' in kwargs:
                start_time = time.time()
                whitening_model.fit(kwargs['whitening_source_vecs'])
                print(f"Whitening model build cost time: {time.time() - start_time}")
            else:
                whitening_model = None

            return whitening_model
        else:
            return None

    def whitening_vecs(self, vecs: np.array):
        return self.whitening_model.transform(vecs)

    def encode(self, text_list: List[str], verbose: int = 1):
        if self.encoder is None:
            raise ValueError(f"{self.model_name}模式下无encoder，请使用predict方法")
        else:
            start_time = time.time()
            if self.bert_client and self.test_service_encode(text_list[0]):
                vecs = self.bert_client.encode(text_list)
                print(f"Remote Bert Service Encode, Cost {time.time() - start_time}s")
            else:
                data_gen = EncodeDataGenerator(data=text_list, dict_path=self.dict_path, batch_size=self.batch_size, max_len=self.max_len)
                vecs = self.encoder.predict(data_gen.forpred(), steps=len(data_gen), verbose=verbose)
                print(f"Local Devices Encode, Cost {time.time() - start_time}s")

            if self.is_whitening:
                if not self.whitening_model:
                    first_batch_size, org_vecs_dim = vecs.shape[0], vecs.shape[1]
                    if first_batch_size > org_vecs_dim:
                        if self.whitening_vec_nums and first_batch_size > self.whitening_vec_nums:
                            shuffle_idx = np.random.permutation(np.arange(self.whitening_vec_nums))
                        else:
                            if self.whitening_vec_nums:
                                print(f"[Warn] First Batch Size Less Than {self.whitening_vec_nums}")
                            else:
                                print(f"[Warn] Whitening Vecs Fit Num Dose not Set, default {first_batch_size}")
                            shuffle_idx = np.random.permutation(np.arange(first_batch_size))
                        self.whitening_model = self.__init_whitening_model(whitening_source_vecs=vecs[shuffle_idx])
                    else:
                        raise Exception(f"Set Whitening Enable But Not Given Whitening Model Path, Encoder Will Fit A Whitening "
                                        f"Model in First Batch, It Need First Batch Size({vecs.shape[0]}) At Least Larger Than Vecs Dim"
                                        f"({vecs.shape[1]}).")
                else:
                    pass

                start_time = time.time()
                vecs = self.whitening_vecs(vecs)
                print(f"Whitening vecs cost time: {time.time() - start_time}")
            else:
                pass

            return vecs

    def predict(self, text_list: List[str], verbose: int = 1):
        if self.model_name in ('base', 'sbert', 'cosent'):
            raise ValueError(f"Not Support Model Name = {self.model_name}")
        else:
            data_gen = InteractDataGenerator(data=text_list, dict_path=self.dict_path, batch_size=self.batch_size, max_len=self.max_len * 2)
            return self.model.predict(data_gen.forpred(), steps=len(data_gen), verbose=verbose)

    def close_service(self):
        if self.bert_client:
            self.bert_client.close()
        else:
            print("Bert service dose not config...")

    def connect_service(self):
        if self.bert_client:
            print("Bert service dose not config...")
        else:
            self.__init_bert_service(self.kwargs)

    def save_whitening_model(self, save_path: str):
        if self.whitening_model:
            self.whitening_model.save_bw_model(save_path)
        else:
            raise NotImplementedError("Whitening Model Is Not Defined Or Not Trained")

    def __check_kwargs(self):
        all_support_kwargs = self.params
        for arg in self.kwargs:
            if arg not in all_support_kwargs:
                print(f"[Warn] param: {arg} = {self.kwargs[arg]} is not registered in param_dict")

    def config_info(self):
        info = f'Max len: [{self.max_len}]\n' \
               f'Output dim: [{self.whitening_dim}]\n' \
               f'Bert service: [{f"ip={self.ip}, port={self.port}, port_out={self.port_out}" if self.bert_client else None}]\n' \
               f'Whitening: [{f"True, whitening_vec_nums={self.whitening_vec_nums}" if self.is_whitening else "False"}]\n'
        print(info)

    def kw_params_info(self):
        print_args_info(self.params)


def load_encoder(model_name, config_path, checkpoint_path, **kwargs):
    if "pool_pos" in kwargs:
        pool_pos = kwargs["pool_pos"]
        if isinstance(pool_pos, str) and pool_pos == 'avg':
            raise ValueError(f"pool_pos not support: {pool_pos}")
        if isinstance(pool_pos, int) and (pool_pos >= 512 or pool_pos < 0):
            raise ValueError(f"pool_pos scalar must in [0, 512), get {pool_pos}")
    else:
        pool_pos = 0

    base = build_transformer_model(config_path, checkpoint_path)

    if "out_layer" in kwargs:
        out_layer = kwargs["out_layer"]
        bert_df = pd.DataFrame(base.get_config()["layers"])
        out_layer_name_list = bert_df[bert_df.name.apply(lambda x: x.endswith("FeedForward-Norm"))].name.tolist()
        if isinstance(out_layer, int) and -len(out_layer_name_list) < out_layer < len(out_layer_name_list):
            out_layer_name = out_layer_name_list[out_layer]
        else:
            raise ValueError(f"out_layer must be a int less than {len(out_layer_name_list)}")
        out_layer = base.get_layer(out_layer_name)
    else:
        out_layer = base

    if pool_pos == "avg":
        output = keras.layers.GlobalAveragePooling1D()(out_layer.output)
    elif pool_pos == "max":
        output = keras.layers.GlobalMaxPooling1D()(out_layer.output)
    else:
        output = layers.Lambda(lambda tensor: tensor[:, pool_pos], name='bert_encoder')(out_layer.output)

    if model_name.lower() == 'base':
        encoder = models.Model(base.inputs, output)
        dense = None
        if 'model_weights_path' in kwargs:
            model_weights_path = kwargs['model_weights_path']
            encoder.load_weights(model_weights_path)

    else:
        if 'model_weights_path' in kwargs:
            model_weights_path = kwargs['model_weights_path']
        else:
            raise Exception("若要加载interact、sbert模型需要传参: model_weights_path")

        if model_name.lower() == 'cosent':
            encoder = models.Model(base.inputs, output)
            encoder.load_weights(model_weights_path)
            dense = None

        elif model_name.lower() == 'sbert':
            encoder = models.Model(base.inputs, output)
            output = layers.Lambda(merge)(output)
            output = layers.Dense(units=2, activation='softmax')(output)
            model = models.Model(base.inputs, output)
            model.load_weights(model_weights_path)
            dense = None

        else:
            raise Exception(f"不支持model_name='{model_name}'")

    return encoder, dense
