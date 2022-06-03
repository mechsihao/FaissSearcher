# FaissSearcher
A common faiss searcher based on pandas DataFrame

![](https://img.shields.io/badge/知乎-MECH-blue)
![](https://img.shields.io/static/v1?label=tensorflow&message=2.3.2&color=orange)
![](https://img.shields.io/static/v1?label=faiss-cpu&message=1.7.x&color=maroon)
![](https://img.shields.io/static/v1?label=pandas&message=1.3.5&color=yellow)
![](https://img.shields.io/static/v1?label=bert4keras&message=1.3.5&color=silver)


基于pandas DataFrame 检索的Faiss封装，pandas尽量使用1.3.5，如不能保证精确版本，也要保证在1.0以上。
## 特点
简单易上手

## 需要准备
  - 1.encoder，一般来说是自己定义的，里面必须有`encode`方法，代表将文本或者图片encode成向量，用来检索，建议直接继承BaseEncoder，如果原始数据本来就是向量，而不是图片、文本这种需要encode的数据集，也很好办，将`encode`方法中输入向量原封不动的输出即可。可以参考自带的bert_encoder，用起来很方便，事先需要下载tf版本的bert预训练权重。
  - 2.vecs_whitening，一种处理向量空间坍缩的有效方法，非必须，如果需要，可见本项目vecs_whitening.py代码，用法和sklearn的pca一致。可以将训练好的vecs_whitening模型地址输入bert_encoder中，也可以自己用本代码训练模型保存，再传入bert_encoder中。
  - 3.items。必须是pandas DataFrame格式，要求只需要第一列为目标item列，其余列随意，检索时会自动带入到结果中。
  - 4.index_param，faiss的构建参数，代表构建什么类型的索引，这个需要你对Faiss的传参模式了解下，可以看下我写的这片文章的第3节：https://zhuanlan.zhihu.com/p/357414033
  - 5.measurement，度量方法，最常用的是cos余弦相似度，l2欧氏距离，还支持1范数、无穷范数、p范数等等
  - 6.is_nrom，是否需要对象量归一化，根据自己的度量方法或者工程场景来使用，cos默认为True。

## 示例
```python
encoder = BertEncoder()
index_param = 'HNSW64'
measurement = 'cos'
searcher = FaissSearcher(encoder, items, index_param, measurement)
# 构建index
searcher.train()
# 保存index，方便下次调用
searcher.save_index('demo.index')
# 搜索，以文本为例
target = ['你好我叫小鲨鱼', '你好我是小兔子', '很高兴认识你']
df_res = searcher.search(target， topK=10)  # df_res即为结果
```
