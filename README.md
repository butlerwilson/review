# 酒店评论标签生成

## 朴素贝叶斯实现文本分类

**外部依赖**

1. numpy
2. sklearn
3. jieba
4. snownlp

pip install package 可直接安装

**代码组织**

1. preprocess.py

训练集，测试集，预测集数据准备；
结巴分词，增加自定义字典；
统计单词频度，去掉低频词语；

2. classify.py

加载所有数据，使用朴素贝叶斯算法进行分类

3. tag\_features.py gen\_results.py

根据自己需求做一部分的结果处理

4. 目录

data目录，所有原始数据目录；
middle目录，所有中间数据目录；
result目录，所有结果数据目录；

5. run.sh

运行测试参数和运行，运行先请修改


## 相关技术博客分享

[blog](http://www.milier.me)

author: youngcy
