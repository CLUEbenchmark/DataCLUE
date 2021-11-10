# if you meet this issue: ConnectionError: Couldn't reach https://raw.githubusercontent.com/huggingface/datasets/1.14.0/metrics/f1/f1.py
# you can add one line at /etc/hosts
#  185.199.108.133 raw.githubusercontent.com
# check more detail below:
# https://programmerah.com/solved-connection-error-couldnt-reach-http-raw-githubusercontent-com-huggingface-41668/

import json
import numpy as np
from textda.data_expansion import data_expansion
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from scipy.stats import entropy
from classifier import get_prediction

"""
主要思路：
1. Miss label data(y): 通过训练一个分类模型根据预测的熵找出数据中最有可能标签错误的样本，并丢弃；
2. Data augment(X):使用数据增强提升数据量，即对输入文本的增强；
3. Label definiation: 将标签定义增强后添加到训练集中增加数据量。比如类别1的定义是“用户询问快递时间”，对这个文本做增强。 

工作流程：
1） 合并训练集和验证集，获得整个数据集的数据；
2） 通过训练一个分类模型，根据预测的熵找出数据中最有可能标签错误的样本并丢弃；
3） 对剩余数据，进行数据扩增；将标签定义增强过的数据加入到数据集。
4） 输出改进过的训练集、验证集：1）拆分；2）写到文件；
"""

# 根据分类模型预测的值，计算熵，并排序
def find_max_entropy(predicted_probabilities):
    print("predicted_probabilities.shape:",predicted_probabilities.shape) # [None, num_labels]
    entros = entropy(predicted_probabilities, axis=1) # entropy(): Calculate the entropy of a distribution for given probability values，计算模型预测的不确定性的程度
    return np.argsort(entros)[::-1]


# add label definitions as training data
# Label definiation，将标签定义增强后添加到训练集中增加数据量。比如类别1的定义是“用户询问快递时间”，对这个文本做增强。
#   输入: labels.txt（标签定义），输出：label_data.json（标签定义及其相似句子）
#   e.g. {"id": -1, "sentence": "买家抱怨商品了", "label_des": "买家抱怨商品涨价了\n", "label": 0}
with open('../../datasets/cic/label_data.json', 'w', encoding='utf-8') as f:
    with open('../../datasets/cic/labels.txt', 'r', encoding='utf-8') as fa:
        for idx, line in enumerate(fa.readlines()):
            sen_list = data_expansion(line, alpha_ri=0.2, alpha_rs=0, num_aug=10) # 扩展相似句：从一个句子变成多个相似的句子的列表
            for sen in sen_list:
                tmp = {}
                tmp['id'] = -1
                tmp['sentence'] = sen
                tmp['label_des'] = line
                tmp['label'] = idx
                str_sen = json.dumps(tmp, ensure_ascii=False)
                f.write(str_sen+'\n')


# 1、合并训练集和验证集，
json_data = [] # 包含所有的数据的列表
labels = [] # 包含所有的数据对应的标签的列表
for data_type in ['train', 'dev']:
    for line in open('../../datasets/cic/{}.json'.format(data_type), 'r', encoding='utf-8'):
        one = json.loads(line) # line = {"id": 13, "label": "79", "sentence": "一斤大概有多少个", "label_des": "买家咨询商品规格数量"}
        json_data.append(one)
        labels.append(int(one['label']))

# run any classifiers to get a preidiction
# 2、通过训练一个分类模型，根据预测的熵找出数据中最有可能标签错误的样本，并丢弃；
numpy_array_of_predicted_probabilities = get_prediction() # TODO CHECK
ordered_label_errors = find_max_entropy(numpy_array_of_predicted_probabilities) # TODO CHECK

# 3、对合并的数据，丢弃最可以发送标签错误的数据后，进行数据扩增。
dic = {}
f = open('../../datasets/cic/all_expan.json', 'w', encoding='utf-8')  # all_expan.json用于存储扩展后的数据
for idx, tmp in enumerate(tqdm(json_data)):
    if idx in ordered_label_errors[:100]: # 丢弃最有可能标签错误的100个数据。TODO。为了获得更好效果，这里的数字100应该改成动态的，当数据点的熵超过一定阈值后就可以丢弃。
        continue
    sen_list = data_expansion(tmp['sentence'], alpha_ri=0.1, alpha_rs=0, num_aug=3) # 扩展相似句：从一个句子变成多个相似的句子的列表
    for sen in sen_list:
        dic = tmp
        dic['sentence'] = sen
        str_sen = json.dumps(dic, ensure_ascii=False)
        f.write(str_sen+'\n')
f.close()

# 4、输出改进过的训练集、验证集：1）拆分；2）写到文件。  prepare output
json_data = [] # 存储所有待输出的数据的列表
labels = [] # 存储所有待输出的数据的标签
for line in open('../../datasets/cic/all_expan.json', 'r', encoding='utf-8'):
    one = json.loads(line)
    json_data.append(one)
    labels.append(int(one['label']))

for line in open('../../datasets/cic/label_data.json', 'r', encoding='utf-8'):
    one = json.loads(line)
    json_data.append(one)
    labels.append(int(one['label']))

train_idx, test_idx, _, _ = train_test_split(
    range(len(labels)), labels, stratify=labels, shuffle=True, test_size=2000) # stratify：表示是否按照样本比例（不同类别的比例）来划分数据集

f = open('../../datasets/cic/train.json', 'w', encoding='utf-8')
for idx in train_idx:
    dic = json_data[idx]
    str_sen = json.dumps(dic, ensure_ascii=False)
    f.write(str_sen+'\n')

f = open('../../datasets/cic/dev.json', 'w', encoding='utf-8')
for idx in test_idx:
    dic = json_data[idx]
    str_sen = json.dumps(dic, ensure_ascii=False)
    f.write(str_sen+'\n')
