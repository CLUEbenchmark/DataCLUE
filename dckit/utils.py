import os
import json
import numpy as np
from sklearn.model_selection import train_test_split

path = os.path.split(os.path.split(os.path.realpath(__file__))[0])[0]


def read_datasets(dataset='cic'):
    """
    根据输入的数据名称读取数据
    参数：
        dataset： 数据集名称，当前支持CIC和TNEWS
    输出：
        full_data： 字典形式存储的数据，包括：
                    - 'json': json数据的每一行，如 {"id": 13, "label": "79", "sentence": "一斤大概有多少个", "label_des": "买家咨询商品规格数量"}
                    这里为了统一输入输出没有区分train和dev了
                    - 'info': 标签号好描述的对应关系，如{79:'买家咨询商品规格数量'}
    """
    dataset = dataset.lower()
    if dataset in ['cic', 'tnews']:
        json_data = []
        for data_type in ['train', 'dev']:
            for line in open('{}/datasets/raw_{}/{}.json'.format(path, dataset, data_type), 'r', encoding='utf-8'):
                # line = {"id": 13, "label": "79", "sentence": "一斤大概有多少个", "label_des": "买家咨询商品规格数量"}
                one = json.loads(line)
                json_data.append(one)

        label_info = {}
        if dataset == 'cic':
            with open('{}/datasets/raw_{}/labels.txt'.format(path, dataset), 'r', encoding='utf-8') as fa:
                for idx, line in enumerate(fa.readlines()):
                    label_info[idx] = line
        full_data = {'json': json_data, 'info': label_info}
        return full_data
    else:
        raise NotImplementedError


def random_split_data(data, test_size=2000, seed=0):
    json_data = data['json']
    labels = []
    for line in json_data:
        labels.append(int(line['label']))
    train_idx, test_idx, _, _ = train_test_split(range(len(labels)), labels, stratify=labels,
                                                 shuffle=True, test_size=test_size, random_state=seed)

    f = open('{}/datasets/cic/train.json'.format(path), 'w', encoding='utf-8')
    for idx in train_idx:
        dic = json_data[idx]
        str_sen = json.dumps(dic, ensure_ascii=False)
        f.write(str_sen + '\n')

    f = open('{}/datasets/cic/dev.json'.format(path), 'w', encoding='utf-8')
    for idx in test_idx:
        dic = json_data[idx]
        str_sen = json.dumps(dic, ensure_ascii=False)
        f.write(str_sen + '\n')
