from tqdm import tqdm
from dckit import read_datasets, random_split_data, evaluate
import random
import numpy as np


def data_mix(data, num_mix=3):
    json_data = data['json']
    new_json = json_data
    # 按类聚合数据
    sentence_by_class = {}
    label_desc_map = {}
    for idx, tmp in enumerate(tqdm(json_data)):
        if tmp['label'] not in label_desc_map:
            label_desc_map[tmp['label']] = tmp['label_des']
        if tmp['label'] not in sentence_by_class:
            sentence_by_class[tmp['label']] = []
        sentence_by_class[tmp['label']].append(tmp['sentence'])
    idx = 0
    for classes, sentences in tqdm(sentence_by_class.items()):
        for _ in range(len(json_data)//len(data['info'])):
            random.shuffle(sentences)
            sentence = '。'.join(sentences[:num_mix])
            dic = {'id': idx, 'sentence': sentence, 'label': classes, 'label_des': label_desc_map[classes]}
            idx += 1
            new_json.append(dic)
    data['json'] = new_json
    return data


def main():
    res = []
    for i in range(5):
        data = read_datasets()
        data = data_mix(data)
        random_split_data(data, seed=i)
        f1 = evaluate()
        res.append(f1)
    print('Macro-F1=', np.mean(res), np.std(res))
    return f1


if __name__ == '__main__':
    main()
