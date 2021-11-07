import json
import numpy as np
from textda.data_expansion import data_expansion
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from scipy.stats import entropy
from classifier import get_prediction


def find_max_entropy(predicted_probabilities):
    entros = entropy(predicted_probabilities, axis=1)
    return np.argsort(entros)[::-1]


# add label definitions as training data
with open('../../datasets/cic/label_data.json', 'w', encoding='utf-8') as f:
    with open('../../datasets/cic/labels.txt', 'r', encoding='utf-8') as fa:
        for idx, line in enumerate(fa.readlines()):
            sen_list = data_expansion(line, alpha_ri=0.2, alpha_rs=0, num_aug=10)
            for sen in sen_list:
                tmp = {}
                tmp['id'] = -1
                tmp['sentence'] = sen
                tmp['label_des'] = line
                tmp['label'] = idx
                str_sen = json.dumps(tmp, ensure_ascii=False)
                f.write(str_sen+'\n')


json_data = []
labels = []
for data_type in ['train', 'dev']:
    for line in open('../../datasets/cic/{}.json'.format(data_type), 'r', encoding='utf-8'):
        one = json.loads(line)
        json_data.append(one)
        labels.append(int(one['label']))

# run any classifiers to get a preidiction
numpy_array_of_predicted_probabilities = get_prediction()
ordered_label_errors = find_max_entropy(numpy_array_of_predicted_probabilities)

dic = {}
f = open('../../datasets/cic/all_expan.json', 'w', encoding='utf-8')
for idx, tmp in enumerate(tqdm(json_data)):
    if idx in ordered_label_errors[:100]:
        continue
    sen_list = data_expansion(tmp['sentence'], alpha_ri=0.1, alpha_rs=0, num_aug=3)
    for sen in sen_list:
        dic = tmp
        dic['sentence'] = sen
        str_sen = json.dumps(dic, ensure_ascii=False)
        f.write(str_sen+'\n')
f.close()

# prepare output
json_data = []
labels = []
for line in open('../../datasets/cic/all_expan.json', 'r', encoding='utf-8'):
    one = json.loads(line)
    json_data.append(one)
    labels.append(int(one['label']))

for line in open('../../datasets/cic/label_data.json', 'r', encoding='utf-8'):
    one = json.loads(line)
    json_data.append(one)
    labels.append(int(one['label']))

train_idx, test_idx, _, _ = train_test_split(
    range(len(labels)), labels, stratify=labels, shuffle=True, test_size=2000)

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
