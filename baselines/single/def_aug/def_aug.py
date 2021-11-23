import json
import os
from tqdm import tqdm
from textda.data_expansion import data_expansion
from dckit import read_datasets, random_split_data, evaluate

path = os.path.split(os.path.split(os.path.realpath(__file__))[0])[0]


def def_aug(data, num_aug=50):
    json_data = data['json']
    label_info = data['info']
    for idx, line in label_info.items():
        if num_aug > 0:
            sen_list = data_expansion(line, alpha_ri=0.2, alpha_rs=0, num_aug=num_aug)
        if num_aug == 0:
            sen_list = [line]
        for sen in sen_list:
            tmp = {}
            tmp['id'] = -1
            tmp['sentence'] = sen
            tmp['label_des'] = line
            tmp['label'] = idx
            json_data.append(tmp)

    data['json'] = json_data
    return data


def main():
    data = read_datasets()
    data = def_aug(data)
    random_split_data(data)
    f1 = evaluate()
    print('Macro-F1=', f1)
    return f1


if __name__ == '__main__':
    main()
