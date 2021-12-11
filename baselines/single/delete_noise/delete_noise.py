from tqdm import tqdm
from dckit import read_datasets, random_split_data, evaluate
from scipy.stats import entropy
import numpy as np
import os
import sys
path = os.path.split(os.path.split(os.path.realpath(__file__))[0])[0]
sys.path.append(path)
from baselines.single.delete_noise.classifier import get_prediction


def find_max_entropy(predicted_probabilities):
    entros = entropy(predicted_probabilities, axis=1)
    return np.argsort(entros)[::-1]


def delete_noise(data, delete_num=100):
    numpy_array_of_predicted_probabilities = get_prediction(data)
    ordered_label_errors = find_max_entropy(numpy_array_of_predicted_probabilities)

    json_data = data['json']
    new_json = []
    for idx, tmp in enumerate(tqdm(json_data)):
        # 每一句都给他扩展
        if idx in ordered_label_errors[:delete_num]:  # and idx not in correct_id:
            # print(tmp['sentence'], tmp['label_des'])
            continue
        new_json.append(tmp)
    data['json'] = new_json
    return data


def main():
    data = read_datasets()
    data = delete_noise(data)
    random_split_data(data)
    f1 = evaluate()
    print('Macro-F1=', f1)
    return f1


if __name__ == '__main__':
    main()
