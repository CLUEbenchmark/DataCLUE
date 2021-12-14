from tqdm import tqdm
from textda.data_expansion import data_expansion
import os
import sys
path = os.path.split(os.getcwd())
sys.path.append(path)

from dckit import read_datasets, random_split_data, evaluate
from baselines.single.data_aug.data_aug import data_aug
from baselines.single.def_aug.def_aug import def_aug
from baselines.single.delete_noise.delete_noise import delete_noise


def simple_baseline(data, use_delete=False, use_aug=False, use_def=False):
    if use_delete:
        data = delete_noise(data)
    if use_aug:
        data = data_aug(data)
    if use_def:
        data = def_aug(data)
    return data


def main():
    data = read_datasets()
    data = simple_baseline(data)
    random_split_data(data)
    f1 = evaluate()
    print('Macro-F1=', f1)
    return f1


if __name__ == '__main__':
    main()