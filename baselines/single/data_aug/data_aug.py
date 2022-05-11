import imp
from tqdm import tqdm
from textda.data_expansion import data_expansion
import sys
sys.path.append('../../../')
sys.path.append('../../../dckit')
from dckit import read_datasets, random_split_data, evaluate
import swifter
import pandas as pd
import numpy as np


def aug_function(sentence, alpha_ri=0.1, alpha_rs=0, num_aug=3):
    aug_list = data_expansion(sentence, alpha_ri, alpha_rs, p_rd=0.2, num_aug=num_aug)
    if len(aug_list) != num_aug:
        l = len(aug_list)
        if l < num_aug:
            for i in range(num_aug-l):
                aug_list.append(None)
        else:
            aug_list = aug_list[:num_aug]
    return aug_list


def data_aug(data, num_aug=3):
    json_data = data['json']
    df = pd.DataFrame.from_records(json_data)
    df.columns = json_data[0].keys()
    aug_lists = df['sentence'].swifter.apply(aug_function)
    aug_lens = [len(aug_list) for aug_list in aug_lists]
    flatten_list = [j for sub in aug_lists for j in sub]
    newdf = pd.DataFrame(np.repeat(df.values, num_aug, axis=0), columns=df.columns)
    newdf['sentence'] = flatten_list
    # remove none
    newdf.dropna(inplace=True)
    data["json"] = newdf.to_dict(orient='records')
    return data



def main():
    data = read_datasets()
    data = data_aug(data)
    random_split_data(data)
    f1 = evaluate()
    print('Macro-F1=', f1)
    return f1


if __name__ == '__main__':
    main()
