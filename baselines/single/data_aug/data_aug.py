from tqdm import tqdm
from textda.data_expansion import data_expansion
from dckit import read_datasets, random_split_data, evaluate


def data_aug(data, num_aug=3):
    json_data = data['json']
    new_json = []
    for idx, tmp in enumerate(tqdm(json_data)):
        # 扩展相似句：从一个句子变成多个相似的句子的列表
        sen_list = data_expansion(
            tmp['sentence'], alpha_ri=0.1, alpha_rs=0, num_aug=num_aug)
        for sen in sen_list:
            dic = tmp
            dic['sentence'] = sen
            new_json.append(dic)
    data['json'] = new_json
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
