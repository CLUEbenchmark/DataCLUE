import os
os.environ["CUDA_VISIBLE_DEVICE"] = '1'
from transformers import Trainer, TrainingArguments, AutoModelForSequenceClassification, AutoTokenizer
import torch
import numpy as np
from datasets import load_metric
from sklearn.model_selection import StratifiedKFold  # StratifiedKFold划分数据集的原理：划分后的训练集和验证集中类别分布尽量和原数据集一样

path = os.path.split(os.path.split(os.path.realpath(__file__))[0])[0]
from baselines.single.data_aug.parallel_textda import parallel_expansion
os.environ["TOKENIZERS_PARALLELISM"] = 'false'
PRETRAIN = 'hfl/rbtl3'  # 加载的预训练模型的名称
metric = load_metric("f1")  # 使用f1 score作为指标


# 计算标签与预测值在给定的指标上的效果
def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    return metric.compute(predictions=predictions, references=labels, average='macro')


class MyDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx])
                for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)


def get_prediction(data):
    """
    训练一个模型，得到数据点上的标签预测：
    1） 加载数据；
    2）使用K折交叉验证训练，并在验证集上做预测；
    3) 合并交叉验证的结果，并得到整个数据集上模型的预测的概率分布
    train a model to get estimation of each data point
    """
    # 1、加载所有数据、标签到列表 all_text, all_label,all_id
    all_text, all_label, all_id = [], [], []
    for idx, line in enumerate(data['json']):
        all_text.append(line['sentence'])
        all_label.append(int(line['label']))
        all_id.append(idx)
    # 加载标签定义增强后的数据
    # label_data.json--->{"id": -1, "sentence": "买家抱怨商品了", "label_des": "买家抱怨商品涨价了\n", "label": 0}
    label_text, label_label = [], []
    # for line in open('../../datasets/cic/label_data.json', 'r', encoding='utf-8'):
    #     label_text.append(json.loads(line)['sentence'])
    #     label_label.append(int(json.loads(line)['label']))

    # 2、使用K折交叉验证训练，并在验证集上做预测：遍历每一折得到训练集和验证子集、数据增强、设置训练参数和数据进行训练、在验证集上进行预测
    dev_out = {}  # 带索引(index)的验证子集的列表
    dev_index = {}  # 带索引(index)的验证集的列表
    kf = StratifiedKFold(n_splits=6, shuffle=True)  # StratifiedKFold划分数据集的原理：划分后的训练集和验证集中类别分布尽量和原数据集一样
    # kf.get_n_splits(all_text, all_label)
    for kf_id, (train_index, test_index) in enumerate(kf.split(all_text, all_label)):
        # 2.1 得到训练和验证子集
        # kf_id:第几折；train_index, test_index这一折的训练、验证集。
        train_text = [all_text[i] for i in train_index][:] + label_text  # 训练集的文本
        train_label = [all_label[i] for i in train_index][:] + label_label  # 训练集的标签
        dev_text = [all_text[i] for i in test_index]
        dev_label = [all_label[i] for i in test_index]
        dev_index[kf_id] = test_index

        # 2.2 对训练数据进行数据扩增
        # new_train_text = []
        # new_train_label = []
        # for idx, tmp_text in enumerate(train_text):
        #     sen_list = data_expansion(tmp_text, alpha_ri=0.1, alpha_rs=0, num_aug=5)
        #     new_train_text.extend(sen_list)
        #     new_train_label.extend([train_label[idx]] * len(sen_list))
        #
        # train_text = new_train_text
        # train_label = new_train_label

        sen_list, label_list = parallel_expansion(train_text, train_label, alpha_ri=0.1, alpha_rs=0, num_aug=5)
        train_text = sen_list
        train_label = label_list
        assert len(train_text) == len(train_label)
        # 2.3 设置使用的预训练模型，并设置tokenizer、数据集对象
        tokenizer = AutoTokenizer.from_pretrained(PRETRAIN, do_lower_case=True)
        train_encodings = tokenizer(train_text, truncation=True, padding=True, max_length=32)
        val_encodings = tokenizer(dev_text, truncation=True, padding=True, max_length=32)

        train_dataset = MyDataset(train_encodings, train_label)
        val_dataset = MyDataset(val_encodings, dev_label)

        # 2.4 实例化训练参数
        training_args = TrainingArguments(
            # output directory
            output_dir='../../tmpresults/tmpresult{}'.format(kf_id),
            num_train_epochs=50,  # total number of training epochs
            per_device_train_batch_size=256,  # batch size per device during training
            per_device_eval_batch_size=32,  # batch size for evaluation
            warmup_steps=500,  # number of warmup steps for learning rate scheduler
            learning_rate=3e-4 if 'electra' in PRETRAIN else 2e-5,
            weight_decay=0.01,  # strength of weight decay
            save_total_limit=1,
            # logging_dir='../../tmplogs',  # directory for storing logs
            # logging_steps=10,
            # evaluation_strategy="epoch",
        )
        model = AutoModelForSequenceClassification.from_pretrained(PRETRAIN, num_labels=len(data['info']))

        # 2.5 利用实例化的训练对象进行训练（模型、训练参数、训练集、验证集、评价指标）
        trainer = Trainer(
            # the instantiated 🤗 Transformers model to be trained
            model=model,
            args=training_args,  # training arguments, defined above
            train_dataset=train_dataset,  # training dataset
            eval_dataset=val_dataset,  # evaluation dataset
            compute_metrics=compute_metrics,
        )
        trainer.train()  # 训练模型

        # 2.6 利用训练好的模型在验证集上进行预测
        dev_outputs = trainer.predict(val_dataset).predictions
        dev_out[kf_id] = dev_outputs  # 将预测结果保存在列表中

    # 3、合并交叉验证的结果，并得到整个数据集上模型的预测的概率分布
    alls = [0] * len(all_label)
    for kfid in range(6):
        for idx, item in enumerate(dev_index[kfid]):
            # dev_index[0]:第0折的验证数据的索引的列表
            alls[item - 1] = dev_out[kfid][idx]
    outputs = np.array(alls)
    return outputs


if __name__ == '__main__':
    get_prediction()
