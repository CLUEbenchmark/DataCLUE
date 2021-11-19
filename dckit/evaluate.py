import os
from sklearn.metrics import f1_score
import json
import numpy as np

path = os.path.split(os.path.split(os.path.realpath(__file__))[0])[0]


def calc_f1(dataset='cic'):
    y_true = []
    for line in open('{}/datasets/raw_{}/test_public.json'.format(path, dataset.lower()), 'r', encoding='utf-8'):
        y_true.append(json.loads(line)['label'])
    y_pred = []
    for line in open('{}/output_dir/bert/test_prediction.json'.format(path), 'r', encoding='utf-8'):
        y_pred.append(json.loads(line)['label'])

    f1_macro = f1_score(y_true, y_pred, average='macro')
    return f1_macro


def evaluate(dataset='cic'):
    cmds = [
        'rm -rf {}/output_dir/bert'.format(path),
        'rm -f {}/datasets/{}/cached*'.format(path, dataset),
        'cd {}/baselines/models_pytorch/classifier_pytorch'.format(path),
        'bash run_classifier_{}.sh'.format(dataset),
        'bash run_classifier_{}.sh predict'.format(dataset),
    ]
    os.system('&&'.join(cmds))
    return calc_f1(dataset.lower())
