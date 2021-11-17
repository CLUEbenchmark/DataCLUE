import json
import pandas as pd

from sklearn.metrics import f1_score

def compute_f1_score_by_list(y_true_list,y_pred_list):
    #y_true = [1, 1, 1, 1, 2, 2, 2, 3, 3]
    # y_pred = [1, 1, 2, 3, 2, 2, 3, 2, 3]
    f1_micro = f1_score(y_true_list, y_pred_list, average='micro')
    f1_macro = f1_score(y_true_list, y_pred_list, average='macro')
    print('f1_micro: {0}'.format(f1_micro))
    print('f1_macro: {0}'.format(f1_macro))

def compute_score_fn(target_file, predict_file):
    predict_object=open(predict_file,'r')
    predict_lines=predict_object.readlines()

    target_object=open(target_file,'r')
    target_lines=target_object.readlines()
    countt=0
    total_ignore=0
    y_pred_list=[]
    y_true_list=[]
    for i, source_line in enumerate(predict_lines):
        source_line_json=json.loads(source_line)
        predict_label=source_line_json['label']
        y_pred_list.append(predict_label)
        target_line_json=json.loads(target_lines[i])
        target_label=target_line_json['label']
        y_true_list.append(target_label)
        if str(target_label)=='-1':
            total_ignore=total_ignore+1
            continue
        if predict_label==target_label:
            countt=countt+1

    compute_f1_score_by_list(y_true_list, y_pred_list)
    avg=float(countt)/float(len(target_lines)-total_ignore)
    print("avg:",avg,";total_ignore:",total_ignore,";target_lines:",len(target_lines))


target_file='test_public.json'
predict_file='test_public_preidct.json'
compute_score_fn(target_file, predict_file)
