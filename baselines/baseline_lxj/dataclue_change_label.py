#-*- coding:utf-8 -*-
import os
import sys
import pdb
import json
import collections

label_file=open("../../datasets/cic/labels.txt")
label_lines=label_file.readlines()
id_label_map={index:str(label_lines[index].strip()) for index in range(len(label_lines))}
label_id_map={str(label_lines[index].strip()):index for index in range(len(label_lines))}

all_sentences=[]
all_labels=[]
all_ids=[]
all_label_des=[]

all_source_lines={}
for split_file_index in range(1,7):

    dev_file=open("./baseline_data/dev_{}.json".format(split_file_index),'r',encoding="utf-8")
    dev_lines=[json.loads(line.strip()) for line in dev_file]

    sentences=[line["sentence"] for line in dev_lines]
    labels=[line["label"] for line in dev_lines]
    ids=[line["id"] for line in dev_lines]
    label_des=[line["label_des"] for line in dev_lines]

    all_source_lines.update({ids[index]:dev_lines[index] for index in range(len(dev_lines))})

    all_sentences.extend(sentences)
    all_labels.extend(labels)
    all_ids.extend(ids)
    all_label_des.extend(label_des)

dev_result_map={}
for seed in [8,9,10]:

    dev_result_map[seed]=[]

    for split_file_index in range(1,7):
        dev_result_file=open("./output_dir/dataclue_{}_{}/eval_preds_{}.txt".format(split_file_index,seed,seed),'r',encoding="utf-8")
        dev_results=[str(line.strip()) for line in dev_result_file]
        dev_result_map[seed].extend(dev_results)

    assert len(all_sentences)==len(all_labels)==len(all_ids)==len(dev_result_map[seed])

dev_result_map_prob={}
for seed in [8,9,10]:

    dev_result_map_prob[seed]=[]

    for split_file_index in range(1,7):
        dev_result_file=open("./output_dir/dataclue_{}_{}/eval_probility_{}.txt".format(split_file_index,seed,seed),'r',encoding="utf-8")
        dev_results=[str(line.strip()) for line in dev_result_file]
        dev_result_map_prob[seed].extend(dev_results)

    assert len(all_sentences)==len(all_labels)==len(all_ids)==len(dev_result_map_prob[seed])

result_map={}
average_score_list=[]
for index in range(len(all_sentences)):
    average_score=str((float(dev_result_map_prob[8][index])+float(dev_result_map_prob[9][index])+float(dev_result_map_prob[10][index]))/3)
    average_score_list.append(average_score)
    result_map[average_score]=all_sentences[index]+"\t"+all_label_des[index]+"\t"+all_labels[index]+"\t"+str(all_ids[index])+"\t"+dev_result_map[8][index]+"\t"+dev_result_map[9][index]+"\t"+dev_result_map[10][index]+"\t"+dev_result_map_prob[8][index]+"\t"+dev_result_map_prob[9][index]+"\t"+dev_result_map_prob[10][index]

need_change_sentence_index=[]

count=0
for index in range(len(all_sentences)):
    if float(average_score_list[index])>0.6 and dev_result_map[8][index]==dev_result_map[9][index]==dev_result_map[10][index] and dev_result_map[8][index]!=all_label_des[index]:
        need_change_id=all_ids[index]
        all_source_lines[need_change_id]["label_des"]=dev_result_map[8][index]
        all_source_lines[need_change_id]["label"]=str(label_id_map[dev_result_map[8][index]])
        count+=1

for id_,line in all_source_lines.items():
    print(json.dumps(line, ensure_ascii=False))
