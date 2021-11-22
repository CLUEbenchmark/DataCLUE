#! /bin/bash
bash -x ./run_multi_classify_bert_multi_seed.sh
python ./dataclue_change_label.py > result.json
