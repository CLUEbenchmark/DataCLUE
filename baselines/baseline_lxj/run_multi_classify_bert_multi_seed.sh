#! /bin/bash
model_type=bert
epoch=6
ttime=`date +"%Y-%m-%d-%H-%M"`
echo $ttime

for seed in $(seq 8 10);do
    for i in $(seq 1 6);do
        CUDA_VISIBLE_DEVICES=0 python ./run_multi_classify.py \
            --model_name_or_path=bert-base-chinese \
            --output_dir=./output_dir/dataclue_$i\_$seed \
            --model_type=$model_type \
            --train_file=./baseline_data/train_$i.json \
            --validation_file=./baseline_data/dev_$i.json \
            --test_file=../../datasets/cic/test_public.json \
            --task_name=dataclue \
            --per_device_train_batch_size=16 \
            --num_train_epochs=$epoch \
            --max_seq_length=64 \
            --learning_rate=2e-5 \
            --seed=$seed \
            --overwrite_output_dir \
            --overwrite_cache \
            --do_train \
            --do_eval \
            --do_predict \
            --evaluation_strategy=epoch \
            --save_strategy=epoch
    done
done
