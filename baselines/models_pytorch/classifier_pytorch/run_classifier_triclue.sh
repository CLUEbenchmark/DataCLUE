
#!/usr/bin/env bash
# @Author: bo.shi
# @Date:   2019-11-04 09:56:36
# @Last Modified by:   bo.shi
# @Last Modified time: 2019-12-05 11:23:45

TASK_NAME="triclue"
MODEL_NAME="chinese_rbtl3_pytorch"
CURRENT_DIR=$(cd -P -- "$(dirname -- "$0")" && pwd -P)
echo "CURRENT_DIR:"+$CURRENT_DIR
export CUDA_VISIBLE_DEVICES="0"
export CLUE_DATA_DIR=../../../datasets  # that is under project path
export OUTPUT_DIR=../../../output_dir/ #  # that is under project path
export PRETRAINED_MODELS_DIR=../../../pre_trained_model # that is project model
export ROBERTA_WWM_SMALL_DIR=$PRETRAINED_MODELS_DIR/$MODEL_NAME

# download base model if not exists
if [ ! -d $ROBERTA_WWM_SMALL_DIR ]; then
  mkdir -p $ROBERTA_WWM_SMALL_DIR
  echo "makedir $ROBERTA_WWM_SMALL_DIR"
fi
cd $ROBERTA_WWM_SMALL_DIR

if [ ! -f "config.json" ] || [ ! -f "vocab.txt" ] || [ ! -f "pytorch_model.bin" ] ; then
  echo "Model not exists, will downloda it now..."
  # rm *
  # you can find detail of the base model from here: https://github.com/ymcui/Chinese-BERT-wwm
  wget -c https://storage.googleapis.com/cluebenchmark/pretrained_models/chinese_rbtl3_pytorch.zip
  unzip chinese_rbtl3_pytorch.zip
  rm chinese_rbtl3_pytorch.zip # chinese_roberta_wwm_large_ext_L-24_H-1024_A-16.zip
else
  echo "Model exists, will reuse it."
fi

# run task
cd $CURRENT_DIR
echo "Start running..."
echo "Data folder.CLUE_DATA_DIR:"$CLUE_DATA_DIR
echo "Model folder.ROBERTA_WWM_SMALL_DIR:"$ROBERTA_WWM_SMALL_DIR

if [ $# == 0 ]; then
    echo "Start training..."
    python run_classifier.py \
      --model_type=bert \
      --model_name_or_path=$ROBERTA_WWM_SMALL_DIR \
      --data_dir=$CLUE_DATA_DIR/${TASK_NAME}/ \
      --task_name=$TASK_NAME \
      --do_train \
      --do_eval \
      --do_lower_case \
      --max_seq_length=32 \
      --per_gpu_train_batch_size=64 \
      --per_gpu_eval_batch_size=32 \
      --learning_rate=2e-5 \
      --num_train_epochs=15 \
      --logging_steps=300 \
      --save_steps=300 \
      --output_dir=$OUTPUT_DIR  \
      --overwrite_output_dir \
      --seed=42

# run below lines to generate predicted file on test.json
elif [ $1 == "predict" ]; then
    echo "Start predict..."
    python run_classifier.py \
      --model_type=bert \
      --model_name_or_path=$ROBERTA_WWM_SMALL_DIR  \
      --data_dir=$CLUE_DATA_DIR/${TASK_NAME}/ \
      --task_name=$TASK_NAME \
      --do_predict \
      --do_lower_case \
      --max_seq_length=32 \
      --per_gpu_train_batch_size=64 \
      --per_gpu_eval_batch_size=32 \
      --learning_rate=2e-5 \
      --num_train_epochs=15 \
      --logging_steps=300 \
      --save_steps=300 \
      --output_dir=$OUTPUT_DIR \
      --overwrite_output_dir \
      --seed=42
 fi
