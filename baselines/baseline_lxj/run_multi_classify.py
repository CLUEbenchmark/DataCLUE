# coding=utf-8
# Copyright 2020 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" Finetuning the library models for sequence classification on GLUE."""
# You can also adapt this script on your own text classification task. Pointers for this are left as comments.

import os
import random
import sys
import pdb
import numpy as np
from dataclasses import dataclass, field
from sklearn.metrics import confusion_matrix
from typing import Optional

from datasets import load_dataset, load_metric

import transformers
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import RandomSampler
from transformers import (
    BertTokenizer,
    AutoConfig,
    AutoModelForSequenceClassification,
    AlbertForSequenceClassification,
    BertForSequenceClassification,
    AutoTokenizer,
    EvalPrediction,
    HfArgumentParser,
    PretrainedConfig,
    Trainer,
    TrainingArguments,
    default_data_collator,
    set_seed,
    LongformerForSequenceClassification,
    LongformerTokenizer,
    BalancedBatchSampler,
    logging,
)
from transformers.trainer_utils import is_main_process,set_logger

task_to_keys = {
    "cola": ("sentence", None),
    "mnli": ("premise", "hypothesis"),
    "mrpc": ("sentence1", "sentence2"),
    "qnli": ("question", "sentence"),
    "qqp": ("question1", "question2"),
    "rte": ("sentence1", "sentence2"),
    "sst2": ("sentence", None),
    "stsb": ("sentence1", "sentence2"),
    "wnli": ("sentence1", "sentence2"),
    "fewClue_classify_opposts": ("sentence1", "sentence2"),
    "fewClue_classify_ocnli": ("sentence1", "sentence2"),
    "fewClue_classify_ocemotion": ("sentence", None),
    "dataclue": ("sentence", None),
}


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.

    Using `HfArgumentParser` we can turn this class
    into argparse arguments to be able to specify them on
    the command line.
    """

    task_name: Optional[str] = field(
        default=None,
        metadata={"help": "The name of the task to train on: " + ", ".join(task_to_keys.keys())},
    )
    max_seq_length: int = field(
        default=128,
        metadata={
            "help": "The maximum total input sequence length after tokenization. Sequences longer "
            "than this will be truncated, sequences shorter will be padded."
        },
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached preprocessed datasets or not."}
    )
    pad_to_max_length: bool = field(
        default=True,
        metadata={
            "help": "Whether to pad all samples to `max_seq_length`. "
            "If False, will pad the samples dynamically when batching to the maximum length in the batch."
        },
    )
    train_file: Optional[str] = field(
        default=None, metadata={"help": "A csv or a json file containing the training data."}
    )
    validation_file: Optional[str] = field(
        default=None, metadata={"help": "A csv or a json file containing the validation data."}
    )
    test_file: Optional[str] = field(
        default=None, metadata={"help": "A csv or a json file containing the validation data."}
    )

    model_type: Optional[str] = field(
        default=None, metadata={"help": "albert or bert or roberta,to choose different model and tokenizer expcially for clue models"}
    )

    def __post_init__(self):
        if self.task_name is not None:
            if self.task_name not in task_to_keys.keys():
                raise ValueError("Unknown task, you should pick one in " + ",".join(task_to_keys.keys()))
        elif self.train_file is None or self.validation_file is None:
            raise ValueError("Need either a GLUE task or a training/validation file.")
        else:
            extension = self.train_file.split(".")[-1]
            assert extension in ["csv", "json"], "`train_file` should be a csv or a json file."
            extension = self.validation_file.split(".")[-1]
            assert extension in ["csv", "json"], "`validation_file` should be a csv or a json file."


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    model_name_or_path: str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None, metadata={"help": "Where do you want to store the pretrained models downloaded from s3"}
    )
    use_fast_tokenizer: bool = field(
        default=True,
        metadata={"help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."},
    )


def main():
    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.

    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    if (
        os.path.exists(training_args.output_dir)
        and os.listdir(training_args.output_dir)
        and training_args.do_train
        and not training_args.overwrite_output_dir
    ):
        raise ValueError(
            f"Output directory ({training_args.output_dir}) already exists and is not empty. "
            "Use --overwrite_output_dir to overcome."
        )

    # Setup logging
    os.makedirs(training_args.output_dir, exist_ok=True)

    # logging.basicConfig(
        # format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        # filename=os.path.join(training_args.output_dir,"log.txt"),
        # filemode='a',
        # datefmt="%m/%d/%Y %H:%M:%S",
        # level=logging.INFO if is_main_process(training_args.local_rank) else logging.WARN,
    # )

    # Log on each process the small summary:
    logger = logging.get_logger(log_path=os.path.join(training_args.output_dir,"log.txt"))
    # set_logger(logger,os.path.join(training_args.output_dir,"log.txt"))
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    # Set the verbosity to info of the Transformers logger (on main process only):
    if is_main_process(training_args.local_rank):
        transformers.utils.logging.set_verbosity_info()
    logger.info(f"Model parameters {model_args}")
    logger.info(f"Data parameters {data_args}")
    logger.info(f"Training/evaluation parameters {training_args}")

    # Set seed before initializing model.
    set_seed(training_args.seed)

    # Get the datasets: you can either provide your own CSV/JSON training and evaluation files (see below)
    # or specify a GLUE benchmark task (the dataset will be downloaded automatically from the datasets Hub
    #
    # For CSV/JSON files, this script will use as labels the column called 'label' and as pair of sentences the
    # sentences in columns called 'sentence1' and 'sentence2' if such column exists or the first two columns not named
    # label if at least two columns are provided.
    #
    # If the CSVs/JSONs contain only one non-label column, the script does single sentence classification on this
    # single column. You can easily tweak this behavior (see below)
    #
    # In distributed training, the load_dataset function guarantee that only one local process can concurrently
    # download the dataset.
    if data_args.task_name is not None:
        if data_args.test_file is not None:
            datasets = load_dataset(
                    data_args.task_name+r".py", data_files={"train": data_args.train_file, "validation": data_args.validation_file,"test":data_args.test_file}
            )
        else:
            datasets = load_dataset(
                data_args.task_name+r".py", data_files={"train": data_args.train_file, "validation": data_args.validation_file}
            )
        # Downloading and loading a dataset from the hub.
        # datasets = load_dataset("ali", data_args.task_name)
    elif data_args.train_file.endswith(".csv"):
        # Loading a dataset from local csv files
        datasets = load_dataset(
            "csv", data_files={"train": data_args.train_file, "validation": data_args.validation_file}
        )
    else:
        # Loading a dataset from local json files
        datasets = load_dataset(
            "json", data_files={"train": data_args.train_file, "validation": data_args.validation_file}
        )
    # See more about loading any type of standard or custom dataset at
    # https://huggingface.co/docs/datasets/loading_datasets.html.

    # Labels
    if data_args.task_name is not None:
        is_regression = data_args.task_name == "stsb"
        if not is_regression:
            label_list = datasets["train"].features["label"].names
            num_labels = len(label_list)
        else:
            num_labels = 1
    else:
        # Trying to have good defaults here, don't hesitate to tweak to your needs.
        is_regression = datasets["train"].features["label"].dtype in ["float32", "float64"]
        if is_regression:
            num_labels = 1
        else:
            # A useful fast method:
            # https://huggingface.co/docs/datasets/package_reference/main_classes.html#datasets.Dataset.unique
            label_list = datasets["train"].unique("label")
            label_list.sort()  # Let's sort it for determinism
            num_labels = len(label_list)

    # Load pretrained model and tokenizer
    #
    # In distributed training, the .from_pretrained methods guarantee that only one local process can concurrently
    # download model & vocab.
    config = AutoConfig.from_pretrained(
        model_args.config_name if model_args.config_name else model_args.model_name_or_path,
        num_labels=num_labels,
        finetuning_task=data_args.task_name,
        cache_dir=model_args.cache_dir,
        gradient_checkpointing=True,
    )
    if data_args.model_type in ['clue','roberta','albert']:
        # for clue albert tiny and roberta
        tokenizer = BertTokenizer.from_pretrained(model_args.model_name_or_path)
    # elif data_args.model_type in ['longformer']:
        # tokenizer=LongformerTokenizer.from_pretrained(model_args.model_name_or_path)
    else:
        tokenizer = AutoTokenizer.from_pretrained(
            model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
            cache_dir=model_args.cache_dir,
            use_fast=model_args.use_fast_tokenizer,
        )

    if data_args.model_type in ['albert']:
        # for clue albert tiny
        model = AlbertForSequenceClassification.from_pretrained(model_args.model_name_or_path,config=config)
    elif data_args.model_type in ['roberta']:
        # for clue roberta
        model = BertForSequenceClassification.from_pretrained(model_args.model_name_or_path,config=config)
        # for longformer
        # model=LongformerForSequenceClassification.from_pretrained(model_args.model_name_or_path,config=config)
    else:
        model = AutoModelForSequenceClassification.from_pretrained(
            model_args.model_name_or_path,
            from_tf=bool(".ckpt" in model_args.model_name_or_path),
            config=config,
            cache_dir=model_args.cache_dir,
        )

    # Preprocessing the datasets
    if data_args.task_name is not None:
        sentence1_key, sentence2_key = task_to_keys[data_args.task_name]
    else:
        # Again, we try to have some nice defaults but don't hesitate to tweak to your use case.
        non_label_column_names = [name for name in datasets["train"].column_names if name != "label"]
        if "sentence1" in non_label_column_names and "sentence2" in non_label_column_names:
            sentence1_key, sentence2_key = "sentence1", "sentence2"
        else:
            if len(non_label_column_names) >= 2:
                sentence1_key, sentence2_key = non_label_column_names[:2]
            else:
                sentence1_key, sentence2_key = non_label_column_names[0], None

    # Padding strategy
    if data_args.pad_to_max_length:
        padding = "max_length"
        max_length = data_args.max_seq_length
    else:
        # We will pad later, dynamically at batch creation, to the max sequence length in each batch
        padding = False
        max_length = None

    # Some models have set the order of the labels to use, so let's make sure we do use it.
    label_to_id = None
    if (
        model.config.label2id != PretrainedConfig(num_labels=num_labels).label2id
        and data_args.task_name is not None
        and is_regression
    ):
        # Some have all caps in their config, some don't.
        label_name_to_id = {k.lower(): v for k, v in model.config.label2id.items()}
        if list(sorted(label_name_to_id.keys())) == list(sorted(label_list)):
            label_to_id = {i: label_name_to_id[label_list[i]] for i in range(num_labels)}
        else:
            logger.warn(
                "Your model seems to have been trained with labels, but they don't match the dataset: ",
                f"model labels: {list(sorted(label_name_to_id.keys()))}, dataset labels: {list(sorted(label_list))}."
                "\nIgnoring the model labels as a result.",
            )
    elif data_args.task_name is None:
        label_to_id = {v: i for i, v in enumerate(label_list)}

    def preprocess_function(examples):
        # Tokenize the texts
        args = (
            (examples[sentence1_key],) if sentence2_key is None else (examples[sentence1_key], examples[sentence2_key])
        )
        result = tokenizer(*args, padding=padding, max_length=max_length, truncation=True)

        # Map labels to IDs (not necessary for GLUE tasks)
        if label_to_id is not None and "label" in examples:
            result["label"] = [label_to_id[l] for l in examples["label"]]
        return result

    datasets = datasets.map(preprocess_function, batched=True, load_from_cache_file=not data_args.overwrite_cache)

    train_dataset = datasets["train"]
    eval_dataset = datasets["validation_matched" if data_args.task_name == "mnli" else "validation"]
    if data_args.task_name is not None and data_args.test_file is not None:
        test_dataset = datasets["test_matched" if data_args.task_name == "mnli" else "test"]

    # Log a few random samples from the training set:
    for index in random.sample(range(len(train_dataset)), 10):
        logger.info(f"Sample {index} of the training set: {train_dataset[index]}.")
        input_decode=tokenizer.decode(train_dataset[index]['input_ids'])
        logger.info(f"the tokenizer decode of Sample {index} is: {input_decode}.")

    # Get the metric function
    # if data_args.task_name is not None:
        # metric = load_metric("glue", data_args.task_name)
    # TODO: When datasets metrics include regular accuracy, make an else here and remove special branch from
    # compute_metrics

    # You can define your custom compute_metrics function. It takes an `EvalPrediction` object (a namedtuple with a
    # predictions and label_ids field) and has to return a dictionary string to float.
    def compute_metrics(p: EvalPrediction):
        preds = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
        variance=np.var((np.exp(preds)/np.sum(np.exp(preds),axis=1).reshape(preds.shape[0],-1)),axis=1)
        probility=np.max(np.exp(preds)/np.sum(np.exp(preds),axis=1).reshape(preds.shape[0],-1),axis=1) if not is_regression else None
        preds = np.squeeze(preds) if is_regression else np.argmax(preds, axis=1)
        # if data_args.task_name in ['yewu_luyou']:
            # preds=np.array([1 if preds[index]==0 and probility[index]<0.98 else preds[index] for index in range(len(preds))])
            # preds=np.array([1 if preds[index]==0 and probility[index]<0.9 else preds[index] for index in range(len(preds))])
        if data_args.task_name in ['yewu_classify']:
            preds=np.array([len(label_list)-1 if preds[index]!=len(label_list)-1 and probility[index]<0.98 else preds[index] for index in range(len(preds))])
        confuse_matrix=confusion_matrix(p.label_ids, preds)
        # if data_args.task_name is not None:
            # result = metric.compute(predictions=preds, references=p.label_ids)
            # if len(result) > 1:
                # result["combined_score"] = np.mean(list(result.values())).item()
            # return result
        # elif is_regression:
            # return {"mse": ((preds - p.label_ids) ** 2).mean().item()}
        # else:
            # return {"accuracy": (preds == p.label_ids).astype(np.float32).mean().item()}
        true_positive=np.array([confuse_matrix[i][i] for i in range(len(confuse_matrix))])
        true_label_total=np.sum(confuse_matrix,axis=1)
        predict_label_total=np.sum(confuse_matrix,axis=0)
        precision=(np.around(true_positive/predict_label_total,decimals=3)).tolist()
        true_precision=precision[0]
        recall=(np.around(true_positive/true_label_total,decimals=3)).tolist()
        accuracy=(preds == p.label_ids).astype(np.float32).mean().item()
        log_matrix={
                "accuracy": accuracy, \
                "confuse_matrix": "\n"+str(confuse_matrix), \
                "precision": precision, \
                "true_precision":true_precision, \
                "recall": recall, \
               }
        logger.info(log_matrix)
        return {
                "accuracy": accuracy, \
                "confuse_matrix": "\n"+str(confuse_matrix), \
                "precision": precision, \
                "true_precision":true_precision, \
                # "average_precision": np.around(np.sum(true_positive/predict_label_total)/len(confuse_matrix),decimals=3), \
                "recall": recall, \
                # "average_recall": np.around(np.sum(true_positive/true_label_total)/len(confuse_matrix),decimals=3), \
                "preds":preds.tolist(), \
                "probility":probility.tolist(), \
                "variance":variance.tolist(),
               }

    # user define sampler
    train_sampler=(
                RandomSampler(train_dataset)
                if training_args.local_rank == -1
                else DistributedSampler(train_dataset)
            )
    # Initialize our Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset if training_args.do_eval else None,
        compute_metrics=compute_metrics,
        tokenizer=tokenizer,
        # Data collator will default to DataCollatorWithPadding, so we change it if we already did the padding.
        data_collator=default_data_collator if data_args.pad_to_max_length else None,
        train_sampler=train_sampler,
        # train_sampler=None,
    )

    # Training
    if training_args.do_train:
        trainer.train(
            model_path=model_args.model_name_or_path if os.path.isdir(model_args.model_name_or_path) else None
        )
        trainer.save_model()  # Saves the tokenizer too for easy upload

    # Evaluation
    eval_results = {}
    if training_args.do_eval:
        logger.info("*** Evaluate ***")

        # Loop to handle MNLI double evaluation (matched, mis-matched)
        tasks = [data_args.task_name]
        eval_datasets = [eval_dataset]
        if data_args.task_name == "mnli":
            tasks.append("mnli-mm")
            eval_datasets.append(datasets["validation_mismatched"])

        for eval_dataset, task in zip(eval_datasets, tasks):
            eval_result = trainer.evaluate(eval_dataset=eval_dataset)

            output_eval_file = os.path.join(training_args.output_dir, f"eval_results_{task}.txt")
            if trainer.is_world_process_zero():
                with open(output_eval_file, "w") as writer:
                    logger.info(f"***** Eval results {task} *****")
                    for key, value in eval_result.items():
                        if key not in ["eval_preds","eval_probility","eval_variance"]:
                            logger.info(f"  {key} = {value}")
                        writer.write(f"{key} = {value}\n")
                    label_details=eval_dataset.features['label'].names
                    logger.info("labels is "+str(label_details))
                    writer.write("labels is "+str(label_details)+"\n")
            train_file_save_path=os.path.split(data_args.train_file)[0]
            if 'eval_preds' in eval_result:
                with open(os.path.join(training_args.output_dir,"eval_preds_"+str(training_args.seed)+".txt"),"w") as  writer:
                    for item in eval_result['eval_preds']:
                        writer.write(label_details[int(item)]+"\n")
            if 'eval_probility' in eval_result and eval_result['eval_probility'] is not None:
                with open(os.path.join(training_args.output_dir,"eval_probility_"+str(training_args.seed)+".txt"),"w") as writer:
                    for item in eval_result['eval_probility']:
                        writer.write(str(item)+"\n")

            eval_results.update(eval_result)

    if training_args.do_predict:
        logger.info("*** Test ***")

        # Loop to handle MNLI double evaluation (matched, mis-matched)
        tasks = [data_args.task_name]
        test_datasets = [test_dataset]
        if data_args.task_name == "mnli":
            tasks.append("mnli-mm")
            test_datasets.append(datasets["test_mismatched"])

        for test_dataset, task in zip(test_datasets, tasks):
            # Removing the `label` columns because it contains -1 and Trainer won't like that.
            # source_label=test_dataset['label']
            # test_dataset.remove_columns_("label")

            # predict_result=trainer.predict(test_dataset=test_dataset)
            predict_result=trainer.evaluate(eval_dataset=test_dataset)

            if 'eval_preds' in predict_result:
                with open(os.path.join(training_args.output_dir,"test_preds_"+str(training_args.seed)+".txt"),"w") as  writer:
                    for item in predict_result['eval_preds']:
                        writer.write(label_details[int(item)]+"\n")
            if 'eval_variance' in predict_result:
                with open(os.path.join(training_args.output_dir,"test_variance_"+str(training_args.seed)+".txt"),"w") as  writer:
                    for item in predict_result['eval_variance']:
                        writer.write(str(item)+"\n")
            if 'eval_probility' in predict_result and predict_result['eval_probility'] is not None:
                with open(os.path.join(training_args.output_dir,"test_probility_"+str(training_args.seed)+".txt"),"w") as writer:
                    for item in predict_result['eval_probility']:
                        writer.write(str(item)+"\n")

            # predictions = trainer.predict(test_dataset=test_dataset).predictions
            # predictions=np.exp(predictions)/np.sum(np.exp(predictions),axis=1).reshape(predictions.shape[0],1)
            # probilitys= np.max(predictions,axis=1) if not is_regression else None
            # predictions = np.squeeze(predictions) if is_regression else np.argmax(predictions, axis=1)

            # output_test_file = os.path.join(training_args.output_dir, f"test_results_{task}.txt")
            # if trainer.is_world_process_zero():
                # with open(output_test_file, "w") as writer:
                    # logger.info(f"***** Test results {task} *****")
                    # writer.write("predict\tsource\tsentence\tprobility\n")
                    # for index, item in enumerate(predictions):
                        # if is_regression:
                            # writer.write(f"{index}\t{item:3.3f}\n")
                        # else:
                            # item = label_list[item]
                            # source_label_item=label_list[source_label[index]]
                            # sentence=sentences[index]
                            # probility=probilitys[index]

                            # writer.write(f"{item}\t{source_label_item}\t{sentence}\t{probility}\n")
    return eval_results

def _mp_fn(index):
    # For xla_spawn (TPUs)
    main()

if __name__ == "__main__":
    main()
