#-*- coding:utf-8 -*-
# coding=utf-8
# Copyright 2020 The HuggingFace Datasets Authors and the current dataset script contributor.
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
"""TODO: Add a description here."""

from __future__ import absolute_import, division, print_function

import csv
import json
import os
import pdb
import json
from tqdm import tqdm
from ali_data_util import weibo_data_process

import datasets

label_file=open("../../datasets/cic/labels.txt")
label_list=[line.strip() for line in label_file]
label_file.close()

# TODO: Add BibTeX citation
_CITATION = """\
@InProceedings{huggingface:dataset,
title = {A great new dataset},
authors={huggingface, Inc.
},
year={2020}
}
"""

# TODO: Add description of the dataset here
_DESCRIPTION = """\
This new dataset is designed to solve this great NLP task and is crafted with a lot of care.
"""

# _URL = "https://huggingface.co/great-new-dataset.zip"


# TODO: Name of the dataset usually match the script name with CamelCase instead of snake_case
# Using a specific configuration class is optional, you can also use the base class if you don't need
# to add specific attributes.
# here we give an example for three sub-set of the dataset with difference sizes.
class AliDatasetConfig(datasets.BuilderConfig):
    """ BuilderConfig for AliDataset"""

    def __init__(self, data_size, **kwargs):
        """

        Args:
            data_size: the size of the training set we want to us (xs, s, m, l, xl)
            **kwargs: keyword arguments forwarded to super.
        """
        self.data_size = data_size


class AliDataset(datasets.GeneratorBasedBuilder):
    """TODO: Short description of my dataset."""

    VERSION = datasets.Version("0.0.1")

    # This is an example of a dataset with multiple configurations.
    # If you don't want/need to define several sub-sets in your dataset,
    # just remove the BUILDER_CONFIG_CLASS and the BUILDER_CONFIGS attributes.
    # BUILDER_CONFIG_CLASS = AliDatasetConfig
    # BUILDER_CONFIGS = [
        # AliDatasetConfig(name="my_dataset_" + size, description="A small dataset", data_size=size) for size in ["small", "medium", "large"]
    # ]

    def _info(self):
        # TODO: Specifies the datasets.DatasetInfo object
        return datasets.DatasetInfo(
            # This is the description that will appear on the datasets page.
            description=_DESCRIPTION,
            # This defines the different columns of the dataset and their types
            features=datasets.Features(
                {
                    "sentence": datasets.Value("string"),
                    "label": datasets.features.ClassLabel(names=label_list)
                    # These are the features of your dataset like images, labels ...
                }
            ),
            # If there's a common (input, target) tuple from the features,
            # specify them here. They'll be used if as_supervised=True in
            # builder.as_dataset.
            supervised_keys=None,
            # Homepage of the dataset for documentation
            homepage="xiaoling@30.54.209.130:/media2/xiaoling/multi_classifier_model/yewu_classify",
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager):
        """Returns SplitGenerators."""
        # TODO: Downloads the data and defines the splits
        # dl_manager is a datasets.download.DownloadManager that can be used to
        # download and extract URLs
        # dl_dir = dl_manager.download_and_extract(_URL)
        # data_dir = os.path.join(dl_dir, "great-new-dataset")
        if not self.config.data_files:
            raise ValueError(f"At least one data file must be specified, but got data_files={self.config.data_files}")
        data_files = dl_manager.download_and_extract(self.config.data_files)
        if isinstance(data_files,(list,tuple)):
            raise ValueError("not right input")
        if isinstance(data_files, str):
            return [datasets.SplitGenerator(name=datasets.Split.TRAIN, gen_kwargs={"filepath": data_files})]
        if isinstance(data_files,dict):
            splits = []
            for split_name, files in data_files.items():
                splits.append(datasets.SplitGenerator(name=split_name, gen_kwargs={"filepath": files}))
            return splits

    def preprocess_text(self, text, MAX_LENGTH=256):
        if len(text) > 2 * MAX_LENGTH:
            text = text[:MAX_LENGTH] + text[-MAX_LENGTH:]
        # obj前预处理是为了obj处理太耗时引起的，后处理的目的是防止源码截断的时候只取前max_length个token，而这边是前后各取half_max_length个token
        text = weibo_data_process(text)

        if len(text) > MAX_LENGTH:
            half_max_length = int(MAX_LENGTH/2)
            return text[:half_max_length] + text[-half_max_length:]
        else:
            return text

    def _generate_examples(self, filepath):
        """ Yields examples. """
        # TODO: Yields (key, example) tuples from the dataset
        with open(filepath) as f:
            for id_, row in tqdm(enumerate(f)):
                # data = json.loads(row)
                # data=row.strip().split(',')
                data=json.loads(row.strip())
                label=int(data["label"]) if "label" in data else 0
                sentence=data["sentence"]
                yield id_, {
                    "sentence": sentence,
                    # "sentence": self.preprocess_text(data[0]),
                    # "sentence": data[0],
                    "label": label,
                    }
