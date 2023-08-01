# coding=utf-8
# Copyright 2021 The HuggingFace Inc. team. All rights reserved.
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
""" Finetuning a ?? Transformers model for sequence classification on GLUE."""
import argparse
import json
import logging
import math
import os
import pickle
import random
from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Any
import datasets
import numpy as np
import pandas as pd
import torch
from datasets import load_dataset, Features, Value
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from sklearn import metrics

import evaluate
import transformers
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import set_seed
from huggingface_hub import Repository
from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    PretrainedConfig,
    SchedulerType,
    default_data_collator,
    get_scheduler, BertTokenizer,
)
from transformers.utils import send_example_telemetry

from model_flash_dep_v4 import RoFormerForSequenceCls
logger = get_logger(__name__)

@dataclass
class DataCollatorWithPadding2(DataCollatorWithPadding):
    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        features1 = [{
            'input_ids': i['input_ids'], 'attention_mask': i['attention_mask'], 'labels': i['labels']
        } for i in features]
        batch = self.tokenizer.pad(
            features1,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors=self.return_tensors,
        )

        features1 = [{
            'input_ids': i['diag_input_ids'], 'attention_mask': i['diag_attention_mask']
        } for i in features]
        batch2 = self.tokenizer.pad(
            features1,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors=self.return_tensors,
        )

        batch["diag_input_ids"] = batch2['input_ids']
        batch["diag_attention_mask"] = batch2['attention_mask']

        features1 = [{
            'input_ids': i['drug_input_ids'], 'attention_mask': i['drug_attention_mask']
        } for i in features]
        batch2 = self.tokenizer.pad(
            features1,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors=self.return_tensors,
        )

        batch["drug_input_ids"] = batch2['input_ids']
        batch["drug_attention_mask"] = batch2['attention_mask']

        batch["exam_input_values"] = torch.tensor([i['exam_input_values'] for i in features], dtype=torch.float)
        batch["exam_value_mask"] = torch.tensor([i['exam_value_mask'] for i in features], dtype=torch.float)

        return batch


def parse_args():
    parser = argparse.ArgumentParser(description="Finetune a transformers model on a text classification task")
    parser.add_argument(
        "--train_file", type=str, default=None, help="A csv or a json file containing the training data."
    )
    parser.add_argument(
        "--validation_file", type=str, default=None, help="A csv or a json file containing the validation data."
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=128,
        help=(
            "The maximum total input sequence length after tokenization. Sequences longer than this will be truncated,"
            " sequences shorter will be padded if `--pad_to_max_lengh` is passed."
        ),
    )
    parser.add_argument(
        "--pad_to_max_length",
        action="store_true",
        help="If passed, pad all samples to `max_length`. Otherwise, dynamic padding is used.",
    )
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
        required=True,
    )
    parser.add_argument(
        "--use_slow_tokenizer",
        action="store_true",
        help="If passed, will use a slow tokenizer (not backed by the ?? Tokenizers library).",
    )
    parser.add_argument(
        "--per_device_train_batch_size",
        type=int,
        default=8,
        help="Batch size (per device) for the training dataloader.",
    )
    parser.add_argument(
        "--per_device_eval_batch_size",
        type=int,
        default=8,
        help="Batch size (per device) for the evaluation dataloader.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=5e-5,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument("--weight_decay", type=float, default=0.0, help="Weight decay to use.")
    parser.add_argument("--num_train_epochs", type=int, default=3, help="Total number of training epochs to perform.")
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=None,
        help="Total number of training steps to perform. If provided, overrides num_train_epochs.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--lr_scheduler_type",
        type=SchedulerType,
        default="linear",
        help="The scheduler type to use.",
        choices=["linear", "cosine", "cosine_with_restarts", "polynomial", "constant", "constant_with_warmup"],
    )
    parser.add_argument(
        "--num_warmup_steps", type=int, default=0, help="Number of steps for the warmup in the lr scheduler."
    )
    parser.add_argument("--output_dir", type=str, default=None, help="Where to store the final model.")
    parser.add_argument("--labels", type=str, default=None, help="Labels.")
    parser.add_argument("--label_weight", type=str, default=None, help="Label weights.")
    parser.add_argument("--seed", type=int, default=42, help="A seed for reproducible training.")
    parser.add_argument(
        "--checkpointing_steps",
        type=str,
        default=None,
        help="Whether the various states should be saved at the end of every n steps, or 'epoch' for each epoch.",
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help="If the training should continue from a checkpoint folder.",
    )
    parser.add_argument(
        "--with_tracking",
        action="store_true",
        help="Whether to enable experiment trackers for logging.",
    )
    parser.add_argument(
        "--report_to",
        type=str,
        default="all",
        help=(
            'The integration to report the results and logs to. Supported platforms are `"tensorboard"`,'
            ' `"wandb"` and `"comet_ml"`. Use `"all"` (default) to report to all integrations.'
            "Only applicable when `--with_tracking` is passed."
        ),
    )
    parser.add_argument(
        "--ignore_mismatched_sizes",
        action="store_true",
        help="Whether or not to enable to load a pretrained model whose head dimensions are different.",
    )
    parser.add_argument(
        "--exam_desc",
        type=str
    )
    parser.add_argument(
        "--exam_labels",
        action="store_true",
        default=True
    )
    parser.add_argument(
        "--empty_pretrained",
        action="store_true",
        default=False
    )
    parser.add_argument(
        "--file_type",
        type=str
    )
    parser.add_argument(
        "--label_name",
        type=str,
        default='y'
    )
    args = parser.parse_args()


    return args


def main():
    args = parse_args()
    # Sending telemetry. Tracking the example usage helps us better allocate resources to maintain them. The
    # information sent is the one passed as arguments along with your Python/PyTorch versions.
    # send_example_telemetry("run_glue_no_trainer", args)

    # Initialize the accelerator. We will let the accelerator handle device placement for us in this example.
    # If we're using tracking, we also need to initialize it here and it will by default pick up all supported trackers
    # in the environment
    accelerator = (
        Accelerator(log_with=args.report_to, logging_dir=args.output_dir) if args.with_tracking else Accelerator()
    )
    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        datasets.utils.logging.set_verbosity_warning()
        transformers.utils.logging.set_verbosity_info()
    else:
        datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()

    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)

    # Handle the repository creation
    if accelerator.is_main_process:
        os.makedirs(args.output_dir, exist_ok=True)
    accelerator.wait_for_everyone()

    # Get the datasets: you can either provide your own CSV/JSON training and evaluation files (see below)
    # or specify a GLUE benchmark task (the dataset will be downloaded automatically from the datasets Hub).

    # For CSV/JSON files, this script will use as labels the column called 'label' and as pair of sentences the
    # sentences in columns called 'sentence1' and 'sentence2' if such column exists or the first two columns not named
    # label if at least two columns are provided.

    # If the CSVs/JSONs contain only one non-label column, the script does single sentence classification on this
    # single column. You can easily tweak this behavior (see below)

    # In distributed training, the load_dataset function guarantee that only one local process can concurrently
    # download the dataset.
    data_files = {}
    if args.validation_file is not None:
        data_files["validation"] = args.validation_file
    extension = (args.train_file if args.train_file is not None else args.validation_file).split(".")[-1]
    raw_datasets = load_dataset(
        extension, data_files=data_files,
        # features=Features({
        #     'jmkh': Value(dtype='string', id=None),
        #     'visit_sn': Value(dtype='string', id=None),
        #     'visit_sn_source': Value(dtype='string', id=None),
        #     'gender_name': Value(dtype='string', id=None),
        #     'age': Value(dtype='int32', id=None),
        #     'marriage_status_name': Value(dtype='string', id=None),
        #     'temperature': Value(dtype='double', id=None),
        #     'pulse': Value(dtype='double', id=None),
        #     'systolicbloodpressure': Value(dtype='double', id=None),
        #     'diastolicbloodpressure': Value(dtype='double', id=None),
        #     'ADA': Value(dtype='double', id=None),
        #     'AFP': Value(dtype='double', id=None),
        #     'ALB': Value(dtype='double', id=None),
        #     'ALP': Value(dtype='double', id=None),
        #     'ALT': Value(dtype='double', id=None),
        #     'APTT': Value(dtype='double', id=None),
        #     'AST': Value(dtype='double', id=None),
        #     'BA#': Value(dtype='double', id=None),
        #     'BA%': Value(dtype='double', id=None),
        #     'BACT': Value(dtype='double', id=None),
        #     'CA199': Value(dtype='double', id=None),
        #     'CEA': Value(dtype='double', id=None),
        #     'CK': Value(dtype='double', id=None),
        #     'CYFRA21-1': Value(dtype='double', id=None),
        #     'Ca': Value(dtype='double', id=None),
        #     'Cl': Value(dtype='double', id=None),
        #     'Cr': Value(dtype='double', id=None),
        #     'DBIL': Value(dtype='double', id=None),
        #     'DD2': Value(dtype='double', id=None),
        #     'EO#': Value(dtype='double', id=None),
        #     'EO%': Value(dtype='double', id=None),
        #     'FIB': Value(dtype='double', id=None),
        #     'FT3': Value(dtype='double', id=None),
        #     'FT4': Value(dtype='double', id=None),
        #     'GGT': Value(dtype='double', id=None),
        #     'GLB': Value(dtype='double', id=None),
        #     'GLU': Value(dtype='double', id=None),
        #     'HBDH': Value(dtype='double', id=None),
        #     'HBsAb': Value(dtype='double', id=None),
        #     'HCT': Value(dtype='double', id=None),
        #     'HDL-C': Value(dtype='double', id=None),
        #     'HGB': Value(dtype='double', id=None),
        #     'HbA1c': Value(dtype='double', id=None),
        #     'K': Value(dtype='double', id=None),
        #     'LDH': Value(dtype='double', id=None),
        #     'LDL-C': Value(dtype='double', id=None),
        #     'LPa': Value(dtype='double', id=None),
        #     'LY#': Value(dtype='double', id=None),
        #     'LY%': Value(dtype='double', id=None),
        #     'MCH': Value(dtype='double', id=None),
        #     'MCHC': Value(dtype='double', id=None),
        #     'MCV': Value(dtype='double', id=None),
        #     'MO#': Value(dtype='double', id=None),
        #     'MO%': Value(dtype='double', id=None),
        #     'MPV': Value(dtype='double', id=None),
        #     'Mg': Value(dtype='double', id=None),
        #     'NE#': Value(dtype='double', id=None),
        #     'NE%': Value(dtype='double', id=None),
        #     'NSE': Value(dtype='double', id=None),
        #     'Na': Value(dtype='double', id=None),
        #     'PCT': Value(dtype='double', id=None),
        #     'PDW': Value(dtype='double', id=None),
        #     'PLT': Value(dtype='double', id=None),
        #     'PT': Value(dtype='double', id=None),
        #     'PT-INR': Value(dtype='double', id=None),
        #     'Phos': Value(dtype='double', id=None),
        #     'RBC': Value(dtype='double', id=None),
        #     'RBC_U': Value(dtype='double', id=None),
        #     'RBP': Value(dtype='double', id=None),
        #     'RDW-CV': Value(dtype='double', id=None),
        #     'SG': Value(dtype='double', id=None),
        #     'TBIL': Value(dtype='double', id=None),
        #     'TC': Value(dtype='double', id=None),
        #     'TG': Value(dtype='double', id=None),
        #     'TP': Value(dtype='double', id=None),
        #     'TSH': Value(dtype='double', id=None),
        #     'TT': Value(dtype='double', id=None),
        #     'UA': Value(dtype='double', id=None),
        #     'Urea': Value(dtype='double', id=None),
        #     'WBC': Value(dtype='double', id=None),
        #     'WBC_U': Value(dtype='double', id=None),
        #     'pH_U': Value(dtype='double', id=None),
        #     'text': Value(dtype='string', id=None),
        #     'drug_names': Value(dtype='string', id=None),
        #     'icd_labels2': Value(dtype='string', id=None),
        #     'disease_name': Value(dtype='string', id=None),
        #     'y_30': Value(dtype='int32', id=None),
        #     'y_60': Value(dtype='int32', id=None),
        #     'y_90': Value(dtype='int32', id=None),
        #     'y_120': Value(dtype='int32', id=None),
        #     'y_150': Value(dtype='int32', id=None),
        #     'y_180': Value(dtype='int32', id=None),
        #     'y_365': Value(dtype='int32', id=None),
        # })
    )

    # See more about loading any type of standard or custom dataset at
    # https://huggingface.co/docs/datasets/loading_datasets.html.

    # Labels
    # Trying to have good defaults here, don't hesitate to tweak to your needs.
    label_list = [str(i) for i in range(2)]
    num_labels = len(label_list)
    # label_list = label_list[:200]
    # num_labels = len(label_list)

    # label_weight = None
    # if args.label_weight:
    #     label_weight = np.load(args.label_weight, allow_pickle=True)[:len(label_list)]
    # Load pretrained model and tokenizer
    #
    # In distributed training, the .from_pretrained methods guarantee that only one local process can concurrently
    # download model & vocab.
    config = AutoConfig.from_pretrained(args.model_name_or_path, num_labels=num_labels)
    config.problem_type = 'multi_label_classification'
    tokenizer = BertTokenizer.from_pretrained(args.model_name_or_path)
    # model = RoFormerForSequenceCls(config, eos_token_id=tokenizer.convert_tokens_to_ids('[SEP]'))
    # model.model.load_state_dict(torch.load(args.model_name_or_path + '/pytorch_model.bin', map_location='cpu'),
    #                             strict=False)
    model = RoFormerForSequenceCls(config, eos_token_id=tokenizer.convert_tokens_to_ids('[SEP]'))
    model.load_state_dict(torch.load(args.model_name_or_path + '/pytorch_model.bin', map_location='cpu'),
                          strict=False)
    model.model.set_input_encoder()
    # model = RoFormerForSequenceCls.from_pretrained(
    #     args.model_name_or_path,
    #     from_tf=bool(".ckpt" in args.model_name_or_path),
    #     config=config,
    #     ignore_mismatched_sizes=args.ignore_mismatched_sizes,
    #     eos_token_id=tokenizer.convert_tokens_to_ids('[SEP]'),
    # )


    # Preprocessing the datasets
    # Again, we try to have some nice defaults but don't hesitate to tweak to your use case.
    sentence1_key, sentence2_key = 'text', None
    label_name = args.label_name

    # Some models have set the order of the labels to use, so let's make sure we do use it.
    label_to_id = None
    label_to_id = {v: i for i, v in enumerate(label_list)}

    if label_to_id is not None:
        model.config.label2id = label_to_id
        model.config.id2label = {id: label for label, id in config.label2id.items()}

    padding = "max_length" if args.pad_to_max_length else False

    person_cols = ['age', 'gender_name', 'marriage_status_name', 'temperature', 'pulse', 'systolicbloodpressure',
                   'diastolicbloodpressure', 'dm_duration']
    exam_cols = ['ADA', 'AFP', 'ALB', 'ALP', 'ALT', 'APTT',
                 'AST', 'BA%', 'CA199', 'CEA', 'CK', 'CYFRA21-1', 'Ca', 'Cl',
                 'Cr', 'DBIL', 'DD2', 'EO%', 'FIB',
                 'FT3', 'FT4', 'GGT', 'GLB', 'GLU', 'HBDH', 'HBsAb', 'HCT', 'HDL-C', 'HGB',
                 'HbA1c', 'K', 'LDH', 'LDL-C', 'LPa',
                 'LY%', 'MCH', 'MCHC', 'MCV', 'MO%',
                 'MPV', 'Mg', 'NE%', 'NSE', 'Na', 'PCT', 'PDW',
                 'PLT', 'PT', 'PT-INR', 'Phos', 'RBC', 'RBC_U',
                 'RBP', 'RDW-CV', 'SG', 'TBIL', 'TC', 'TG', 'TP',
                 'TSH', 'TT', 'UA', 'Urea', 'WBC', 'WBC_U']
    exam_cols = person_cols + exam_cols
    with open(args.exam_desc, 'r', encoding='utf8') as f:
        exam_col_names = json.loads(f.read())

    if args.exam_labels:
        print('set exam_labels...')
        exam_inputs = tokenizer(
            ['[unused1]'] + ['%s %s' % (exam_col_names[i]['name'], exam_col_names[i].get('unit', '')) for i in
                             exam_cols],
            padding=True, max_length=args.max_length, truncation=True, return_tensors='pt'
        )
        model.set_exam_labels(exam_inputs['input_ids'], exam_inputs['attention_mask'])

    def preprocess_function(examples):
        result = tokenizer(examples[sentence1_key], padding=padding, max_length=args.max_length, truncation=True)

        icd_labels = []
        for i in examples['icd_labels2']:
            icd_labels.append(
                '<S> %s %s %s' % (
                    tokenizer.cls_token,
                    ('%s %s' % (tokenizer.sep_token, tokenizer.cls_token)).join(i.split(';')),
                    tokenizer.sep_token
                )
            )
        diags_input = tokenizer(icd_labels, padding=padding, max_length=args.max_length, truncation=True)
        result['diag_input_ids'] = diags_input['input_ids']
        result['diag_attention_mask'] = diags_input['attention_mask']

        drug_names = []
        for i in examples['drug_names']:
            i = '' if i is None else i
            drug_names.append(
                '<T> %s %s %s' % (
                    tokenizer.cls_token,
                    ('%s %s' % (tokenizer.sep_token, tokenizer.cls_token)).join(i.split(';')),
                    tokenizer.sep_token
                )
            )
        drug_input = tokenizer(drug_names, padding=padding, max_length=args.max_length, truncation=True)
        result['drug_input_ids'] = drug_input['input_ids']
        result['drug_attention_mask'] = drug_input['attention_mask']

        nan_token = 103
        cls_token = 101
        # exam_data = list(zip(*[[j for j in examples[i]] for i in exam_cols]))
        exam_data = list(
            zip(*[[j for j in examples.get(i, [nan_token] * len(examples[exam_cols[0]]))] for i in exam_cols]))
        exam_data = [(cls_token,) + i for i in exam_data]
        exam_mask = [[0 if j == nan_token else 1 for j in i] for i in exam_data]

        result['exam_input_values'] = exam_data
        result['exam_value_mask'] = exam_mask

        result["labels"] = examples[label_name]

        # result["labels"] = [i[:200] for i in examples["y"]]

        # def process_number(col):
        #     exam_data = []
        #     for i in examples[col]:
        #         if i == 0:
        #             val = '[MASK]'
        #         else:
        #             val = '[unused%d]' % i
        #         exam_data.append(exam_str_tmpl(col, val))
        #     return exam_data
        #
        # exam_str_list = ['; '.join(i) for i in zip(*[
        #     process_number(i) for i in exam_cols
        # ])]
        #
        # exam_str_list = tokenizer(exam_str_list, padding=padding, max_length=args.max_length, truncation=True)

        # result['exam_input_ids'] = exam_str_list['input_ids']
        # result['exam_attention_mask'] = [[0 if j in [tokenizer.pad_token_id, tokenizer.mask_token_id] else 1 for j in i] for i in result['exam_input_ids'] ]

        return result

    with accelerator.main_process_first():
        processed_datasets = raw_datasets.map(
            preprocess_function,
            batched=True,
            remove_columns=raw_datasets["validation"].column_names,
            desc="Running tokenizer on dataset",
        )

    eval_dataset = processed_datasets["validation"]

    # Log a few random samples from the training set:
    for index in random.sample(range(len(eval_dataset)), 1):
        logger.info(f"Sample {index} of the training set: {eval_dataset[index]}.")

    # DataLoaders creation:
    if args.pad_to_max_length:
        # If padding was already done ot max length, we use the default data collator that will just convert everything
        # to tensors.
        data_collator = default_data_collator
    else:
        # Otherwise, `DataCollatorWithPadding` will apply dynamic padding for us (by padding to the maximum length of
        # the samples passed). When using mixed precision, we add `pad_to_multiple_of=8` to pad all tensors to multiple
        # of 8s, which will enable the use of Tensor Cores on NVIDIA hardware with compute capability >= 7.5 (Volta).
        data_collator = DataCollatorWithPadding2(tokenizer, pad_to_multiple_of=(8 if accelerator.use_fp16 else None))

    eval_dataloader = DataLoader(eval_dataset, collate_fn=data_collator, batch_size=args.per_device_eval_batch_size)


    # Prepare everything with our `accelerator`.
    model, eval_dataloader = accelerator.prepare(
        model, eval_dataloader
    )

    # Get the metric function
    metric = evaluate.load('f1_score.py')

    # Only show the progress bar once on each machine.
    progress_bar = tqdm(range(len(eval_dataloader)), disable=not accelerator.is_local_main_process)
    model.eval()
    samples_seen = 0
    preds, pred_probs, refs = [], [], []
    for step, batch in enumerate(eval_dataloader):
        with torch.no_grad():
            outputs = model(**batch)
        predictions = outputs.logits.softmax(-1).argmax(-1)
        predict_probs = outputs.logits.softmax(-1)[:, 1].contiguous()
        # predictions = outputs.logits
        predictions, references, predict_probs = accelerator.gather((predictions, batch["labels"], predict_probs))
        # If we are in a multiprocess environment, the last batch has duplicates
        if accelerator.num_processes > 1:
            if step == len(eval_dataloader) - 1:
                predictions = predictions[: len(eval_dataloader.dataset) - samples_seen]
                references = references[: len(eval_dataloader.dataset) - samples_seen]
                predict_probs = predict_probs[: len(eval_dataloader.dataset) - samples_seen]
            else:
                samples_seen += references.shape[0]
        metric.add_batch(
            predictions=predictions,
            # predictions=(predictions > 0).int(),
            references=references.int(),
            predict_probs=predict_probs
        )
        preds = preds + predictions.cpu().numpy().tolist()
        refs = refs + references.cpu().numpy().tolist()
        pred_probs = pred_probs + predict_probs.cpu().numpy().tolist()
        progress_bar.update(1)

    eval_metric = metric.compute()
    logger.info(f"{eval_metric}")

    if args.output_dir is not None:
        accelerator.wait_for_everyone()
        with open(os.path.join(args.output_dir, "eval_results_%s.json" % args.file_type), "w") as f:
            json.dump(eval_metric, f)

        with open(os.path.join(args.output_dir, "eval_results_%s.pkl" % args.file_type), "wb") as f:
            pickle.dump({
                'prediction': preds, 'reference': refs, 'predict_probs': pred_probs
            }, f)


if __name__ == "__main__":
    main()
