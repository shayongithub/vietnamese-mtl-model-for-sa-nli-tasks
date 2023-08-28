from __future__ import annotations

import argparse
import logging
import random
import sys
from dataclasses import dataclass
from dataclasses import field
from typing import List
from typing import Optional

import datasets
import evaluate
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import transformers
from datasets import Dataset
from datasets import DatasetDict
from datasets import load_from_disk
from loguru import logger
from transformers import AutoModel
from transformers import AutoTokenizer
from transformers import DataCollatorWithPadding
from transformers import EvalPrediction
from transformers import IntervalStrategy
from transformers import set_seed
from transformers import Trainer
from transformers import TrainingArguments


@dataclass
class Task:
    task_id: int
    name: str
    task_type: str
    num_labels: int


def tokenize_seq_classification_dataset(
    task_name,
    tokenizer,
    raw_datasets,
    task_id,
    data_args,
    training_args,
    sentence1_key='sentence1',
    sentence2_key='sentence2',
):
    # Padding strategy
    if data_args.pad_to_max_length:
        padding = 'max_length'
    else:
        # We will pad later, dynamically at batch creation, to the max sequence length in each batch
        padding = False

    if data_args.max_seq_length > tokenizer.model_max_length:
        logger.warning(
            f'The max_seq_length passed ({data_args.max_seq_length}) is larger than the maximum length for the'
            f'model ({tokenizer.model_max_length}). Using max_seq_length={tokenizer.model_max_length}.',
        )
    max_seq_length = min(data_args.max_seq_length, tokenizer.model_max_length)

    def tokenize_text(examples):
        args = (
            (examples[sentence1_key],)
            if sentence2_key is None
            else (examples[sentence1_key], examples[sentence2_key])
        )
        result = tokenizer(
            *args, padding=padding, max_length=max_seq_length, truncation=True,
        )

        result['task_ids'] = [task_id] * len(examples['labels'])
        return result

    def tokenize_and_pad_text(examples):
        result = tokenize_text(examples)

        examples['labels'] = [
            [label] + [-100] * (max_seq_length - 1) for label in examples['labels']
        ]
        return result

    with training_args.main_process_first(
        desc=f'Dataset {task_name} map pre-processing',
    ):
        if task_name == 'uit-nlp/vietnamese_students_feedback':
            col_to_remove = [sentence1_key]
        else:
            col_to_remove = [sentence1_key, sentence2_key]

        train_dataset = raw_datasets['train'].map(
            tokenize_text,
            batched=True,
            load_from_cache_file=not data_args.overwrite_cache,
            remove_columns=col_to_remove,
            desc=f'Running tokenizer on dataset {task_name}',
        )

        validation_dataset = raw_datasets['validation'].map(
            tokenize_text,
            batched=True,
            load_from_cache_file=not data_args.overwrite_cache,
            remove_columns=col_to_remove,
            desc=f'Running tokenizer on dataset {task_name}',
        )

    return train_dataset, validation_dataset


def load_seq_classification_dataset(
    task_name, task_id, tokenizer, data_args, training_args,
):
    if task_name == 'uit-nlp/vietnamese_students_feedback':
        raw_datasets = load_from_disk('/content/drive/MyDrive/Shay/training_script/merged_uit_sa_ds')
        sentence2_key = None
    else:
        raw_datasets = load_from_disk('/content/drive/MyDrive/Shay/training_script/merged_vi_nli_ds')

        sentence2_key = 'sentence2'

    num_labels = len(raw_datasets['train'].features['labels'].names)

    train_dataset, validation_dataset = tokenize_seq_classification_dataset(
        task_name,
        tokenizer,
        raw_datasets,
        task_id,
        data_args,
        training_args,
        sentence2_key=sentence2_key,
    )

    task_info = Task(
        task_id=task_id,
        name=task_name,
        num_labels=num_labels,
        task_type='seq_classification',
    )

    return train_dataset, validation_dataset, task_info


def load_datasets(tokenizer, data_args, training_args):
    (
        sentiment_analysis_train_dataset,
        sentiment_analysis_validation_dataset,
        sentiment_analysis_task,
    ) = load_seq_classification_dataset(
        task_name='uit-nlp/vietnamese_students_feedback',
        task_id=0,
        # task_data_files=uit_sa_json_data_files,
        tokenizer=tokenizer,
        data_args=data_args,
        training_args=training_args,
    )

    (
        zsl_train_dataset,
        zsl_validation_dataset,
        zsl_task,
    ) = load_seq_classification_dataset(
        task_name='vinli',
        task_id=1,
        # task_data_files=vi_nli_json_data_files,
        tokenizer=tokenizer,
        data_args=data_args,
        training_args=training_args,
    )

    # Merge train datasets
    train_dataset_df = pd.concat(
        [sentiment_analysis_train_dataset.to_pandas(), zsl_train_dataset.to_pandas()],
    )

    train_dataset = Dataset.from_pandas(train_dataset_df)
    logger.info('Shuffling train dataset')
    train_dataset = train_dataset.shuffle(seed=42)

    # Append validation datasets
    validation_dataset = [
        sentiment_analysis_validation_dataset,
        zsl_validation_dataset,
    ]

    # Merge validation datasets
    # validation_dataset_df = pd.concat([sentiment_analysis_validation_dataset.to_pandas(), zsl_validation_dataset.to_pandas()])  # noqa: E501

    # validation_dataset = Dataset.from_pandas(validation_dataset_df)
    # logger.info('Shuffling validation dataset')
    # validation_dataset = validation_dataset.shuffle(seed=42)

    dataset = DatasetDict(
        {'train': train_dataset, 'validation': validation_dataset})
    tasks = [sentiment_analysis_task, zsl_task]

    return tasks, dataset


task_to_keys = {
    'cola': ('sentence', None),
    'mnli': ('premise', 'hypothesis'),
    'mrpc': ('sentence1', 'sentence2'),
    'qnli': ('question', 'sentence'),
    'qqp': ('question1', 'question2'),
    'rte': ('sentence1', 'sentence2'),
    'sst2': ('sentence', None),
    'stsb': ('sentence1', 'sentence2'),
    'wnli': ('sentence1', 'sentence2'),
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
        metadata={
            'help': 'The name of the task to train on: '
            + ', '.join(task_to_keys.keys()),
        },
    )
    dataset_name: Optional[str] = field(
        default=None,
        metadata={
            'help': 'The name of the dataset to use (via the datasets library).'},
    )
    dataset_config_name: Optional[str] = field(
        default=None,
        metadata={
            'help': 'The configuration name of the dataset to use (via the datasets library).',
        },
    )
    max_seq_length: int = field(
        default=128,
        metadata={
            'help': (
                'The maximum total input sequence length after tokenization. Sequences longer '
                'than this will be truncated, sequences shorter will be padded.'
            ),
        },
    )

    data_cache_dir: Optional[str] = field(
        default='~/hf_datasets', metadata={'help': 'Directory to read/write data'},
    )

    overwrite_cache: bool = field(
        default=False,
        metadata={'help': 'Overwrite the cached preprocessed datasets or not.'},
    )
    pad_to_max_length: bool = field(
        default=True,
        metadata={
            'help': (
                'Whether to pad all samples to `max_seq_length`. '
                'If False, will pad the samples dynamically when batching to the maximum length in the batch.'
            ),
        },
    )
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            'help': (
                'For debugging purposes or quicker training, truncate the number of training examples to this '
                'value if set.'
            ),
        },
    )
    max_eval_samples: Optional[int] = field(
        default=None,
        metadata={
            'help': (
                'For debugging purposes or quicker training, truncate the number of evaluation examples to this '
                'value if set.'
            ),
        },
    )
    max_predict_samples: Optional[int] = field(
        default=None,
        metadata={
            'help': (
                'For debugging purposes or quicker training, truncate the number of prediction examples to this '
                'value if set.'
            ),
        },
    )


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    encoder_name_or_path: str = field(
        metadata={
            'help': 'Path to pretrained model or model identifier from huggingface.co/models',
        },
    )
    config_name: Optional[str] = field(
        default=None,
        metadata={
            'help': 'Pretrained config name or path if not the same as model_name',
        },
    )
    tokenizer_name: Optional[str] = field(
        default=None,
        metadata={
            'help': 'Pretrained tokenizer name or path if not the same as model_name',
        },
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={
            'help': 'Where do you want to store the pretrained models downloaded from huggingface.co',
        },
    )
    use_fast_tokenizer: bool = field(
        default=True,
        metadata={
            'help': 'Whether to use one of the fast tokenizer (backed by the tokenizers library) or not.',
        },
    )
    model_revision: str = field(
        default='main',
        metadata={
            'help': 'The specific model version to use (can be a branch name, tag name or commit id).',
        },
    )
    use_auth_token: bool = field(
        default=False,
        metadata={
            'help': (
                'Will use the token generated when running `huggingface-cli login` (necessary to use this script '
                'with private models).'
            ),
        },
    )
    ignore_mismatched_sizes: bool = field(
        default=False,
        metadata={
            'help': 'Will enable to load a pretrained model whose head dimensions are different.',
        },
    )


class SequenceClassificationHead(nn.Module):
    def __init__(self, hidden_size, num_labels, dropout_p=0.1):
        super().__init__()
        # super().__init__(config=PretrainedConfig())
        self.num_labels = num_labels
        self.dropout = nn.Dropout(dropout_p)
        self.classifier = nn.Linear(hidden_size, num_labels)

        self._init_weights()

    def forward(self, sequence_output, pooled_output, labels=None, **kwargs):
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        loss = None
        if labels is not None:
            if labels.dim() != 1:
                # Remove padding
                labels = labels[:, 0]

            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels),
                            labels.long().view(-1))

        return logits, loss

    def _init_weights(self):
        self.classifier.weight.data.normal_(mean=0.0, std=0.02)
        if self.classifier.bias is not None:
            self.classifier.bias.data.zero_()


class MultiTaskModel(nn.Module):
    def __init__(self, encoder_name_or_path, tasks: List):
        super().__init__()
        # self.tasks = tasks

        self.encoder = AutoModel.from_pretrained(
            encoder_name_or_path, return_dict=False,
        )

        self.output_heads = nn.ModuleDict()
        for task in tasks:
            decoder = self._create_output_head(
                self.encoder.config.hidden_size, task)
            # ModuleDict requires keys to be strings
            self.output_heads[str(task.task_id)] = decoder

    @staticmethod
    def _create_output_head(encoder_hidden_size: int, task):
        if task.task_type == 'seq_classification':
            return SequenceClassificationHead(encoder_hidden_size, task.num_labels)
        else:
            raise NotImplementedError()

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        task_ids=None,
        **kwargs,
    ):
        # print("input_ids: ", input_ids)
        # print("attention_mask: ", attention_mask)
        # print("token_type_ids: ", token_type_ids)
        # print("labels: ", labels)
        # print("task_ids: ", task_ids)

        outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
        )

        sequence_output, pooled_output = outputs[:2]

        unique_task_ids_list = torch.unique(task_ids).tolist()

        loss_list = []
        logits = None
        for unique_task_id in unique_task_ids_list:
            task_id_filter = task_ids == unique_task_id

            logits, task_loss = self.output_heads[str(unique_task_id)].forward(
                sequence_output[task_id_filter],
                pooled_output[task_id_filter],
                labels=None if labels is None else labels[task_id_filter],
                attention_mask=attention_mask[task_id_filter],
            )

            if labels is not None:
                loss_list.append(task_loss)

        # logits are only used for eval. and in case of eval the batch is not multi task
        # For training only the loss is used
        outputs = (logits, outputs[2:])

        if loss_list:
            loss = torch.stack(loss_list)
            outputs = (loss.mean(),) + outputs

        return outputs


accuracy_metric = evaluate.load('accuracy')
f1_metric = evaluate.load('f1')
precision_metric = evaluate.load('precision')
recall_metric = evaluate.load('recall')


def compute_metrics(p: EvalPrediction):
    preds = p.predictions[0] if isinstance(
        p.predictions, tuple) else p.predictions

    if preds.shape[1] == 2:
        average = 'binary'
    elif preds.shape[1] == 3:
        average = 'macro'
    else:
        raise NotImplementedError()

    if preds.ndim == 2:
        logits, labels = p

        predictions = np.argmax(preds, axis=1)

        accuracy = accuracy_metric.compute(predictions=predictions, references=labels)[
            'accuracy'
        ]
        precision = precision_metric.compute(
            predictions=predictions, references=labels, average=average,
        )['precision']
        recall = recall_metric.compute(
            predictions=predictions, references=labels, average=average,
        )['recall']
        f1 = f1_metric.compute(
            predictions=predictions, references=labels, average=average,
        )['f1']

        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
        }
    else:
        raise NotImplementedError()


def parse_args():
    parser = argparse.ArgumentParser(description='Trainer for MTL model')

    parser.add_argument(
        '--save_path', default='./models/mtl_model', help='Path to save models',
    )

    parser.add_argument(
        '--checkpoint_dir', default='./runs', help='Path to save checkpoints',
    )

    parser.add_argument(
        '--encoder_name',
        default='vinai/phobert-base-v2',
        help='Pre-trained language model used',
    )

    parser.add_argument(
        '--max_seq_length', default=512, type=int, help='Max sequence length to pad',
    )

    parser.add_argument(
        '--max_train_samples',
        nargs='?',
        default=None,
        type=int,
        help='Max train samples for training',
    )

    parser.add_argument(
        '--max_eval_samples',
        nargs='?',
        default=None,
        type=int,
        help='Max evaluation samples for eval',
    )

    parser.add_argument(
        '--per_device_train_batch_size', type=int, default=16, help='Train batch size',
    )

    parser.add_argument(
        '--per_device_eval_batch_size', type=int, default=16, help='Evaluate batch size',
    )

    parser.add_argument(
        '--save_steps',
        default=3000,
        type=int,
        help='Number of steps to save checkpoint',
    )

    parser.add_argument(
        '--logging_steps', default=3000, type=int, help='Number of steps to log',
    )

    parser.add_argument(
        '--learning_rate', default=5e-5, type=float, help='Learning rate',
    )

    parser.add_argument(
        '--num_train_epochs',
        default=10,
        type=int,
        help='Number of epochs',
    )

    parser.add_argument('--weight_decay', default=0.01,
                        type=float, help='Weight decay')

    parser.add_argument(
        '--save_strategy',
        default='epoch',
        type=str,
        help='What type of save strategy to use, either steps or epoch',
    )

    parser.add_argument(
        '--load_best_model_at_end',
        nargs='?',
        default=False,
        const=True,
        type=bool,
        help='Whether or not to load the best model found during training at the end of training',
    )

    parser.add_argument(
        '--remove_unused_columns',
        nargs='?',
        default=False,
        const=True,
        type=bool,
        help='Whether to remove unused columns in dataset by model or not',
    )

    args = parser.parse_args()

    return args


def main(model_args, data_args, training_args, args):
    # Setup logging
    logging.basicConfig(
        format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
        datefmt='%m/%d/%Y %H:%M:%S',
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    log_level = training_args.get_process_log_level()

    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    # Log on each process the small summary:
    logger.warning(
        f'Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}'
        + f'distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}',
    )
    logger.info(f'Training/evaluation parameters {training_args}')

    transformers.logging.set_verbosity_info()

    set_seed(42)

    tokenizer = AutoTokenizer.from_pretrained(
        model_args.encoder_name_or_path,
        cache_dir=model_args.cache_dir,
        use_fast=model_args.use_fast_tokenizer,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
    )

    tasks, raw_datasets = load_datasets(tokenizer, data_args, training_args)

    mtl_model = MultiTaskModel(model_args.encoder_name_or_path, tasks)

    # -----------------------------------------------------------------------------
    train_dataset = raw_datasets['train']

    if data_args.max_train_samples is not None:
        train_ind_rans = random.sample(
            range(len(train_dataset)), data_args.max_train_samples,
        )

        train_dataset = train_dataset.select(train_ind_rans)

    # -----------------------------------------------------------------------------
    eval_datasets = raw_datasets['validation']

    if data_args.max_eval_samples is not None:
        for index, eval_dataset in enumerate(eval_datasets):
            eval_ind_rans = random.sample(
                range(len(eval_dataset)), data_args.max_eval_samples,
            )

            eval_datasets[index] = eval_dataset.select(eval_ind_rans)

    data_collator = DataCollatorWithPadding(tokenizer)

    # -----------------------------------------------------------------------------
    # Initialize our Trainer
    trainer = Trainer(
        model=mtl_model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_datasets,
        compute_metrics=compute_metrics,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

    train_result = trainer.train()
    metrics = train_result.metrics

    max_train_samples = (
        data_args.max_train_samples
        if data_args.max_train_samples is not None
        else len(train_dataset)
    )
    metrics['train_samples'] = min(max_train_samples, len(train_dataset))

    trainer.save_model(args.save_path)
    trainer.log_metrics('train', metrics)
    trainer.save_metrics('train', metrics)
    trainer.save_state()

    # Evaluation
    if training_args.do_eval:
        for eval_ds, task in zip(eval_datasets, tasks):
            logger.info(f'*** Evaluate {task} ***')

            metrics = trainer.evaluate(eval_dataset=eval_ds)

            max_eval_samples = (
                data_args.max_eval_samples
                if data_args.max_eval_samples is not None
                else len(eval_ds)
            )
            metrics['eval_samples'] = min(max_eval_samples, len(eval_ds))

            trainer.log_metrics(f'eval for task {task}', metrics)
            print('--------------------------------------------')

            if task.name == 'uit-nlp/vietnamese_students_feedback':
                task_name = 'uit_sa_'
            else:
                task_name = 'vinli_'

            trainer.save_metrics(f'eval_{task_name}', metrics)


if __name__ == '__main__':
    if torch.cuda.is_available():
        device = torch.cuda.get_device_name(0)  # Get the name of the first GPU
        logger.success(f'PyTorch is running on GPU: {device}')
    else:
        logger.warning('PyTorch is running on CPU')
        raise ValueError('GPU is not available')

    transformers.logging.set_verbosity_error()

    args = parse_args()

    model_args = ModelArguments(encoder_name_or_path=args.encoder_name)

    if args.save_strategy == 'epoch':
        save_strategy = IntervalStrategy.EPOCH
    elif args.save_strategy == 'steps':
        save_strategy = IntervalStrategy.STEPS

    training_args = TrainingArguments(
        do_eval=True,
        output_dir=args.checkpoint_dir,
        #   evaluation_strategy = IntervalStrategy.STEPS,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        #   eval_steps = 1,
        save_steps=args.save_steps,
        logging_steps=args.logging_steps,
        learning_rate=args.learning_rate,
        fp16=True,
        num_train_epochs=args.num_train_epochs,
        weight_decay=args.weight_decay,
        save_strategy=save_strategy,
        load_best_model_at_end=args.load_best_model_at_end,
        metric_for_best_model='f1',
        optim='adamw_torch',
        # resume_from_checkpoint=True,
        remove_unused_columns=args.remove_unused_columns,
        # push_to_hub=True
    )

    data_args = DataTrainingArguments(
        max_seq_length=args.max_seq_length,
        max_train_samples=args.max_train_samples,
        max_eval_samples=args.max_eval_samples,
    )

    logger.info('***** Running training arguments *****')
    logger.info(f'  Encoder name = {model_args.encoder_name_or_path}')

    logger.info(f'  Max sequence length = {data_args.max_seq_length:,}')
    logger.info(f'  Max train samples = {data_args.max_train_samples}')
    logger.info(f'  Max evaluation samples = {data_args.max_eval_samples}')

    logger.info(f'  Num Epochs = {training_args.num_train_epochs}')
    logger.info(
        f'  Training batch size per device = {training_args.per_device_train_batch_size}',
    )
    logger.info(
        f'  Evaluation batch size per device = {training_args.per_device_eval_batch_size}',
    )
    logger.info(f'  Checkpoint dir = {training_args.output_dir}')
    logger.info(f'  Save Strategy = {training_args.save_strategy}')
    logger.info(f'  Number of save steps = {training_args.save_steps:,}')
    logger.info(f'  Number of logging steps = {training_args.logging_steps:,}')
    logger.info(f'  Learning rate = {training_args.learning_rate:,}')
    logger.info(f'  Weight decay = {training_args.weight_decay:,}')
    logger.info(
        f'  Load best model at end = {training_args.load_best_model_at_end}')
    logger.info(
        f'  Remove unused columns = {training_args.remove_unused_columns}')

    main(model_args, data_args, training_args, args)
