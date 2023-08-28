from __future__ import annotations

import argparse
import logging
import random
import sys
from dataclasses import dataclass
from dataclasses import field
from typing import Optional

import datasets
import evaluate
import numpy as np
import torch
import transformers
from datasets import load_from_disk
from loguru import logger
from transformers import AutoModelForSequenceClassification
from transformers import AutoTokenizer
from transformers import DataCollatorWithPadding
from transformers import IntervalStrategy
from transformers import set_seed
from transformers import Trainer
from transformers import TrainingArguments


# Take from run_glue.py

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


accuracy_metric = evaluate.load('accuracy')
f1_metric = evaluate.load('f1', average='macro')
precision_metric = evaluate.load('precision', average='macro')
recall_metric = evaluate.load('recall', average='macro')


def compute_metrics(eval_preds):
    logits, labels = eval_preds
    predictions = np.argmax(logits, axis=-1)

    accuracy = accuracy_metric.compute(predictions=predictions, references=labels)[
        'accuracy'
    ]
    precision = precision_metric.compute(
        predictions=predictions, references=labels, average='macro',
    )['precision']
    recall = recall_metric.compute(
        predictions=predictions, references=labels, average='macro',
    )['recall']
    f1 = f1_metric.compute(predictions=predictions, references=labels, average='macro')[
        'f1'
    ]

    return {'accuracy': accuracy, 'precision': precision, 'recall': recall, 'f1': f1}


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

    # ViNLI dataset
    raw_datasets = load_from_disk('/content/drive/MyDrive/Shay/training_script/merged_vi_nli_ds')

    train_dataset = raw_datasets['train']
    validation_dataset = raw_datasets['validation']

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    # Tokenize the data
    def pre_process_and_tokenize(batch):
        return tokenizer(
            batch['sentence1'], batch['sentence2'], truncation=True, padding=True,
        )

    tokenized_train_dataset = train_dataset.map(
        pre_process_and_tokenize, batched=True)
    tokenized_validation_dataset = validation_dataset.map(
        pre_process_and_tokenize, batched=True,
    )

    # Load the pre-trained model and tokenizer
    model = AutoModelForSequenceClassification.from_pretrained(
        model_args.encoder_name_or_path, num_labels=3,
    )

    if data_args.max_train_samples is not None:
        train_ind_rans = random.sample(
            range(len(raw_datasets['train'])), data_args.max_train_samples,
        )

        tokenized_train_dataset = tokenized_train_dataset.select(
            train_ind_rans)

    if data_args.max_eval_samples is not None:
        eval_ind_rans = random.sample(
            range(len(raw_datasets['validation'])), data_args.max_eval_samples,
        )

        tokenized_validation_dataset = tokenized_validation_dataset.select(
            eval_ind_rans,
        )

    # -----------------------------------------------------------------------------
    # Initialize our Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train_dataset,
        eval_dataset=tokenized_validation_dataset,
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
    logger.info(f'Metrics: {metrics}')
    trainer.save_metrics('train', metrics)
    trainer.save_state()

    # Evaluation
    if training_args.do_eval:
        logger.info('*** Evaluate Validation dataset ***')

        metrics = trainer.evaluate(eval_dataset=tokenized_validation_dataset)

        max_eval_samples = (
            data_args.max_eval_samples
            if data_args.max_eval_samples is not None
            else len(tokenized_validation_dataset)
        )
        metrics['eval_samples'] = min(
            max_eval_samples, len(tokenized_validation_dataset),
        )

        trainer.log_metrics('eval for zsl', metrics)
        print('--------------------------------------------')

        trainer.save_metrics('eval_zsl', metrics)


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
