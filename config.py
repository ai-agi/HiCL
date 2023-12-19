"""
@Author : Fendi Zhang <fendizh001@gmail.com>
@Start-Date : 2022-11-15
@Filename : config.py'
@Framework : Pytorch
"""

from curses import meta
from dataclasses import dataclass, field
# from email.policy import default
# from importlib.metadata import metadata
from typing import Optional
import os

@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """
    model_name_or_path: str = field(
        default="bert-base-uncased",
        # default="roberta-base",
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where do you want to store the pretrained models downloaded from huggingface.co"}
    )
    num_tags: int = field(
        default=3,
        metadata={"help": "number of tags"}
    )
    dropout_rate: float = field(
        default=0.3,
        metadata={"help": "dropout rate"}
    )

    embedding_dim: int = field(
        default=100,
        metadata={"help": "Gaussian embedding output"}
    )

    use_gaussian_cl: bool = field(
        default=True,
        metadata={"help": "Whether to use Gaussian CL."}
    )

    use_coarse_gold_label: bool = field(
        default=True,
        metadata={"help": "Whether to use coarse-grained slot label."}
    )

    use_fine_gold_label: bool = field(
        default=True,
        metadata={"help": "Whether to use coarse-grained slot label."}
    )

    alpha: float = field(
        default=0.8,
        metadata={"help": "tunable hyper-parameters for crf_loss in total loss"}
    )

    beta: float = field(
        default=0.1,
        metadata={"help": "tunable hyper-parameters for crf_loss in total loss"}
    )

    gamma: float = field(
        default=0.1,
        metadata={"help": "tunable hyper-parameters for crf_loss in total loss"}
    )

    zeta: float = field(
        default=1,
        metadata={"help": "tunable hyper-parameters for crf_loss in total loss"}
    )

@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """
    task_name: Optional[str] = field(default="slot_filling", metadata={"help": "The name of the task (ner, pos...)."})
    dataset_name: Optional[str] = field(
        default="SNIPS", metadata={"help": "The name of the dataset to use (via the datasets library)."}
    )
    dataset_path: str = field(
        default="./data",
        metadata={"help": "path of dataset"}
    )
    dataset_config_name: Optional[str] = field(
        default=None, metadata={"help": "The configuration name of the dataset to use (via the datasets library)."}
    )
    train_file: Optional[str] = field(
        default=None, metadata={"help": "The input training data file (a csv or JSON file)."}
    )
    validation_file: Optional[str] = field(
        default=None,
        metadata={"help": "An optional input evaluation data file to evaluate on (a csv or JSON file)."}
    )
    test_file: Optional[str] = field(
        default=None,
        metadata={"help": "An optional input test data file to predict on (a csv or JSON file)."}
    )
    text_column_name: Optional[str] = field(
        default=None, metadata={"help": "The column name of text to input in the file (a csv or JSON file)."}
    )
    label_column_name: Optional[str] = field(
        default=None, metadata={"help": "The column name of label to input in the file (a csv or JSON file)."}
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached training and evaluation sets"}
    )
    preprocessing_num_workers: Optional[int] = field(
        default=None,
        metadata={"help": "The number of processes to use for the preprocessing."}
    )
    max_seq_length: int = field(
        default=128,
        metadata={
            "help": "The maximum total input sequence length after tokenization. If set, sequences longer "
            "than this will be truncated, sequences shorter will be padded."
        }
    )
    pad_to_max_length: bool = field(
        default=False,
        metadata={
            "help": "Whether to pad all samples to model maximum sentence length. "
            "If False, will pad the samples dynamically when batching to the maximum length in the batch. More "
            "efficient on GPU but very bad for TPU."
        }
    )
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of training examples to this "
            "value if set."
        }
    )
    max_eval_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of evaluation examples to this "
            "value if set."
        }
    )
    max_predict_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of prediction examples to this "
            "value if set."
        }
    )
    label_all_tokens: bool = field(
        default=False,
        metadata={
            "help": "Whether to put the label for one word on all tokens of generated by that word or just on the "
            "one (in which case the other tokens will have a padding index)."
        }
    )
    return_entity_level_metrics: bool = field(
        default=False,
        metadata={"help": "Whether to return all the entity levels during evaluation or just the overall ones."}
    )
    target_domain: str = field(
        default="AddToPlaylist",
        metadata={"help": "target domain"}
    )
    n_samples: int = field(
        default=0,
        metadata={"help": "number of samples for few shot learning"}
    )
    early_stopping_patience: int = field(
        default=30,
        metadata={"help": "patience for early stopping"}
    )
    run_mode: str = field(
        default='train',
        metadata={"help": "mode of current configuration, \"train(default)\" or \"test\""}
    )

    def __post_init__(self):
        if self.dataset_name is None and self.train_file is None and self.validation_file is None:
            raise ValueError("Need either a dataset name or a training/validation file.")
        else:
            if self.train_file is not None:
                extension = self.train_file.split(".")[-1]
                assert extension in ["csv", "json"], "`train_file` should be a csv or a json file."
            if self.validation_file is not None:
                extension = self.validation_file.split(".")[-1]
                assert extension in ["csv", "json"], "`validation_file` should be a csv or a json file."
        self.task_name = self.task_name.lower()

@dataclass
class TrainingArguments:
    """
    We implement this TrainingArguments class for code debugging purpose and facilitating code understanding and
    coding. TrainingArguments is the subset of the arguments which usually are used in example scripts **which relate
    to the training loop itself**.

    Using ['HfArgumentParser'] we can turn this class into [argparse](
    https://docs.python.org/3/library/argparse#module-argparse) arguments that can be specified on the command line.

    """

    output_dir: str = field(
        default='./experiments',
        metadata={"help": "the experiments save path"}
    )

    overwrite_output_dir: bool = field(
        default=False,
        metadata={"help": "Whether to overwrite the output directory."}
    )

    do_train: bool = field(
        default=True,
        metadata={"help": "Whether to do train."}
    )

    do_eval: bool = field(
        default=True,
        metadata={"help": "Whether to do eval."}
    )

    evaluation_strategy: str = field(
        default="steps",
        metadata={"help": "evaluation_strategy"}
    )

    save_total_limit: int = field(
        default=1,
        metadata={"help": "save_total_limit"}
    )

    max_steps: int = field(
        default=400000,
        metadata={"help": "max_steps"}
    )

    eval_steps: int = field(
        default=500,
        metadata={"help": "eval_steps"}
    )
    gradient_accumulation_steps: int = field(
        default=1,
        metadata={"help": "gradient_accumulation_steps"}
    )
    eval_accumulation_steps: int = field(
        default=1,
        metadata={"help": "eval_accumulation_steps"}
    )
    per_device_train_batch_size: int = field(
        default=16,
        metadata={"help": "per_device_train_batch_size"}
    )
    per_device_eval_batch_size: int = field(
        default=16,
        metadata={"help": "per_device_eval_batch_size"}
    )
    warmup_steps: int = field(
        default=4000,
        metadata={"help": "warmup_steps"}
    )
    learning_rate: float = field(
        default=1e-5, #=> origin
        # default=5e-5,
        metadata={"help": "learning_rate"}
    )

    use_dep: bool = field(
        default=True,
        metadata={"help": "Whether to use the spacy component of dependency parsing."}
    )

    use_pos: bool = field(
        default=False,
        metadata={"help": "Whether to use the spacy component of part-of-speech."}
    )

    gcl_training_loss: str = field(
        default="KL",
        metadata={"help": "What type of loss to use, KL, euclidean, or joint of KL and classification"}
    )


