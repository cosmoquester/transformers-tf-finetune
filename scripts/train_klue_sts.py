import argparse
import json
import random
import sys
import urllib.request
from math import ceil
from typing import Tuple

import tensorflow as tf
import tensorflow_addons as tfa
from transformers import PreTrainedTokenizerFast

from transformers_bart_finetune.metrics import PearsonCorrelationMetric, SpearmanCorrelationMetric
from transformers_bart_finetune.models import TFBartForSequenceClassification
from transformers_bart_finetune.utils import LRScheduler, get_device_strategy, get_logger, path_join, set_random_seed

# fmt: off
KLUE_STS_TRAIN_URI = "https://raw.githubusercontent.com/KLUE-benchmark/KLUE/main/klue_benchmark/klue-sts-v1/klue-sts-v1_train.json"
KLUE_STS_DEV_URI = "https://raw.githubusercontent.com/KLUE-benchmark/KLUE/main/klue_benchmark/klue-sts-v1/klue-sts-v1_dev.json"

parser = argparse.ArgumentParser(description="Script to train KLUE STS Task with BART")
parser.add_argument("--pretrained-model", type=str, required=True, help="transformers bart pretrained path")
parser.add_argument("--pretrained-tokenizer", type=str, required=True, help="pretrained tokenizer fast pretrained path")
parser.add_argument("--train-dataset-path", default=KLUE_STS_TRAIN_URI, help="klue sts train dataset if using local file")
parser.add_argument("--dev-dataset-path", default=KLUE_STS_DEV_URI, help="klue sts dev dataset if using local file")
parser.add_argument("--output-path", default="output", help="output directory to save log and model checkpoints")
parser.add_argument("--epochs", type=int, default=10)
parser.add_argument("--learning-rate", type=float, default=5e-5)
parser.add_argument("--min-learning-rate", type=float, default=1e-5)
parser.add_argument("--warmup-rate", type=float, default=0.06)
parser.add_argument("--warmup-steps", type=int)
parser.add_argument("--batch-size", type=int, default=128)
parser.add_argument("--dev-batch-size", type=int, default=128)
parser.add_argument("--num-valid-dataset", type=int, default=2000)
parser.add_argument("--tensorboard-update-freq", type=int, default=1)
parser.add_argument("--mixed-precision", action="store_true", help="Use mixed precision FP16")
parser.add_argument("--seed", type=int, help="Set random seed")
parser.add_argument("--device", type=str, default="CPU", choices=["CPU", "GPU", "TPU"], help="device to use (TPU or GPU or CPU)")
parser.add_argument("--use-auth-token", action="store_true", help="use huggingface-cli credential for private model")
# fmt: on


def load_dataset(
    dataset_path: str, tokenizer: PreTrainedTokenizerFast, shuffle: bool = False
) -> Tuple[tf.data.Dataset, int]:
    """
    Load KLUE STS dataset from local file or web

    :param dataset_path: local file path or file uri
    :param tokenizer: PreTrainedTokenizer for tokenizing
    :param shuffle: whether shuffling lines or not
    :returns: KLUE STS dataset, number of dataset
    """
    if dataset_path.startswith("https://"):
        with urllib.request.urlopen(dataset_path) as response:
            data = response.read().decode("utf-8")
    else:
        with open(dataset_path) as f:
            data = f.read()
    examples = json.loads(data)
    if shuffle:
        random.shuffle(examples)

    bos = tokenizer.bos_token
    eos = tokenizer.eos_token
    sep = tokenizer.sep_token

    sentences = []
    binary_labels = []
    real_labels = []
    for example in examples:
        sentences.append(bos + example["sentence1"] + sep + example["sentence2"] + eos)
        binary_labels.append(int(example["labels"]["binary-label"]))
        real_labels.append(float(example["labels"]["real-label"]))

    tokens = tokenizer(
        sentences,
        padding=True,
        return_tensors="tf",
        return_token_type_ids=False,
        return_attention_mask=False,
    )["input_ids"]

    dataset = tf.data.Dataset.from_tensor_slices(({"input_ids": tokens}, (binary_labels, real_labels)))
    return dataset, len(binary_labels)


def main(args: argparse.Namespace):
    logger = get_logger(__name__)

    if args.seed:
        set_random_seed(args.seed)
        logger.info(f"Set random seed to {args.seed}")

    # Copy config file
    assert not tf.io.gfile.exists(args.output_path), f'output path: "{args.output_path}" is already exists!'
    tf.io.gfile.makedirs(args.output_path)
    with tf.io.gfile.GFile(path_join(args.output_path, "argument_configs.txt"), "w") as fout:
        for k, v in vars(args).items():
            fout.write(f"{k}: {v}\n")

    with get_device_strategy(args.device).scope():
        if args.mixed_precision:
            logger.info("Use Mixed Precision FP16")
            mixed_type = "mixed_bfloat16" if args.device == "TPU" else "mixed_float16"
            policy = tf.keras.mixed_precision.experimental.Policy(mixed_type)
            tf.keras.mixed_precision.experimental.set_policy(policy)

        logger.info("[+] Load Tokenizer")
        tokenizer = PreTrainedTokenizerFast.from_pretrained(
            args.pretrained_tokenizer, use_auth_token=args.use_auth_token
        )

        # Construct Dataset
        logger.info("[+] Load Datasets")
        dataset, total_dataset_size = load_dataset(args.train_dataset_path, tokenizer, True)
        train_dataset = dataset.skip(args.num_valid_dataset).batch(args.batch_size)
        valid_dataset = dataset.take(args.num_valid_dataset).batch(args.dev_batch_size)
        dev_dataset = load_dataset(args.dev_dataset_path, tokenizer)[0].batch(args.dev_batch_size)

        # Model Initialize
        logger.info("[+] Model Initialize")
        model = TFBartForSequenceClassification.from_pretrained(
            args.pretrained_model, num_labels=1, use_auth_token=args.use_auth_token
        )

        # Model Compile
        logger.info("[+] Model compiling complete")
        train_dataset_size = total_dataset_size - args.num_valid_dataset
        total_steps = ceil(train_dataset_size / args.batch_size) * args.epochs
        model.compile(
            optimizer=tf.optimizers.Adam(
                LRScheduler(
                    total_steps,
                    args.learning_rate,
                    args.min_learning_rate,
                    args.warmup_rate,
                    args.warmup_steps,
                )
            ),
            loss=[[tf.keras.losses.BinaryCrossentropy(from_logits=True)], None],
            metrics=[
                (
                    tf.keras.metrics.BinaryAccuracy(name="accuracy"),
                    tfa.metrics.F1Score(model.config.num_labels, "macro", threshold=1e-12),
                ),
                (
                    PearsonCorrelationMetric(name="pearson_coef"),
                    SpearmanCorrelationMetric(name="spearman_coef"),
                ),
            ],
        )

        # Training
        logger.info("[+] Start training")
        checkpoint_path = path_join(args.output_path, "best_model.ckpt")
        model.fit(
            train_dataset,
            validation_data=valid_dataset,
            epochs=args.epochs,
            callbacks=[
                tf.keras.callbacks.ModelCheckpoint(
                    checkpoint_path,
                    save_weights_only=True,
                    save_best_only=True,
                    monitor="val_pearson_coef",
                    mode="max",
                    verbose=1,
                ),
                tf.keras.callbacks.TensorBoard(
                    path_join(args.output_path, "logs"), update_freq=args.tensorboard_update_freq
                ),
            ],
        )
        logger.info("[+] Load and Save Best Model")
        model.load_weights(checkpoint_path)
        model.save_pretrained(path_join(args.output_path, "pretrained_model"))

        logger.info("[+] Start testing")
        loss, accuracy, f1_score, pearson_score, spearman_score = model.evaluate(dev_dataset)
        logger.info(
            f"[+] Dev loss: {loss:.4f}, "
            f"Dev Accuracy: {accuracy:.4f}, "
            f"Dev F1: {f1_score:.4f}, "
            f"Dev Pearson: {pearson_score:.4f}, "
            f"Dev Spearman: {spearman_score:.4f}"
        )


if __name__ == "__main__":
    sys.exit(main(parser.parse_args()))
