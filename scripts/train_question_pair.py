import argparse
import csv
import random
import sys
import urllib.request
from math import ceil
from typing import Tuple

import tensorflow as tf
from transformers import AutoTokenizer

from transformers_bart_finetune.models import TFBartForSequenceClassification
from transformers_bart_finetune.utils import LRScheduler, get_device_strategy, get_logger, path_join, set_random_seed

# fmt: off
QUESTION_PAIR_TRAIN_URI = "https://raw.githubusercontent.com/aisolab/nlp_classification/master/BERT_pairwise_text_classification/qpair/train.txt"
QUESTION_PAIR_VALID_URI = "https://raw.githubusercontent.com/aisolab/nlp_classification/master/BERT_pairwise_text_classification/qpair/validation.txt"
QUESTION_PAIR_TEST_URI = "https://raw.githubusercontent.com/aisolab/nlp_classification/master/BERT_pairwise_text_classification/qpair/test.txt"

parser = argparse.ArgumentParser(description="Script to train Question Pair Task with BART")
parser.add_argument("--pretrained-model", type=str, required=True, help="transformers bart pretrained path")
parser.add_argument("--pretrained-tokenizer", type=str, required=True, help="pretrained tokenizer fast pretrained path")
parser.add_argument("--train-dataset-path", default=QUESTION_PAIR_TRAIN_URI, help="question pair train dataset if using local file")
parser.add_argument("--valid-dataset-path", default=QUESTION_PAIR_VALID_URI, help="question pair validation dataset if using local file")
parser.add_argument("--test-dataset-path", default=QUESTION_PAIR_TEST_URI, help="question pair test dataset if using local file")
parser.add_argument("--output-path", default="output", help="output directory to save log and model checkpoints")
parser.add_argument("--epochs", type=int, default=5)
parser.add_argument("--learning-rate", type=float, default=5e-5)
parser.add_argument("--min-learning-rate", type=float, default=1e-5)
parser.add_argument("--warmup-rate", type=float, default=0.06)
parser.add_argument("--warmup-steps", type=int)
parser.add_argument("--batch-size", type=int, default=128)
parser.add_argument("--dev-batch-size", type=int, default=128)
parser.add_argument("--tensorboard-update-freq", type=int, default=1)
parser.add_argument("--mixed-precision", action="store_true", help="Use mixed precision FP16")
parser.add_argument("--seed", type=int, help="Set random seed")
parser.add_argument("--device", type=str, default="CPU", choices=["CPU", "GPU", "TPU"], help="device to use (TPU or GPU or CPU)")
parser.add_argument("--use-auth-token", action="store_true", help="use huggingface-cli credential for private model")
parser.add_argument("--from-pytorch", action="store_true", help="load from pytorch weight")
# fmt: on


def load_dataset(dataset_path: str, tokenizer: AutoTokenizer, shuffle: bool = False) -> Tuple[tf.data.Dataset, int]:
    """
    Load QuestionPair dataset from local file or web

    :param dataset_path: local file path or file uri
    :param tokenizer: PreTrainedTokenizer for tokenizing
    :param shuffle: whether shuffling lines or not
    :returns: QuestionPair dataset, number of dataset
    """
    if dataset_path.startswith("https://"):
        with urllib.request.urlopen(dataset_path) as response:
            data = response.read().decode("utf-8")
    else:
        with open(dataset_path) as f:
            data = f.read()
    lines = data.splitlines()[1:]
    if shuffle:
        random.shuffle(lines)

    bos = tokenizer.bos_token
    eos = tokenizer.eos_token
    sep = tokenizer.sep_token

    sentences = []
    labels = []
    for question1, question2, label in csv.reader(lines, delimiter="\t"):
        sentences.append(bos + question1 + sep + question2 + eos)
        labels.append(int(label))

    tokens = tokenizer(
        sentences,
        padding=True,
        return_tensors="tf",
        return_token_type_ids=False,
        return_attention_mask=False,
    )["input_ids"]

    dataset = tf.data.Dataset.from_tensor_slices(({"input_ids": tokens}, labels))
    return dataset, len(labels)


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
        tokenizer = AutoTokenizer.from_pretrained(args.pretrained_tokenizer, use_auth_token=args.use_auth_token)

        # Construct Dataset
        logger.info("[+] Load Datasets")
        train_dataset, train_dataset_size = load_dataset(args.train_dataset_path, tokenizer, True)
        train_dataset = train_dataset.batch(args.batch_size)
        valid_dataset = load_dataset(args.valid_dataset_path, tokenizer)[0].batch(args.dev_batch_size)
        test_dataset = load_dataset(args.test_dataset_path, tokenizer)[0].batch(args.dev_batch_size)

        # Model Initialize
        logger.info("[+] Model Initialize")
        model = TFBartForSequenceClassification.from_pretrained(
            args.pretrained_model, num_labels=2, use_auth_token=args.use_auth_token, from_pt=args.from_pytorch
        )
        model.config.id2label = {0: "non-duplicate", 1: "duplicate"}
        model.config.label2id = {"non-duplicate": 0, "duplicate": 1}

        # Model Compile
        logger.info("[+] Model compiling complete")
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
            loss={
                "logits": tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                "encoder_last_hidden_state": None,
            },
            metrics={"logits": [tf.keras.metrics.SparseCategoricalAccuracy(name="accuracy")]},
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
                    monitor="val_logits_accuracy",
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
        _, loss, accuracy = model.evaluate(test_dataset)
        logger.info(f"[+] Test loss: {loss:.4f}, Test Accuracy: {accuracy:.4f}")


if __name__ == "__main__":
    sys.exit(main(parser.parse_args()))
