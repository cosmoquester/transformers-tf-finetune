import argparse
import csv
import sys
import urllib.request
from math import ceil
from typing import Tuple

import tensorflow as tf
from transformers import PreTrainedTokenizerFast

from transformers_bart_finetune.models import TFBartForSequenceClassification
from transformers_bart_finetune.utils import LRScheduler, get_device_strategy, get_logger, path_join, set_random_seed

QUESTION_PAIR_TRAIN_URI = "https://raw.githubusercontent.com/songys/Question_pair/master/train.txt"
QUESTION_PAIR_VALID_URI = "https://raw.githubusercontent.com/songys/Question_pair/master/validation.txt"
QUESTION_PAIR_TEST_URI = "https://raw.githubusercontent.com/songys/Question_pair/master/test.txt"

# fmt: off
parser = argparse.ArgumentParser(description="Script to train Question Pair Task with BART")
parser.add_argument("--pretrained-model", type=str, required=True, help="transformers bart pretrained path")
parser.add_argument("--pretrained-tokenizer", type=str, required=True, help="pretrained tokenizer fast pretrained path")
parser.add_argument("--train-dataset-path", default=QUESTION_PAIR_TRAIN_URI, help="question pair train dataset if using local file")
parser.add_argument("--valid-dataset-path", default=QUESTION_PAIR_VALID_URI, help="question pair validation dataset if using local file")
parser.add_argument("--test-dataset-path", default=QUESTION_PAIR_TEST_URI, help="question pair test dataset if using local file")
parser.add_argument("--shuffle-buffer-size", type=int, default=5000)
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
# fmt: on


def load_dataset(dataset_path: str, tokenizer: PreTrainedTokenizerFast) -> Tuple[tf.data.Dataset, int]:
    """
    Load QuestionPair dataset from local file or web

    :param dataset_path: local file path or file uri
    :param tokenizer: PreTrainedTokenizer for tokenizing
    :returns: QuestionPair dataset, number of dataset
    """
    if dataset_path.startswith("https://"):
        with urllib.request.urlopen(dataset_path) as response:
            data = response.read().decode("utf-8")
    else:
        with open(dataset_path) as f:
            data = f.read()
    lines = data.splitlines()[1:]

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
    args = parser.parse_args()

    logger = get_logger(__name__)

    if args.seed:
        set_random_seed(args.seed)
        logger.info(f"Set random seed to {args.seed}")

    # Copy config file
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
        tokenizer = PreTrainedTokenizerFast.from_pretrained(args.pretrained_tokenizer)

        # Construct Dataset
        logger.info("[+] Load Datasets")
        train_dataset, train_dataset_size = load_dataset(args.train_dataset_path, tokenizer)
        train_dataset = train_dataset.shuffle(args.shuffle_buffer_size).batch(args.batch_size)
        valid_dataset = load_dataset(args.valid_dataset_path, tokenizer)[0].batch(args.dev_batch_size)
        test_dataset = load_dataset(args.test_dataset_path, tokenizer)[0].batch(args.dev_batch_size)

        # Model Initialize
        logger.info("[+] Model Initialize")
        model = TFBartForSequenceClassification.from_pretrained(args.pretrained_model, num_labels=2)
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
            loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            metrics=[tf.keras.metrics.SparseCategoricalAccuracy(name="accuracy")],
        )

        # Training
        logger.info("[+] Start training")
        model.fit(
            train_dataset,
            validation_data=valid_dataset,
            epochs=args.epochs,
            callbacks=[
                tf.keras.callbacks.ModelCheckpoint(
                    path_join(
                        args.output_path,
                        "models",
                        "model-{epoch}epoch-{val_loss:.4f}loss_{val_accuracy:.4f}acc.ckpt",
                    ),
                    save_weights_only=True,
                    verbose=1,
                ),
                tf.keras.callbacks.TensorBoard(
                    path_join(args.output_path, "logs"), update_freq=args.tensorboard_update_freq
                ),
            ],
        )

        logger.info("[+] Start testing")
        loss, accuracy = model.evaluate(test_dataset)
        logger.info(f"[+] Test loss: {loss:.4f}, Test Accuracy: {accuracy:.4f}")


if __name__ == "__main__":
    sys.exit(main(parser.parse_args()))
