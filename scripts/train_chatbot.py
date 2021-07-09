import argparse
import csv
import random
import sys
import urllib.request
from math import ceil
from typing import Tuple

import tensorflow as tf
from transformers import PreTrainedTokenizerFast, TFBartForConditionalGeneration

from transformers_bart_finetune.losses import SparseCategoricalCrossentropy
from transformers_bart_finetune.utils import LRScheduler, get_device_strategy, get_logger, path_join, set_random_seed

CHATBOT_URI = "https://raw.githubusercontent.com/songys/Chatbot_data/master/ChatbotData%20.csv"

# fmt: off
parser = argparse.ArgumentParser(description="Script to train Korean Chatbot conversation with BART")
parser.add_argument("--pretrained-model", type=str, required=True, help="transformers bart pretrained path")
parser.add_argument("--pretrained-tokenizer", type=str, required=True, help="pretrained tokenizer fast pretrained path")
parser.add_argument("--dataset-path", default=CHATBOT_URI, help="dataset if using local file")
parser.add_argument("--output-path", default="output", help="output directory to save log and model checkpoints")
parser.add_argument("--epochs", type=int, default=3)
parser.add_argument("--learning-rate", type=float, default=5e-5)
parser.add_argument("--min-learning-rate", type=float, default=1e-5)
parser.add_argument("--warmup-rate", type=float, default=0.06)
parser.add_argument("--warmup-steps", type=int)
parser.add_argument("--batch-size", type=int, default=256)
parser.add_argument("--dev-batch-size", type=int, default=256)
parser.add_argument("--num-dev-dataset", type=int, default=2000)
parser.add_argument("--tensorboard-update-freq", type=int, default=1)
parser.add_argument("--mixed-precision", action="store_true", help="Use mixed precision FP16")
parser.add_argument("--seed", type=int, help="Set random seed")
parser.add_argument("--device", type=str, default="CPU", choices=["CPU", "GPU", "TPU"], help="device to use (TPU or GPU or CPU)")
# fmt: on


def load_dataset(
    dataset_path: str, tokenizer: PreTrainedTokenizerFast, shuffle: bool = False
) -> Tuple[tf.data.Dataset, int]:
    """
    Load Chatbot Conversation dataset from local file or web

    :param dataset_path: local file path or file uri
    :param tokenizer: PreTrainedTokenizer for tokenizing
    :param shuffle: whether shuffling lines or not
    :returns: conversation dataset, number of dataset
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

    questions = []
    answers = []
    for question, answer, _ in csv.reader(lines):
        questions.append(bos + question + eos)
        answers.append(bos + answer + eos)

    max_length = max(len(text) for text in questions + answers)
    input_tokens = tokenizer(
        questions,
        padding="max_length",
        max_length=max_length,
        return_tensors="tf",
        return_token_type_ids=False,
        return_attention_mask=False,
    )["input_ids"]

    target_tokens = tokenizer(
        answers,
        padding="max_length",
        max_length=max_length,
        return_tensors="tf",
        return_token_type_ids=False,
        return_attention_mask=False,
    )["input_ids"]

    dataset = tf.data.Dataset.from_tensor_slices(({"input_ids": input_tokens}, target_tokens))
    return dataset, len(answers)


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
        tokenizer = PreTrainedTokenizerFast.from_pretrained(args.pretrained_tokenizer)

        # Construct Dataset
        logger.info("[+] Load Datasets")
        dataset, total_dataset_size = load_dataset(args.dataset_path, tokenizer, True)
        train_dataset = dataset.skip(args.num_dev_dataset).batch(args.batch_size)
        dev_dataset = dataset.take(args.num_dev_dataset).batch(args.dev_batch_size)

        # Model Initialize
        logger.info("[+] Model Initialize")
        model = TFBartForConditionalGeneration.from_pretrained(args.pretrained_model)

        # Model Compile
        logger.info("[+] Model compiling complete")
        train_dataset_size = total_dataset_size - args.num_dev_dataset
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
                "logits": SparseCategoricalCrossentropy(from_logits=True, ignore_index=tokenizer.pad_token_id),
                "encoder_last_hidden_state": None,
            },
            metrics={"logits": tf.keras.metrics.SparseCategoricalAccuracy(name="accuracy")},
        )

        # Training
        logger.info("[+] Start training")
        checkpoint_path = path_join(args.output_path, "best_model.ckpt")
        model.fit(
            train_dataset,
            validation_data=dev_dataset,
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
        loss, _, accuracy = model.evaluate(dev_dataset)
        logger.info(f"[+] Test loss: {loss:.4f}, Test Accuracy: {accuracy:.4f}")

        logger.info("[+] Start prediction")
        input_tokens = []
        predict_tokens = []
        for batch, _ in dev_dataset:
            output = model(batch)["logits"]
            input_tokens.extend(batch["input_ids"].numpy())
            predict_tokens.extend(tf.argmax(output, axis=2).numpy())

        input_sentences = tokenizer.batch_decode(input_tokens, skip_special_tokens=True)
        predict_sentences = tokenizer.batch_decode(predict_tokens, skip_special_tokens=True)
        for question, answer in zip(input_sentences, predict_sentences):
            print("Q:", question, "A:", answer)


if __name__ == "__main__":
    sys.exit(main(parser.parse_args()))
