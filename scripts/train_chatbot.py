import argparse
import csv
import random
import sys
import urllib.request

import tensorflow as tf
from transformers import AutoTokenizer, TFAutoModelForSeq2SeqLM

from transformers_tf_finetune.losses import SparseCategoricalCrossentropy
from transformers_tf_finetune.metrics import SparseCategoricalAccuracy
from transformers_tf_finetune.models import GenerationSearchWrapper
from transformers_tf_finetune.utils import LRScheduler, get_device_strategy, get_logger, path_join, set_random_seed

CHATBOT_URI = "https://raw.githubusercontent.com/songys/Chatbot_data/master/ChatbotData.csv"

# fmt: off
parser = argparse.ArgumentParser(description="Script to train Korean Chatbot conversation with Seq2SeqLM")
parser.add_argument("--pretrained-model", type=str, required=True, help="transformers seq2seq lm pretrained path")
parser.add_argument("--pretrained-tokenizer", type=str, required=True, help="pretrained tokenizer fast pretrained path")
parser.add_argument("--dataset-path", default=CHATBOT_URI, help="dataset if using local file")
parser.add_argument("--output-path", default="output", help="output directory to save log and model checkpoints")
parser.add_argument("--epochs", type=int, default=2)
parser.add_argument("--learning-rate", type=float, default=1e-4)
parser.add_argument("--min-learning-rate", type=float, default=1e-5)
parser.add_argument("--warmup-rate", type=float, default=0.06)
parser.add_argument("--warmup-steps", type=int)
parser.add_argument("--batch-size", type=int, default=16)
parser.add_argument("--dev-batch-size", type=int, default=256)
parser.add_argument("--num-dev-dataset", type=int, default=128)
parser.add_argument("--tensorboard-update-freq", type=int, default=1)
parser.add_argument("--mixed-precision", action="store_true", help="Use mixed precision FP16")
parser.add_argument("--seed", type=int, help="Set random seed")
parser.add_argument("--device", type=str, default="CPU", choices=["CPU", "GPU", "TPU"], help="device to use (TPU or GPU or CPU)")
parser.add_argument("--use-auth-token", action="store_true", help="use huggingface-cli credential for private model")
parser.add_argument("--from-pytorch", action="store_true", help="load from pytorch weight")
parser.add_argument("--max-sequence-length", type=int, default=128, help="max sequence length for decoding")
parser.add_argument("--beam-size", type=int, default=0, help="beam size, use greedy search if this is zero")
# fmt: on


def load_dataset(dataset_path: str, tokenizer: AutoTokenizer, shuffle: bool = False) -> tf.data.Dataset:
    """
    Load Chatbot Conversation dataset from local file or web

    :param dataset_path: local file path or file uri
    :param tokenizer: PreTrainedTokenizer for tokenizing
    :param shuffle: whether shuffling lines or not
    :returns: conversation dataset
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
    inputs = tokenizer(
        questions,
        padding="max_length",
        max_length=max_length,
        return_tensors="tf",
        return_token_type_ids=False,
        return_attention_mask=True,
    )

    target_tokens = tokenizer(
        answers,
        padding="max_length",
        max_length=max_length,
        return_tensors="tf",
        return_token_type_ids=False,
        return_attention_mask=False,
    )["input_ids"]

    dataset = tf.data.Dataset.from_tensor_slices(
        ({**inputs, "decoder_input_ids": target_tokens[:, :-1]}, target_tokens[:, 1:])
    )
    return dataset


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

    strategy = get_device_strategy(args.device)
    with strategy.scope():
        if args.mixed_precision:
            logger.info("Use Mixed Precision FP16")
            mixed_type = "mixed_bfloat16" if args.device == "TPU" else "mixed_float16"
            policy = tf.keras.mixed_precision.experimental.Policy(mixed_type)
            tf.keras.mixed_precision.experimental.set_policy(policy)

        logger.info("[+] Load Tokenizer")
        tokenizer = AutoTokenizer.from_pretrained(args.pretrained_tokenizer, use_auth_token=args.use_auth_token)

        # Construct Dataset
        logger.info("[+] Load Datasets")
        dataset = load_dataset(args.dataset_path, tokenizer, True)
        train_dataset = dataset.skip(args.num_dev_dataset).batch(args.batch_size)
        dev_dataset = dataset.take(args.num_dev_dataset).batch(args.dev_batch_size)

        # Model Initialize
        logger.info("[+] Model Initialize")
        model = TFAutoModelForSeq2SeqLM.from_pretrained(
            args.pretrained_model, use_auth_token=args.use_auth_token, from_pt=args.from_pytorch, use_cache=False
        )

        # Model Compile
        logger.info("[+] Model compiling complete")
        model.compile(
            optimizer=tf.optimizers.Adam(
                LRScheduler(
                    len(train_dataset) * args.epochs,
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
            metrics={"logits": SparseCategoricalAccuracy(ignore_index=tokenizer.pad_token_id, name="accuracy")},
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
        ppls = []
        searcher = GenerationSearchWrapper(
            model,
            args.max_sequence_length,
            tokenizer.convert_tokens_to_ids(tokenizer.bos_token),
            tokenizer.convert_tokens_to_ids(tokenizer.eos_token),
            tokenizer.convert_tokens_to_ids(tokenizer.pad_token),
            beam_size=args.beam_size,
        )
        for batch, _ in strategy.experimental_distribute_dataset(dev_dataset):
            if args.beam_size > 0:
                output, ppl = strategy.run(searcher.beam_search, args=(batch["input_ids"], batch["attention_mask"]))
                output = strategy.gather(output, axis=0)[:, 0, :]
                ppl = strategy.gather(ppl, axis=0)[:, 0]
            else:
                output, ppl = strategy.run(searcher.greedy_search, args=(batch["input_ids"], batch["attention_mask"]))
                output = strategy.gather(output, axis=0)
                ppl = strategy.gather(ppl, axis=0)
            input_tokens.extend(strategy.gather(batch["input_ids"], axis=0).numpy())
            predict_tokens.extend(output.numpy())
            ppls.extend(ppl.numpy())

        input_sentences = tokenizer.batch_decode(input_tokens, skip_special_tokens=True)
        predict_sentences = tokenizer.batch_decode(predict_tokens, skip_special_tokens=True)
        for question, answer, ppl in zip(input_sentences, predict_sentences, ppls):
            print(f"Q: {question} A: {answer} PPL:{ppl:.2f}")


if __name__ == "__main__":
    sys.exit(main(parser.parse_args()))
