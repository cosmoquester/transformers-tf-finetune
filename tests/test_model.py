import pytest
import tensorflow as tf
from transformers import BartConfig

from transformers_bart_finetune.models import (
    SemanticTextualSimailarityWrapper,
    TFBartClassificationHead,
    TFBartForSequenceClassification,
)


@pytest.fixture(scope="module")
def config():
    config = BartConfig(
        vocab_size=1000,
        encoder_layers=2,
        decoder_layers=2,
        encoder_ffn_dim=32,
        decoder_ffn_dim=32,
        d_model=16,
        num_labels=3,
        id2label={0: "zero", 1: "one", 2: "two"},
        label2id={"zero": 0, "one": 1, "two": 2},
    )
    return config


def test_classification_head(config: BartConfig):
    head = TFBartClassificationHead(config.d_model, config.num_labels, config.classifier_dropout)

    batch_size = 2
    input = tf.random.normal([batch_size, config.d_model])
    output = head(input)
    tf.debugging.assert_equal(tf.shape(output), [batch_size, config.num_labels])


def test_classification_model(config: BartConfig):
    model = TFBartForSequenceClassification(config)

    batch_size = 3
    sequence_length = 13
    input = tf.random.uniform([batch_size, sequence_length], 0, config.vocab_size, dtype=tf.int32)
    output = model({"input_ids": input})["logits"]
    tf.debugging.assert_equal(tf.shape(output), [batch_size, config.num_labels])


def test_semantic_textual_simailarity_wrapper(config: BartConfig):
    model = SemanticTextualSimailarityWrapper(config)

    batch_size = 3
    sequence_length = 13
    input1 = tf.random.uniform([batch_size, sequence_length], 0, config.vocab_size, dtype=tf.int32)
    input2 = tf.random.uniform([batch_size, sequence_length], 0, config.vocab_size, dtype=tf.int32)
    output = model((input1, input2))

    tf.debugging.assert_equal(tf.shape(output), [batch_size])
    tf.debugging.assert_less_equal(output, 1.0)
    tf.debugging.assert_greater_equal(output, -1.0)
