from typing import Dict, List, Optional, Union

import tensorflow as tf
from transformers import BartConfig, TFAutoModel, TFBartForConditionalGeneration, TFBartPretrainedModel
from transformers.modeling_tf_outputs import TFBaseModelOutput, TFSeq2SeqSequenceClassifierOutput
from transformers.modeling_tf_utils import TFSequenceClassificationLoss, input_processing
from transformers.models.bart.modeling_tf_bart import TFBartMainLayer


class TFBartClassificationHead(tf.keras.layers.Layer):
    """Head for sentence-level classification tasks."""

    def __init__(
        self,
        inner_dim: int,
        num_classes: int,
        pooler_dropout: float,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.dense = tf.keras.layers.Dense(inner_dim, name="dense")
        self.dropout = tf.keras.layers.Dropout(pooler_dropout, name="dropout")
        self.out_proj = tf.keras.layers.Dense(num_classes, name="out_proj")

    def call(self, hidden_states: tf.Tensor):
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.dense(hidden_states)
        hidden_states = tf.math.tanh(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.out_proj(hidden_states)
        return hidden_states


class TFBartForSequenceClassification(TFBartPretrainedModel, TFSequenceClassificationLoss):
    """
    TFBart model having classification head for sequence classification.
    call input arguments are same as arguments of TFBartMainLayer input.

    Arguments:
        config: BartConfig, transformers BartConfig instance.

    Output Shape:
        2D tensor with shape:
            `[BatchSize, NumLabels]`
    """

    _keys_to_ignore_on_load_unexpected = [
        r"model.encoder.embed_tokens.weight",
        r"model.decoder.embed_tokens.weight",
    ]

    def __init__(self, config: BartConfig, *args, **kwargs):
        super().__init__(config, *args, **kwargs)

        self.model = TFBartMainLayer(config, name="model")
        self.classification_head = TFBartClassificationHead(
            config.d_model, config.num_labels, config.classifier_dropout, name="classification_head"
        )

    def call(
        self,
        input_ids=None,
        attention_mask=None,
        decoder_input_ids=None,
        decoder_attention_mask=None,
        head_mask=None,
        decoder_head_mask=None,
        cross_attn_head_mask=None,
        encoder_outputs: Optional[TFBaseModelOutput] = None,
        past_key_values=None,
        inputs_embeds=None,
        decoder_inputs_embeds=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        labels=None,
        training=False,
        **kwargs,
    ):
        inputs = input_processing(
            func=self.call,
            config=self.config,
            input_ids=input_ids,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            decoder_attention_mask=decoder_attention_mask,
            head_mask=head_mask,
            decoder_head_mask=decoder_head_mask,
            cross_attn_head_mask=cross_attn_head_mask,
            encoder_outputs=encoder_outputs,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            decoder_inputs_embeds=decoder_inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            labels=labels,
            training=training,
            kwargs_call=kwargs,
        )

        if inputs["decoder_input_ids"] is None and inputs["input_ids"] is not None:
            inputs["decoder_input_ids"] = inputs["input_ids"]

        outputs = self.model(
            inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            decoder_input_ids=inputs["decoder_input_ids"],
            encoder_outputs=inputs["encoder_outputs"],
            decoder_attention_mask=inputs["decoder_attention_mask"],
            head_mask=inputs["head_mask"],
            decoder_head_mask=inputs["decoder_head_mask"],
            cross_attn_head_mask=inputs["cross_attn_head_mask"],
            past_key_values=inputs["past_key_values"],
            inputs_embeds=inputs["inputs_embeds"],
            decoder_inputs_embeds=inputs["decoder_inputs_embeds"],
            use_cache=inputs["use_cache"],
            output_attentions=inputs["output_attentions"],
            output_hidden_states=inputs["output_hidden_states"],
            return_dict=inputs["return_dict"],
            training=inputs["training"],
        )
        last_hidden_state = outputs.last_hidden_state[:, -1, :]

        logits = self.classification_head(hidden_states=last_hidden_state)
        loss = None if inputs["labels"] is None else self.compute_loss(labels=inputs["labels"], logits=logits)

        if not inputs["return_dict"]:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return TFSeq2SeqSequenceClassifierOutput(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            decoder_hidden_states=outputs.decoder_hidden_states,
            decoder_attentions=outputs.decoder_attentions,
            encoder_last_hidden_state=outputs.encoder_last_hidden_state,
            encoder_hidden_states=outputs.encoder_hidden_states,
            encoder_attentions=outputs.encoder_attentions,
        )


class TFBartForSequenceMultiClassification(TFBartPretrainedModel):
    """
    TFBart model having multiple classification heads for sequence classification.
    call input arguments are same as arguments of TFBartMainLayer input.

    Arguments:
        config: BartConfig, transformers BartConfig instance.
        list_num_labels: List[int],

    Output Shape:
        List or Dictionary of 2D tensors with shape:
            `[BatchSize, NumLabels]`
    """

    _keys_to_ignore_on_load_unexpected = [
        r"model.encoder.embed_tokens.weight",
        r"model.decoder.embed_tokens.weight",
    ]

    def __init__(self, config: BartConfig, list_num_labels: Union[List[int], Dict[str, int]], *args, **kwargs):
        super().__init__(config, *args, **kwargs)

        self.model = TFBartMainLayer(config, name="model")
        self.keys = None

        if isinstance(list_num_labels, dict):
            items = list_num_labels.items()
            self.keys = [key for key, _ in items]
            list_num_labels = [value for _, value in items]

        self.classification_heads = [
            TFBartClassificationHead(
                self.config.d_model, num_labels, self.config.classifier_dropout, name=f"classification_head{i}"
            )
            for i, num_labels in enumerate(list_num_labels)
        ]

    def call(
        self,
        input_ids=None,
        attention_mask=None,
        decoder_input_ids=None,
        decoder_attention_mask=None,
        head_mask=None,
        decoder_head_mask=None,
        cross_attn_head_mask=None,
        encoder_outputs: Optional[TFBaseModelOutput] = None,
        past_key_values=None,
        inputs_embeds=None,
        decoder_inputs_embeds=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        labels=None,
        training=False,
        **kwargs,
    ):
        inputs = input_processing(
            func=self.call,
            config=self.config,
            input_ids=input_ids,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            decoder_attention_mask=decoder_attention_mask,
            head_mask=head_mask,
            decoder_head_mask=decoder_head_mask,
            cross_attn_head_mask=cross_attn_head_mask,
            encoder_outputs=encoder_outputs,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            decoder_inputs_embeds=decoder_inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            labels=labels,
            training=training,
            kwargs_call=kwargs,
        )

        if inputs["decoder_input_ids"] is None and inputs["input_ids"] is not None:
            inputs["decoder_input_ids"] = inputs["input_ids"]

        outputs = self.model(
            inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            decoder_input_ids=inputs["decoder_input_ids"],
            encoder_outputs=inputs["encoder_outputs"],
            decoder_attention_mask=inputs["decoder_attention_mask"],
            head_mask=inputs["head_mask"],
            decoder_head_mask=inputs["decoder_head_mask"],
            cross_attn_head_mask=inputs["cross_attn_head_mask"],
            past_key_values=inputs["past_key_values"],
            inputs_embeds=inputs["inputs_embeds"],
            decoder_inputs_embeds=inputs["decoder_inputs_embeds"],
            use_cache=inputs["use_cache"],
            output_attentions=inputs["output_attentions"],
            output_hidden_states=inputs["output_hidden_states"],
            return_dict=inputs["return_dict"],
            training=inputs["training"],
        )
        last_hidden_state = outputs.last_hidden_state[:, -1, :]
        outputs = [classification_head(last_hidden_state) for classification_head in self.classification_heads]

        if self.keys is not None:
            outputs = {key: output for key, output in zip(self.keys, outputs)}
        return outputs


class SemanticTextualSimailarityWrapper(tf.keras.Model):
    """
    Tensorflow model for semantic textual similarity task.

    Arguments:
        model: TFAutoModel, model instance being capable of TFAutomodel.
        embedding_dropout: float, dropout rate of embeddings.

    Output Shape:
        2D tensor with shape:
            `[BatchSize, 1]`
    """

    def __init__(self, model: TFAutoModel, embedding_dropout: float, *args, **kwargs):
        super().__init__(*args, **kwargs, name="semantic_textual_simailarity")

        self.model = model
        self.dropout = tf.keras.layers.Dropout(embedding_dropout)

    def call(self, inputs, training=None, **kwargs):
        input_ids1, input_ids2 = inputs

        embedding1 = self.model(input_ids=input_ids1).last_hidden_state[:, -1, :]
        embedding2 = self.model(input_ids=input_ids2).last_hidden_state[:, -1, :]

        embedding1 = self.dropout(embedding1)
        embedding2 = self.dropout(embedding2)

        return self.cosine_simailarity(embedding1, embedding2)

    def cosine_simailarity(self, embedding1, embedding2):
        dot_product = tf.reduce_sum(embedding1 * embedding2, axis=-1)
        norm = tf.norm(embedding1, axis=-1) * tf.norm(embedding2, axis=-1)
        return dot_product / norm


class GenerationSearchWrapper:
    """Provide search functions for BART model"""

    def __init__(
        self, model: TFBartForConditionalGeneration, max_sequence_length: int, bos_id: int, eos_id: int, pad_id: int = 0
    ):
        """
        :param model: Bart for conditional generation model instance.
        :param max_sequence_length: max sequence length of decoded sequences.
        :param bos_id: bos id for decoding.
        :param eos_id: eos id for decoding.
        :param pad_id: when a sequence is shorter thans other sentences, the back token ids of the sequence is filled pad id.
        """
        self.model = model
        self.max_sequence_length = max_sequence_length
        self.bos_id = bos_id
        self.eos_id = eos_id
        self.pad_id = pad_id

    @tf.function(input_signature=[tf.TensorSpec([None, None], tf.int32), tf.TensorSpec([None, None], tf.int32)])
    def greedy_search(self, encoder_input: tf.Tensor, attention_mask: tf.Tensor) -> tf.Tensor:
        """
        Generate sentences using decoder by beam searching.
        :param encoder_input: seq2seq model inputs [BatchSize, EncoderSequenceLength].
        :param attention_mask: attention mask [BatchSize, EncoderSequenceLength].
        :return: generated tensor shaped. and ppl value of each generated sentences
        """
        batch_size = tf.shape(encoder_input)[0]
        decoder_input = tf.concat(
            [tf.fill([batch_size, 1], self.bos_id), tf.fill([batch_size, self.max_sequence_length - 1], self.pad_id)],
            axis=1,
        )
        log_perplexity = tf.fill([batch_size, 1], 0.0)
        sequence_lengths = tf.fill([batch_size, 1], self.max_sequence_length)
        is_ended = tf.zeros([batch_size, 1], tf.bool)
        sequence_idx = tf.constant(1)

        while sequence_idx < self.max_sequence_length and not tf.reduce_all(is_ended):
            decoder_attention_mask = tf.cast(decoder_input != self.pad_id, tf.int32)

            # [BatchSize, VocabSize]
            output = self.model(
                {
                    "input_ids": encoder_input,
                    "attention_mask": attention_mask,
                    "decoder_input_ids": decoder_input,
                    "decoder_attention_mask": decoder_attention_mask,
                }
            )["logits"][:, sequence_idx - 1, :]
            output = tf.nn.log_softmax(output, axis=1)

            # [BatchSize, 1]
            log_probs, new_tokens = tf.math.top_k(output)
            log_probs, new_tokens = tf.cast(log_probs, log_perplexity.dtype), tf.cast(new_tokens, tf.int32)
            log_perplexity = tf.where(is_ended, log_perplexity, log_perplexity + log_probs)
            new_tokens = tf.where(is_ended, self.pad_id, new_tokens)
            is_ended = tf.logical_or(is_ended, new_tokens == self.eos_id)
            sequence_lengths = tf.where(new_tokens == self.eos_id, tf.shape(decoder_input)[1] + 1, sequence_lengths)

            # [BatchSize, DecoderSequenceLength + 1]
            mask = tf.one_hot(
                sequence_idx,
                depth=self.max_sequence_length,
                on_value=True,
                off_value=False,
            )[tf.newaxis, :]
            mask = tf.repeat(mask, batch_size, axis=0)
            new_tokens = tf.repeat(new_tokens, self.max_sequence_length, axis=1)
            decoder_input = tf.where(mask, new_tokens, decoder_input)

            sequence_idx += 1

        perplexity = tf.squeeze(
            tf.pow(tf.exp(log_perplexity), tf.cast(-1 / sequence_lengths, log_perplexity.dtype)), axis=1
        )
        return decoder_input, perplexity

    @tf.function(
        input_signature=[
            tf.TensorSpec([None, None], tf.int32),
            tf.TensorSpec([None, None], tf.int32),
            tf.TensorSpec([], tf.int32),
            tf.TensorSpec([], tf.float64),
            tf.TensorSpec([], tf.int32),
        ]
    )
    def beam_search(
        self,
        encoder_input: tf.Tensor,
        attention_mask: tf.Tensor,
        beam_size: int,
        alpha: float = 1,
        beta: int = 32,
    ) -> tf.Tensor:
        """
        Generate sentences using decoder by beam searching.

        :param encoder_input: seq2seq model inputs [BatchSize, EncoderSequenceLength].
        :param attention_mask: attention mask [BatchSize, EncoderSequenceLength].
        :param beam_size: beam size for beam search.
        :param alpha: length penalty control variable
        :param beta: length penalty control variable, meaning minimum length.
        :return: generated tensor shaped. and ppl value of each generated sentences
            decoder_input: (BatchSize, BeamSize, SequenceLength)
            perplexity: (BatchSize, BeamSize)
        """
        batch_size = tf.shape(encoder_input)[0]
        decoder_input = tf.fill([batch_size, 1], self.bos_id)
        log_perplexity = tf.fill([batch_size, 1], 0.0)

        def _to_sequence_lengths(decoder_single_input):
            eos_indices = tf.where(decoder_single_input == self.eos_id)
            if tf.size(eos_indices) == 0:
                return tf.size(decoder_single_input, tf.int32)
            return tf.cast(tf.math.reduce_min(eos_indices) + 1, tf.int32)

        def get_sequnce_lengths(decoder_input):
            original_shape = tf.shape(decoder_input)
            decoder_input = tf.reshape(decoder_input, (-1, original_shape[-1]))
            sequence_lengths = tf.map_fn(_to_sequence_lengths, decoder_input)
            return tf.reshape(sequence_lengths, original_shape[:-1])

        def has_eos(decoder_input):
            return tf.reduce_any(decoder_input == self.eos_id, axis=-1)

        def _cond(encoder_input, attention_mask, decoder_input, log_perplexity):
            return tf.shape(decoder_input)[1] < self.max_sequence_length and tf.reduce_any(
                tf.logical_not(has_eos(decoder_input))
            )

        def _body(encoder_input, attention_mask, decoder_input, log_perplexity):
            # [BatchSize, VocabSize]
            output = self.model(
                {"input_ids": encoder_input, "attention_mask": attention_mask, "decoder_input_ids": decoder_input}
            )["logits"][:, -1, :]
            output = tf.nn.log_softmax(output, axis=1)

            # [BatchSize, BeamSize] at first, [BatchSize * BeamSize, BeamSize] after second loops
            log_probs, new_tokens = tf.math.top_k(output, k=beam_size)

            # log_probs: [BatchSize, BeamSize] at first, [BatchSize, BeamSize ** 2] after second loops
            # new_tokens: [BatchSize, 1] at first, [BatchSize * BeamSize, 1] after second loops
            log_probs, new_tokens = tf.reshape(log_probs, [batch_size, -1]), tf.reshape(new_tokens, [-1, 1])
            is_end_sequences = tf.reshape(tf.repeat(has_eos(decoder_input), beam_size, axis=0), [batch_size, -1])
            log_probs = tf.where(is_end_sequences, tf.cast(0.0, log_probs.dtype), log_probs)
            log_probs += tf.cast(tf.repeat(log_perplexity, beam_size, axis=1), log_probs.dtype)

            # Generate first token
            if tf.shape(decoder_input)[1] == 1:
                # [BatchSize * BeamSize, EncoderInputSequence]
                encoder_input = tf.repeat(encoder_input, beam_size, axis=0)
                attention_mask = tf.repeat(attention_mask, beam_size, axis=0)

                # [BatchSize * BeamSize, 2]
                decoder_input = tf.concat([tf.fill([batch_size * beam_size, 1], self.bos_id), new_tokens], axis=1)
                log_perplexity = tf.cast(log_probs, log_perplexity.dtype)
                return encoder_input, attention_mask, decoder_input, log_perplexity
            else:
                # [BatchSize * BeamSize, BeamSize, DecoderSequenceLength + 1]
                decoder_input = tf.reshape(
                    tf.concat((tf.repeat(decoder_input, beam_size, axis=0), new_tokens), axis=1),
                    [batch_size, beam_size * beam_size, -1],
                )

            length_penalty = tf.pow((1 + get_sequnce_lengths(decoder_input)) / (1 + beta), alpha)
            length_penalty = tf.cast(tf.reshape(length_penalty, tf.shape(log_probs)), log_probs.dtype)
            # [BatchSize, BeamSize]
            _, top_indices = tf.math.top_k(log_probs * length_penalty, k=beam_size)

            # [BatchSize * BeamSize, 2]
            indices_for_decoder_input = tf.concat(
                [
                    tf.reshape(tf.repeat(tf.range(batch_size), beam_size), [batch_size * beam_size, 1]),
                    tf.reshape(top_indices, [batch_size * beam_size, 1]),
                ],
                axis=1,
            )

            # [BatchSize * BeamSize, DecoderSequenceLength]
            decoder_input = tf.gather_nd(decoder_input, indices_for_decoder_input)
            log_perplexity = tf.cast(tf.gather_nd(log_probs, indices_for_decoder_input), log_perplexity.dtype)
            log_perplexity = tf.reshape(log_perplexity, [batch_size, beam_size])

            return encoder_input, attention_mask, decoder_input, log_perplexity

        encoder_input, attention_mask, decoder_input, log_perplexity = tf.while_loop(
            _cond,
            _body,
            [encoder_input, attention_mask, decoder_input, log_perplexity],
            shape_invariants=[
                tf.TensorSpec([None, None], tf.int32),
                tf.TensorSpec([None, None], tf.int32),
                tf.TensorSpec([None, None], tf.int32),
                tf.TensorSpec([None, None]),
            ],
        )

        decoder_input = tf.reshape(decoder_input, [batch_size, beam_size, -1])
        sequence_lengths = get_sequnce_lengths(decoder_input)
        decoder_input = tf.where(
            tf.sequence_mask(sequence_lengths, tf.shape(decoder_input)[2]), decoder_input, self.pad_id
        )
        perplexity = tf.pow(tf.exp(log_perplexity), tf.cast(-1 / sequence_lengths, log_perplexity.dtype))

        return decoder_input, perplexity
