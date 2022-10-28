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

    def call(self, hidden_states: tf.Tensor, training: bool = False):
        hidden_states = self.dropout(hidden_states, training=training)
        hidden_states = self.dense(hidden_states)
        hidden_states = tf.math.tanh(hidden_states)
        hidden_states = self.dropout(hidden_states, training=training)
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

        logits = self.classification_head(hidden_states=last_hidden_state, training=inputs["training"])
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
        outputs = [
            classification_head(last_hidden_state, training=inputs["training"])
            for classification_head in self.classification_heads
        ]

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

    def call(self, inputs, training=False, **kwargs):
        input_ids1, input_ids2 = inputs

        embedding1 = self.model(input_ids=input_ids1, training=training).last_hidden_state[:, -1, :]
        embedding2 = self.model(input_ids=input_ids2, training=training).last_hidden_state[:, -1, :]

        embedding1 = self.dropout(embedding1, training=training)
        embedding2 = self.dropout(embedding2, training=training)

        return self.cosine_simailarity(embedding1, embedding2)

    def cosine_simailarity(self, embedding1, embedding2):
        dot_product = tf.reduce_sum(embedding1 * embedding2, axis=-1)
        norm = tf.norm(embedding1, axis=-1) * tf.norm(embedding2, axis=-1)
        return dot_product / norm
