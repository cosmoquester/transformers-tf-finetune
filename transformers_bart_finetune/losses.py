import tensorflow as tf


class SparseCategoricalCrossentropy(tf.keras.losses.Loss):
    """Normal sparse categorical crossentrophy with ignore index"""

    def __init__(
        self,
        ignore_index: int = 0,
        from_logits=True,
        reduction=tf.keras.losses.Reduction.SUM_OVER_BATCH_SIZE,
        name="sparse_categorical_crossentropy",
    ):
        super().__init__(name=name, reduction=reduction)
        self.ignore_index = ignore_index
        self.from_logits = from_logits

    def call(self, y_true, y_pred):
        loss = tf.keras.losses.sparse_categorical_crossentropy(y_true, y_pred, from_logits=self.from_logits)
        loss = tf.boolean_mask(loss, y_true != self.ignore_index)
        return loss
