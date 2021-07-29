import tensorflow as tf

from .metrics import pearson_correlation_coefficient


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


class PearsonCorrelationLoss(tf.keras.losses.Loss):
    """Loss function inversely propotional to PearsonCorrelation"""

    def __init__(self, name="sparse_categorical_crossentropy"):
        super().__init__(name=name, reduction=tf.keras.losses.Reduction.NONE)

    def call(self, y_true, y_pred):
        pearson_coef = pearson_correlation_coefficient(y_true, y_pred)
        pearson_loss = 1 - pearson_coef
        return pearson_loss
