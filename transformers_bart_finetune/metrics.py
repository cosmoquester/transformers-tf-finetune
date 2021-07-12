import tensorflow as tf


class SpearmanCorrelationCoefficient(tf.keras.metrics.Metric):
    """Spearman correlation coefficient"""

    def __init__(self, name="spearman_coef"):
        super().__init__(name=name)

        self.coef_sum = self.add_weight(name="coef_sum", initializer="zeros")
        self.total_count = self.add_weight(name="total_count", initializer="zeros")

    def update_state(self, y_true: tf.Tensor, y_pred: tf.Tensor, sample_weight=None) -> tf.Tensor:
        """
        Update inner metric value using y_ture and y_pred


        :param y_true: true label tensor shaped [BatchSize]
        :param y_pred: pred label tensor shaped [BatchSize]
        :return: pearson correlation coefficient metric value of inputs
        """
        spearman_coef = self.spearman_correlation_coefficient(y_true, y_pred)

        count = tf.cast(tf.shape(y_true)[0], tf.float32)
        self.coef_sum.assign_add(spearman_coef * count)
        self.total_count.assign_add(count)
        return spearman_coef

    def result(self):
        return self.coef_sum / self.total_count

    def get_rank(self, data: tf.Tensor) -> tf.Tensor:
        """
        Get averaged rank of input data

        Example
        data: [ 1,  2,  5, 12,  3,  4,  5,  3,  1,  3]
        return: [ 1.5,  3. ,  8.5, 10. ,  5. ,  7. ,  8.5,  5. ,  1.5,  5. ]

        :param data: Input data shaped [BatchSize]
        :returns: averaged rank of each item.
        """
        data = tf.squeeze(data)
        tf.debugging.assert_rank(data, 1)

        _, index, counts = tf.unique_with_counts(tf.sort(data))
        counts = tf.cast(counts, tf.float32)
        end_numbers = tf.scan(lambda x, y: x + y, counts)
        unique_ranks = end_numbers - (counts - 1) / 2
        average_ranks = tf.map_fn(lambda i: unique_ranks[i], index, dtype=tf.float32)

        increasing_rank = tf.argsort(tf.argsort(data))
        average_ranks_of_input = tf.map_fn(lambda i: average_ranks[i], increasing_rank, dtype=tf.float32)
        return average_ranks_of_input

    def pearson_correlation_coefficient(self, x: tf.Tensor, y: tf.Tensor) -> tf.Tensor:
        """Calculate Pearson correlation coefficients"""
        x = tf.cast(x, tf.float32)
        y = tf.cast(y, tf.float32)

        x_deviation = x - tf.reduce_mean(x, axis=-1)
        y_deviation = y - tf.reduce_mean(y, axis=-1)

        pearson_corr = tf.reduce_sum(x_deviation * y_deviation, axis=-1) / (
            tf.norm(x_deviation, axis=-1) * tf.norm(y_deviation, axis=-1)
        )
        return pearson_corr

    def spearman_correlation_coefficient(self, x: tf.Tensor, y: tf.Tensor) -> tf.Tensor:
        """Calculate Spearman correlation coefficients"""
        x_rank = self.get_rank(x)
        y_rank = self.get_rank(y)
        return self.pearson_correlation_coefficient(x_rank, y_rank)
