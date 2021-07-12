import pytest
import tensorflow as tf

from transformers_bart_finetune.metrics import SpearmanCorrelationCoefficient


@pytest.mark.parametrize(
    "x, true_rank",
    [
        ([1, 2, 5, 12, 3, 4, 5, 3, 1, 3], [1.5, 3.0, 8.5, 10.0, 5.0, 7.0, 8.5, 5.0, 1.5, 5.0]),
        ([1, 2, 3, 4, 5, 6], [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]),
        ([10, 8, 6, 4, 2, 0], [6.0, 5.0, 4.0, 3.0, 2.0, 1.0]),
        ([0, 4, 1, 1, 2, 3, 4, 2, 1, 1], [1.0, 9.5, 3.5, 3.5, 6.5, 8.0, 9.5, 6.5, 3.5, 3.5]),
    ],
)
def test_get_rank(x, true_rank):
    spearman_correlation = SpearmanCorrelationCoefficient()
    tf.debugging.assert_equal(spearman_correlation.get_rank(x), true_rank)


@pytest.mark.parametrize(
    "x, y, true_coef",
    [
        ([1, 2, 5, 12, 3, 4, 5, 3, 1, 3], [1, 2, 3, 4, 5, 6, 7, 8, 9, 10], -0.10970203638498251),
        ([10, 8, 6, 4, 2, 0], [1, 3, 1, 4, 0, 2], 0.03631365196012816),
        ([0, 4, 1, 1, 2, 3, 4, 2, 1, 1], [2, 4, 2, 2, 3, 3, 2, 3, 1, 2], 0.6153846153846153),
    ],
)
def test_pearson_correlation_coefficient(x, y, true_coef):
    spearman_correlation = SpearmanCorrelationCoefficient()
    pred_coef = spearman_correlation.pearson_correlation_coefficient(x, y)
    tf.debugging.assert_near(pred_coef, true_coef)


@pytest.mark.parametrize(
    "x, y, true_coef",
    [
        ([1, 2, 5, 12, 3, 4, 5, 3, 1, 3], [1, 2, 3, 4, 5, 6, 7, 8, 9, 10], 0.006173898221471968),
        ([10, 8, 6, 4, 2, 0], [1, 3, 1, 4, 0, 2], 0.028988551782622423),
        ([0, 4, 1, 1, 2, 3, 4, 2, 1, 1], [2, 4, 2, 2, 3, 3, 2, 3, 1, 2], 0.6536058961645335),
    ],
)
def test_spearman_correlation_coefficient(x, y, true_coef):
    spearman_correlation = SpearmanCorrelationCoefficient()
    pred_coef = spearman_correlation.spearman_correlation_coefficient(x, y)
    tf.debugging.assert_near(pred_coef, true_coef)
