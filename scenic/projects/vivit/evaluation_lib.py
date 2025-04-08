"""Functions for evaluation."""

import os
from typing import Any, Callable, Dict, Optional, Sequence, Tuple, Union

from absl import logging
import flax
from flax.metrics import tensorboard
from flax.training import checkpoints
import jax
import jax.numpy as jnp
import numpy as np
from scenic.train_lib import pretrain_utils
from scenic.train_lib import train_utils
from sklearn.metrics import average_precision_score
from tensorflow.io import gfile
from clu import metric_writers


# Aliases for custom types:
Array = Union[jnp.ndarray, np.ndarray]


def restore_checkpoint(
    checkpoint_path: str,
    train_state: Optional[train_utils.TrainState] = None,
    assert_exist: bool = False,
    step: Optional[int] = None
) -> Tuple[train_utils.TrainState, int]:
    """Restores the last checkpoint."""
    if assert_exist:
        glob_path = os.path.join(checkpoint_path, 'checkpoint_*')
        if not gfile.glob(glob_path):
            raise ValueError(f'No checkpoint found in: {checkpoint_path}')

    restored_train_state = checkpoints.restore_checkpoint(checkpoint_path, None, step)
    
    if restored_train_state is None:
        raise ValueError(f'No checkpoint found in: {checkpoint_path}')

    if 'params' in restored_train_state:
        # Checkpoint was trained using optax
        restored_params = flax.core.freeze(restored_train_state['params'])
        restored_model_state = restored_train_state.get('model_state')
    else:
        # Checkpoint was trained using flax.optim
        restored_params = restored_train_state['optimizer']['target']
        restored_model_state = restored_train_state.get('model_state')

    if 'params' in restored_params:
        restored_params = restored_params['params']

    restored_params = dict(checkpoints.convert_pre_linen(restored_params))
    restored_params = flax.core.freeze(restored_params)

    if train_state is None:
        train_state = train_utils.TrainState()

    train_state = train_state.replace(
        params=restored_params,
        model_state=restored_model_state,
        global_step=int(restored_train_state['global_step']),
        rng=restored_train_state['rng'],
        metadata=restored_train_state.get('metadata', None)
    )

    return train_state, int(train_state.global_step)


def compute_mean_average_precision(logits, labels, suffix='',
                                   suffix_separator='_',
                                   return_per_class_ap=False):
  """Computes mean average precision for multi-label classification.

  Args:
    logits: Numpy array of shape [num_examples, num_classes]
    labels: Numpy array of shape [num_examples, num_classes]
    suffix: Suffix to add to the summary
    suffix_separator: Separator before adding the suffix
    return_per_class_ap: If True, return results for each class in the summary.

  Returns:
    summary: Dictionary containing the mean average precision, and also the
      average precision per class.
  """

  assert logits.shape == labels.shape, 'Logits and labels have different shapes'
  n_classes = logits.shape[1]
  average_precisions = []
  if suffix:
    suffix = suffix_separator + suffix
  summary = {}

  for i in range(n_classes):
    ave_precision = average_precision_score(labels[:, i], logits[:, i])
    if np.isnan(ave_precision):
      logging.warning('AP for class %d is NaN', i)

    if return_per_class_ap:
      summary_key = f'per_class_average_precision_{i}{suffix}'
      summary[summary_key] = ave_precision
    average_precisions.append(ave_precision)

  mean_ap = np.nanmean(average_precisions)
  summary[f'mean_average_precision{suffix}'] = mean_ap
  logging.info('Mean AP is %0.3f', mean_ap)

  return summary


def compute_confusion_matrix_metrics(
    confusion_matrices: Sequence[Array],
    return_per_class_metrics: bool) -> Dict[str, float]:
  """Computes classification metrics from a confusion matrix.

  Computes the recall, precision and jaccard index (IoU) from the input
  confusion matrices. The confusion matrices are assumed to be of the form
  [ground_truth, predictions]. In other words, ground truth classes along the
  rows, and predicted classes along the columns.

  Args:
    confusion_matrices: Sequence of [n_batch, n_class, n_class] confusion
      matrices. The first two dimensions will be summed over to get an
      [n_class, n_class] matrix for further metrics.
    return_per_class_metrics: If true, return per-class metrics.

  Returns:
    A dictionary of metrics (recall, precision and jaccard index).
  """

  conf_matrix = np.sum(confusion_matrices, axis=0)  # Sum over eval batches.
  if conf_matrix.ndim != 3:
    raise ValueError(
        'Expecting confusion matrix to have shape '
        f'[batch_size, num_classes, num_classes], got {conf_matrix.shape}.')
  conf_matrix = np.sum(conf_matrix, axis=0)  # Sum over batch dimension.
  n_classes = conf_matrix.shape[0]
  metrics_dict = {}

  # We assume that the confusion matrix is [ground_truth x predictions].
  true_positives = np.diag(conf_matrix)
  sum_rows = np.sum(conf_matrix, axis=0)
  sum_cols = np.sum(conf_matrix, axis=1)

  recall_per_class = true_positives / sum_cols
  precision_per_class = true_positives / sum_rows
  jaccard_index_per_class = (
      true_positives / (sum_rows + sum_cols - true_positives))

  metrics_dict['recall/mean'] = np.nanmean(recall_per_class)
  metrics_dict['precision/mean'] = np.nanmean(precision_per_class)
  metrics_dict['jaccard/mean'] = np.nanmean(jaccard_index_per_class)

  def add_per_class_results(metric: Array, name: str) -> None:
    for i in range(n_classes):
      # We set NaN values (from dividing by 0) to 0, to not cause problems with
      # logging.
      metrics_dict[f'{name}/{i}'] = np.nan_to_num(metric[i])

  if return_per_class_metrics:
    add_per_class_results(recall_per_class, 'recall')
    add_per_class_results(precision_per_class, 'precision')
    add_per_class_results(jaccard_index_per_class, 'jaccard')

  return metrics_dict


def log_eval_summary(
    step: int,
    eval_metrics: Sequence[Dict[str, Tuple[float, int]]],
    extra_eval_summary: Optional[Dict[str, Any]] = None,
    writer: Optional[metric_writers.MetricWriter] = None,
    metrics_normalizer_fn: Optional[Callable[[Dict[str, Tuple[float, int]], str], Dict[str, float]]] = None,
    prefix: str = 'valid',
    key_separator: str = '_'
) -> Dict[str, float]:
    """Computes and logs eval metrics."""
    eval_metrics = train_utils.stack_forest(eval_metrics)
    eval_metrics_summary = jax.tree_util.tree_map(lambda x: x.sum(), eval_metrics)
    
    metrics_normalizer_fn = metrics_normalizer_fn or train_utils.normalize_metrics_summary
    eval_metrics_summary = metrics_normalizer_fn(eval_metrics_summary, 'eval')
    
    extra_eval_summary = extra_eval_summary or {}
    eval_metrics_summary.update(extra_eval_summary)

    if jax.process_index() == 0:
        message = ' | '.join([f'{key}: {val}' for key, val in eval_metrics_summary.items()])
        logging.info('step: %d -- %s -- {%s}', step, prefix, message)

        if writer is not None:
            writer.write_scalars(
                step,
                {f'{prefix}{key_separator}{key}': val for key, val in eval_metrics_summary.items()}
            )
            writer.flush()

    return eval_metrics_summary