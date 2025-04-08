"""Data generators for the window pileup images."""

import functools
from typing import Optional, Dict, Tuple

from absl import logging
from flax import jax_utils
import jax
import jax.numpy as jnp
import numpy as np
from scenic.dataset_lib import dataset_utils
from scenic.dataset_lib import datasets
import tensorflow as tf


PILEUP_FLANK = 150
PILEUP_HEIGHT = 299
PILEUP_FLANK_PAIRED = 500
PILEUP_HEIGHT_PAIRED = 149
PILEUP_NUM_CHANNELS = 7


def pileup_parser(
    serialized: str,
    mode: str,
    pileup_height: int,
    pileup_width: int,
) -> Tuple[Dict[str, tf.Tensor], Optional[tf.Tensor]]:
  """Parses an EventFeatures serialized example into feature and label tensors."""
  example = tf.io.parse_single_example(
      serialized, {
          'event_name':
              tf.io.FixedLenFeature([1], dtype=tf.string),
          'pileup_1d':
              tf.io.FixedLenFeature(
                  [pileup_height * pileup_width * PILEUP_NUM_CHANNELS],
                  dtype=tf.float32),
          'zygosity':
              tf.io.FixedLenFeature([1], dtype=tf.int64)
      })

  features = {
      'pileup':
          tf.reshape(example['pileup_1d'], [
              pileup_height,
              pileup_width,
              PILEUP_NUM_CHANNELS,
          ])
  }

  # Define the label.
  if mode == 'train':
    label_tensor = tf.squeeze(example['zygosity'], [-1])
    labels = tf.one_hot(label_tensor, 3)
  else:
    labels = None

  return features, labels


def pileup_normalizer(
    features: Dict[str, tf.Tensor],
    labels: tf.Tensor,
) -> Tuple[Dict[str, tf.Tensor], tf.Tensor]:
  """Normalization."""
  features['pileup'] = features['pileup'] / 255
  return features, labels


def create_dataset(
    file_patterns: str = '',
    mode: str = 'test',
    buffer_size: int = 100000,
    batch_size: int = 1024,
    drop_remainder: bool = False,
    paired_read: bool = False,
    tf_data_service_address: str = '',
) -> tf.data.Dataset:
  """Creates dataset from file_patterns.

  Args:
    file_patterns: The data directory/pattern containing data files.
    mode: Whether we are in the train or test mode.
    buffer_size: Size of the tf.data.dataset shuffle buffer.
    batch_size: Size of the tf.data.dataset batch.
    drop_remainder: Whether the last batch should be dropped in the case it has
      fewer than `batch_size` elements.
    paired_read: Whether reads are displayed as pairs or singles in the pileup.
    tf_data_service_address: Indicates how to connect to the tf.data service. If
      it is an empty string, we won’t connect to tf.data service.

  Returns:
    tf.data.Dataset
  """
  logging.info('Finding all data files matching the file pattern.')
  dataset_files = tf.io.matching_files(file_patterns)

  logging.info('Loading TFRecords as bytes.')
  dataset = tf.data.TFRecordDataset(
      dataset_files, num_parallel_reads=tf.data.experimental.AUTOTUNE)

  logging.info('Reading pileup dimensions.')
  if paired_read:
    pileup_height = PILEUP_HEIGHT_PAIRED
    pileup_width = 2 * PILEUP_FLANK_PAIRED - 1
  else:
    pileup_height = PILEUP_HEIGHT
    pileup_width = 2 * PILEUP_FLANK - 1

  # pylint: disable=g-long-lambda
  # Map the parser on every data point.
  logging.info('Parsing input features.')
  dataset = dataset.map(
      lambda x: pileup_parser(
          x,
          mode=mode,
          pileup_height=pileup_height,
          pileup_width=pileup_width,
      ),
      num_parallel_calls=tf.data.experimental.AUTOTUNE,
  )
  # Map the normalizer on every data point.
  logging.info('Normalizing the dataset.')
  dataset = dataset.map(
      pileup_normalizer,
      num_parallel_calls=tf.data.experimental.AUTOTUNE,
  )
  # Set the number of datapoints you want to load and shuffle.
  logging.info('Shuffling the dataset.')
  dataset = dataset.shuffle(buffer_size)

  # Set the batchsize.
  logging.info('Batching the dataset.')
  dataset = dataset.batch(batch_size, drop_remainder=drop_remainder)
  if tf_data_service_address:
    dataset = dataset.apply(
        tf.data.experimental.service.distribute(
            processing_mode='parallel_epochs', service=tf_data_service_address))
  return dataset


def preprocess(features, label):
  """Preprocessing code specific to ViT models."""
  return {
      'inputs': tf.image.resize(
          features['pileup'],
          [256, 256]),  # Resize pileups to make side length divisible by 4.
      'label': label
  }


def build_dataset(dataset_fn,
                  batch_size=None,
                  shuffle_buffer_size=256,
                  seed=None,
                  strategy=None,
                  **dataset_kwargs):
  """Dataset builder that takes care of strategy, batching and shuffling.

  Args:
    dataset_fn: function; A function that loads the dataset.
    batch_size: int; Size of the batch.
    shuffle_buffer_size: int; Size of the buffer for used for shuffling.
    seed: int; Random seed used for shuffling.
    strategy: TF strategy that handles the distribution policy.
    **dataset_kwargs: dict; Arguments passed to TFDS.

  Returns:
    Dataset.
  """

  def _dataset_fn(input_context=None):
    """Dataset function."""
    replica_batch_size = batch_size
    if input_context:
      replica_batch_size = input_context.get_per_replica_batch_size(batch_size)
    ds = dataset_fn(**dataset_kwargs)
    split = dataset_kwargs.get('split')
    if split == 'train':
      # First repeat then shuffle, then batch.
      ds = ds.repeat()
      local_seed = seed  # Seed for this machine.
      if local_seed is not None and input_context:
        local_seed += input_context.input_pipeline_id
      ds = ds.shuffle(shuffle_buffer_size, seed=local_seed)
      ds = ds.batch(replica_batch_size, drop_remainder=True)
    else:  # Test and validation.
      # First batch then repeat.
      ds = ds.batch(replica_batch_size, drop_remainder=False)
      ds = ds.repeat()
    options = tf.data.Options()
    options.experimental_optimization.parallel_batch = True
    ds = ds.with_options(options)
    return ds.prefetch(tf.data.experimental.AUTOTUNE)

  if strategy:
    ds = strategy.experimental_distribute_datasets_from_function(_dataset_fn)
  else:
    ds = _dataset_fn()

  return ds


@datasets.add_dataset('pileup_window')
def get_dataset(*,
                batch_size,
                eval_batch_size,
                num_shards,
                dtype_str='float32',
                shuffle_seed=0,
                rng=None,
                prefetch_buffer_size=2,
                dataset_configs=None,
                dataset_service_address: Optional[str] = None):
  """Returns generators for the pileup window train, validation, and test set.

  Args:
    batch_size: int; Determines the train batch size.
    eval_batch_size: int; Determines the evaluation batch size.
    num_shards: int;  Number of shards --> batch shape: [num_shards, bs, ...].
    dtype_str: Data type of the image (e.g. 'float32').
    shuffle_seed: int; Seed for shuffling the training data.
    rng: JAX rng key, which can be used for augmentation, shuffling, etc.
    prefetch_buffer_size: int; Buffer size for the prefetch.
    dataset_configs: dict; Dataset specific configurations.
    dataset_service_address: If set, will distribute the training dataset using
      the given tf.data service at the given address.

  Returns:
    A dataset_utils.Dataset() which includes a train_iter, a valid_iter,
      a test_iter, and a dict of meta_data.
  """
  del rng
  del dataset_configs

  def build_pileup_window_dataset(split):
    """dataset_fn called by data.build_dataset(**kwargs)."""

    if split == 'train':
      # Add path to your training data here:
      path = ''
    else:
      # Add path to your test data here:
      path = ''
    if not path:
      raise ValueError('No path provide. Please modify the path variable to '
                       'hardcode the %s path.' %split)
    dataset_files = tf.io.matching_files(path)

    # Creating a Dataset that includes only 1/num_shards of data so the data is
    # splitted between different hosts.
    num_hosts, host_id = jax.process_count(), jax.process_index()

    if len(dataset_files) >= num_hosts:
      # Sharding on data sources (e.g. filenames) if there are enough files.
      dataset_files = np.array_split(dataset_files, num_hosts)[host_id]
      dataset = tf.data.TFRecordDataset(
          dataset_files, num_parallel_reads=tf.data.experimental.AUTOTUNE)
    else:
      # Sharding using tf.shard.
      dataset = tf.data.TFRecordDataset(
          dataset_files, num_parallel_reads=tf.data.experimental.AUTOTUNE)
      dataset = dataset.shard(num_shards=num_hosts, index=host_id)

    # pylint: disable=g-long-lambda
    dataset = dataset.map(
        lambda x: pileup_parser(
            x, mode='train', pileup_height=299, pileup_width=299),
        num_parallel_calls=tf.data.experimental.AUTOTUNE)
    dataset = dataset.map(
        pileup_normalizer, num_parallel_calls=tf.data.experimental.AUTOTUNE)

    dataset = dataset.map(
        preprocess, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    return dataset

  # Use different seed for each host.
  if shuffle_seed is None:
    local_seed = None
  else:
    data_seed = 0
    local_seed = data_seed + jax.process_index()

  train_dataset = build_dataset(
      dataset_fn=build_pileup_window_dataset,
      batch_size=batch_size,
      seed=local_seed,
      split='train',
      strategy=None)

  if dataset_service_address:
    if shuffle_seed is not None:
      raise ValueError('Using dataset service with a random seed causes each '
                       'worker to produce exactly the same data. Add '
                       'config.shuffle_seed = None to your config if you '
                       'want to run with dataset service.')
    logging.info('Using the tf.data service at %s', dataset_service_address)
    train_dataset = dataset_utils.distribute(train_dataset,
                                             dataset_service_address)

  eval_dataset = build_dataset(
      dataset_fn=build_pileup_window_dataset,
      split='valid',
      batch_size=eval_batch_size,
      strategy=None)

  shard_batches = functools.partial(dataset_utils.shard, n_devices=num_shards)
  maybe_pad_batches_train = functools.partial(
      dataset_utils.maybe_pad_batch, train=True, batch_size=batch_size)
  maybe_pad_batches_eval = functools.partial(
      dataset_utils.maybe_pad_batch, train=False, batch_size=eval_batch_size)

  train_iter = iter(train_dataset)
  train_iter = map(dataset_utils.tf_to_numpy, train_iter)
  train_iter = map(maybe_pad_batches_train, train_iter)
  train_iter = map(shard_batches, train_iter)
  train_iter = jax_utils.prefetch_to_device(train_iter, prefetch_buffer_size)

  valid_iter = iter(eval_dataset)
  valid_iter = map(dataset_utils.tf_to_numpy, valid_iter)
  valid_iter = map(maybe_pad_batches_eval, valid_iter)
  valid_iter = map(shard_batches, valid_iter)
  valid_iter = jax_utils.prefetch_to_device(valid_iter, prefetch_buffer_size)

  num_classes = 3
  image_size = 256
  input_shape = [-1, image_size, image_size, 7]

  meta_data = {
      'num_classes': num_classes,
      'input_shape': input_shape,
      'num_train_examples': 30_000 * 19,
      'num_eval_examples': 30_000 * 5,
      'input_dtype': getattr(jnp, dtype_str),
      'target_is_onehot': True,
  }

  return dataset_utils.Dataset(train_iter, valid_iter, None, meta_data)
