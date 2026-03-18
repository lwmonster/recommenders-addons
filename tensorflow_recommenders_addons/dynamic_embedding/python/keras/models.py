# Copyright 2023 The TensorFlow Recommenders-Addons Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# lint-as: python3

import functools
import os.path

from tensorflow_recommenders_addons import dynamic_embedding as de

try:
  from keras.saving.saved_model import save as keras_saved_model_save
except:
  keras_saved_model_save = None
from tensorflow.python.keras.saving.saved_model import save as tf_saved_model_save
from tensorflow.python.ops import array_ops
from tensorflow.python.platform import tf_logging
from tensorflow.python.saved_model.save_options import SaveOptions

tf_original_save_func = tf_saved_model_save.save
if keras_saved_model_save is not None:
  keras_original_save_func = keras_saved_model_save.save


def _de_keras_save_func(original_save_func,
                        model,
                        filepath,
                        overwrite,
                        include_optimizer,
                        signatures=None,
                        options=None,
                        save_traces=True,
                        *args,
                        **kwargs):
  """Overwrite TF Keras save function
    Calling the TF save API for all ranks causes file conflicts, 
    so KV files other than rank0 need to be saved by calling the underlying API separately.
    This is a convenience function for saving HvdAllToAllEmbedding to KV files in different rank.
  """
  try:
    import horovod.tensorflow as hvd
    try:
      hvd.rank()
    except:
      hvd = None
  except:
    hvd = None

  if hvd is not None:
    filepath = hvd.broadcast_object(filepath,
                                    root_rank=0,
                                    name='de_hvd_broadcast_filepath')

  call_original_save_func = functools.partial(
      original_save_func,
      model=model,
      filepath=filepath,
      overwrite=overwrite,
      include_optimizer=include_optimizer,
      signatures=signatures,
      options=options,
      save_traces=save_traces,
      *args,
      **kwargs)

  de_dir = os.path.join(filepath, "variables", "TFRADynamicEmbedding")

  def _check_saveable_and_redirect_new_de_dir(hvd_rank=0):
    for var in model.variables:
      if not hasattr(var, "params"):
        continue
      if not hasattr(var.params, "_created_in_class"):
        continue
      de_var = var.params
      a2a_emb = de_var._created_in_class
      if issubclass(a2a_emb.__class__, de.keras.layers.HvdAllToAllEmbedding):
        if de_var._saveable_object_creator is None:
          if hvd_rank == 0:
            tf_logging.warning(
                "Please use FileSystemSaver when use HvdAllToAllEmbedding. "
                "It will allow TFRA load KV files when Embedding tensor parallel. "
                f"The embedding shards at each horovod rank are now temporarily stored in {de_dir}"
            )
      if not isinstance(de_var.kv_creator.saver, de.FileSystemSaver):
        # This function only serves FileSystemSaver.
        continue
      # Redirect new de_dir
      if hasattr(de_var, 'saveable'):
        de_var.saveable._saver_config.save_path = de_dir

  def _traverse_emb_layers_and_save(proc_size=1, proc_rank=0):
    for var in model.variables:
      if not hasattr(var, "params"):
        continue
      if not hasattr(var.params, "_created_in_class"):
        continue
      de_var = var.params
      a2a_emb = de_var._created_in_class
      if de_var._saveable_object_creator is not None:
        if not isinstance(de_var.kv_creator.saver, de.FileSystemSaver):
          # This function only serves FileSystemSaver.
          continue
        # save optimizer parameters of Dynamic Embedding
        if include_optimizer is True:
          de_opt_vars = a2a_emb.optimizer_vars.as_list() if hasattr(
              a2a_emb.optimizer_vars, "as_list") else a2a_emb.optimizer_vars
          for de_opt_var in de_opt_vars:
            de_opt_var.save_to_file_system(dirpath=de_dir,
                                           proc_size=proc_size,
                                           proc_rank=proc_rank)
        if proc_rank == 0:
          # FileSystemSaver works well at rank 0.
          continue
        # save Dynamic Embedding Parameters
        de_var.save_to_file_system(dirpath=de_dir,
                                   proc_size=proc_size,
                                   proc_rank=proc_rank)

  if hvd is None:
    call_original_save_func()
    _traverse_emb_layers_and_save()
  else:
    _check_saveable_and_redirect_new_de_dir(hvd.rank())
    if hvd.rank() == 0:
      call_original_save_func()
    _traverse_emb_layers_and_save(hvd.size(), hvd.rank())
    hvd.join()  # Sync for avoiding rank conflict


def de_hvd_save_model(model,
                      filepath,
                      overwrite=True,
                      include_optimizer=True,
                      signatures=None,
                      options=None,
                      save_traces=True,
                      *args,
                      **kwargs):
  return de_save_model(model=model,
                       filepath=filepath,
                       overwrite=True,
                       include_optimizer=True,
                       signatures=None,
                       options=None,
                       save_traces=True,
                       *args,
                       **kwargs)


def de_save_model(model,
                  filepath,
                  overwrite=True,
                  include_optimizer=True,
                  signatures=None,
                  options=None,
                  save_traces=True,
                  *args,
                  **kwargs):
  if keras_saved_model_save is not None:
    _save_handle = functools.partial(_de_keras_save_func,
                                     keras_original_save_func)
  else:
    _save_handle = functools.partial(_de_keras_save_func, tf_original_save_func)
  if options is None:
    options = SaveOptions(namespace_whitelist=['TFRA'])
  elif isinstance(options, SaveOptions) and hasattr(options,
                                                    'namespace_whitelist'):
    options.namespace_whitelist.append('TFRA')

  return _save_handle(model,
                      filepath,
                      overwrite,
                      include_optimizer,
                      signatures=signatures,
                      options=options,
                      save_traces=save_traces,
                      *args,
                      **kwargs)


def fit_ps(model,
           dataset_fn,
           strategy,
           epochs=1,
           steps_per_epoch=None,
           callbacks=None,
           verbose=1):
  """PS mode training function for models with dynamic embedding layers.

  Creates N independent ShadowVariables (one per worker) and dispatches
  N step functions via ClusterCoordinator with round-robin scheduling,
  avoiding the shared ShadowVariable race condition in multi-worker PS mode.

  Args:
    model: A compiled tf.keras.Model containing de.keras.layers.Embedding
      or SquashedEmbedding layers. Each DE layer must have `input_key` set,
      and the model must implement `call_with_embeddings(inputs, embeddings,
      training)`.
    dataset_fn: A callable that returns a tf.data.Dataset. This function
      will be executed on each Worker to create the dataset locally.
      Signature: dataset_fn() -> tf.data.Dataset
    strategy: The ParameterServerStrategy instance.
    epochs: Number of training epochs.
    steps_per_epoch: Number of steps per epoch.
    callbacks: List of Keras callbacks (optional).
    verbose: Verbosity mode. 0 = silent, 1 = progress.

  Returns:
    A dict of training history {metric_name: [values_per_epoch]}.
  """
  import time
  import tensorflow as tf
  from tensorflow.python.framework import ops
  from tensorflow.python.ops import math_ops
  from tensorflow_recommenders_addons.dynamic_embedding.python.ops.ps_embedding_optimizer import (
      is_ps_strategy, apply_ps_de_update)

  coordinator = (
      tf.distribute.experimental.coordinator.ClusterCoordinator(strategy))
  num_workers = strategy._num_workers

  de_layers = []
  for layer in model._flatten_layers():
    if isinstance(
        layer, (de.keras.layers.Embedding, de.keras.layers.SquashedEmbedding)):
      de_layers.append(layer)

  if not de_layers:
    raise ValueError("fit_ps requires model to contain at least one "
                     "de.keras.layers.Embedding or SquashedEmbedding layer.")

  for layer in de_layers:
    if not hasattr(layer, 'input_key') or layer.input_key is None:
      raise ValueError(
          "DE layer '{}' must have input_key set for fit_ps. "
          "Use: de.keras.layers.SquashedEmbedding(..., input_key='feature_name')"
          .format(layer.name))

  if not hasattr(model, 'call_with_embeddings'):
    raise ValueError(
        "Model must implement call_with_embeddings(self, inputs, embeddings, "
        "training) for fit_ps. This method receives pre-computed embeddings "
        "and runs the DNN part of the model.")

  N = num_workers * 2  # 2x workers for safety margin on join frequency
  per_shadow = []
  for sid in range(N):
    group = {}
    for layer in de_layers:
      shadow = de.shadow_ops.ShadowVariable(layer.params,
                                            name="{}_ps_shadow_{}".format(
                                                layer.name, sid),
                                            trainable=True)
      group[layer.name] = shadow
    per_shadow.append(group)

  optimizer = model.optimizer
  compiled_loss = model.compiled_loss
  compiled_metrics = model.compiled_metrics

  lr = optimizer.learning_rate

  dnn_vars = [
      v for v in model.trainable_variables
      if not isinstance(v, de.TrainableWrapper)
  ]

  def make_step_fn(shadow_group):

    @tf.function
    def step_fn(iterator):
      data = next(iterator)
      if isinstance(data, tuple) and len(data) >= 2:
        x, y = data[0], data[1]
      else:
        x, y = data, None

      with tf.GradientTape() as tape:
        embeddings = {}
        shadow_list = []
        unique_info = {}
        for layer_name, shadow in shadow_group.items():
          layer = model.get_layer(layer_name)
          ids = x[layer.input_key]
          ids_flat = tf.reshape(ids, [-1])

          if layer.with_unique:
            unique_ids, idx = tf.unique(ids_flat)
            with ops.control_dependencies([shadow._reset_ids(unique_ids)]):
              unique_emb = shadow.read_value(do_prefetch=True)
            emb = tf.gather(unique_emb, idx)
            unique_info[layer_name] = (unique_ids, idx)
          else:
            with ops.control_dependencies([shadow._reset_ids(ids_flat)]):
              emb = shadow.read_value(do_prefetch=True)
            unique_info[layer_name] = None

          if hasattr(layer, 'reduce_pooling'):
            emb_reshaped = tf.reshape(
                emb, tf.concat([tf.shape(ids), [layer.embedding_size]], axis=0))
            emb = layer.reduce_pooling(emb_reshaped)
          emb = tf.ensure_shape(emb, [None, layer.embedding_size])
          embeddings[layer_name] = emb
          shadow_list.append(shadow)

        y_pred = model.call_with_embeddings(x, embeddings, training=True)
        loss = compiled_loss(y, y_pred)

      all_vars = shadow_list + dnn_vars
      grads = tape.gradient(loss, all_vars)

      n_de = len(shadow_list)
      de_grads = grads[:n_de]
      dnn_grads_list = grads[n_de:]

      for i, (layer_name, shadow) in enumerate(shadow_group.items()):
        grad = de_grads[i]
        if grad is not None:
          info = unique_info[layer_name]
          if info is not None:
            unique_ids, idx = info
            if isinstance(grad, tf.IndexedSlices):
              n_unique = tf.shape(unique_ids)[0]
              agg_values = tf.math.unsorted_segment_sum(grad.values,
                                                        grad.indices, n_unique)
            else:
              agg_values = grad
            grad = tf.IndexedSlices(values=agg_values,
                                    indices=unique_ids,
                                    dense_shape=None)
          else:
            if not isinstance(grad, tf.IndexedSlices):
              grad = tf.IndexedSlices(values=grad,
                                      indices=shadow.ids,
                                      dense_shape=None)
          apply_ps_de_update(optimizer=optimizer,
                             de_var=shadow.params,
                             grad=grad,
                             slot_de_vars={},
                             bp_v2=shadow.params.bp_v2)

      for g, v in zip(dnn_grads_list, dnn_vars):
        if g is not None:
          v.assign_sub(math_ops.cast(lr, g.dtype) * g)

      if compiled_metrics:
        compiled_metrics.update_state(y, y_pred)

    return step_fn

  step_fns = [make_step_fn(per_shadow[sid]) for sid in range(N)]

  callback_list = tf.keras.callbacks.CallbackList(callbacks,
                                                  add_history=True,
                                                  add_progbar=False,
                                                  model=model)
  callback_list.on_train_begin()

  dist_dataset = coordinator.create_per_worker_dataset(dataset_fn)

  history = {}
  for epoch in range(epochs):
    callback_list.on_epoch_begin(epoch)
    epoch_start = time.time()

    for m in model.metrics:
      m.reset_states()

    iterator = iter(dist_dataset)
    for step in range(steps_per_epoch):
      fn_idx = step % N
      coordinator.schedule(step_fns[fn_idx], args=(iterator,))
      # Every N steps, wait for all pending tasks to complete
      # to ensure no two Workers use the same shadow concurrently
      if (step + 1) % N == 0:
        coordinator.join()
    coordinator.join()

    epoch_time = time.time() - epoch_start
    logs = {}
    for m in model.metrics:
      val = m.result()
      if hasattr(val, 'numpy'):
        val = val.numpy()
      logs[m.name] = float(val)
      history.setdefault(m.name, []).append(float(val))

    if verbose:
      metrics_str = ' - '.join(
          ['{}: {:.4f}'.format(k, v) for k, v in logs.items()])
      tf_logging.info('Epoch {}/{} - {:.0f}s - {}'.format(
          epoch + 1, epochs, epoch_time, metrics_str))
      print('Epoch {}/{} - {:.0f}s - {}'.format(epoch + 1, epochs, epoch_time,
                                                metrics_str))

    callback_list.on_epoch_end(epoch, logs)

  callback_list.on_train_end()
  return history
