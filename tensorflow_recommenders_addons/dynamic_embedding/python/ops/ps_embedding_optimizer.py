# Copyright 2024 The TensorFlow Recommenders-Addons Authors.
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
"""PS mode dynamic embedding optimizer update implementations.

This module provides pure-Python optimizer update logic for dynamic embedding
variables under ParameterServerStrategy. It bypasses TF native optimizer ops
(ResourceScatterAdd, AssignSubVariableOp) which assume static variable shapes,
and instead directly operates on dynamic hash tables via lookup/upsert.

Supported optimizers: SGD (with/without momentum), Adam, Adagrad.
"""

from packaging import version

import tensorflow as tf

from tensorflow import version as tf_version

if version.parse(tf_version.VERSION) >= version.parse("2.14"):
  from tensorflow.python.distribute import distribute_lib as distribute_ctx
else:
  from tensorflow.python.distribute import distribution_strategy_context as distribute_ctx
from tensorflow.python.distribute import parameter_server_strategy
from tensorflow.python.distribute import parameter_server_strategy_v2
if version.parse(tf_version.VERSION) >= version.parse("2.13"):
  from tensorflow.python.framework.indexed_slices import IndexedSlices
else:
  from tensorflow.python.framework.ops import IndexedSlices
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.platform import tf_logging


def is_ps_strategy(strategy=None):
  """Check whether the current or given strategy is ParameterServerStrategy.

  Args:
    strategy: A `tf.distribute.Strategy` instance. If None, uses the current
      strategy from the context.

  Returns:
    Boolean indicating whether PS strategy is active.
  """
  if strategy is None:
    if not distribute_ctx.has_strategy():
      return False
    strategy = distribute_ctx.get_strategy()

  return isinstance(
      strategy,
      (
          parameter_server_strategy.ParameterServerStrategyV1,
          parameter_server_strategy_v2.ParameterServerStrategyV2,
      ),
  )


def _normalize_indexed_slices(grad):
  """Deduplicate and aggregate IndexedSlices gradients.

  When multiple gradient entries share the same index, their values are
  summed together. This is necessary to avoid duplicate updates to the
  same key in the hash table.

  Args:
    grad: An `IndexedSlices` or dense `Tensor` gradient.

  Returns:
    A tuple of (indices, values) where indices are unique.
    For dense gradients, returns (None, grad).
  """
  if isinstance(grad, IndexedSlices):
    unique_indices, idx = tf.unique(grad.indices)
    num_unique = tf.shape(unique_indices)[0]
    unique_values = math_ops.unsorted_segment_sum(grad.values, idx, num_unique)
    return unique_indices, unique_values
  else:
    return None, grad


def _get_optimizer_type(optimizer):
  """Identify the optimizer type from its class hierarchy.

  Args:
    optimizer: A TF optimizer instance.

  Returns:
    A string identifier: 'sgd', 'adam', 'adagrad', or 'unknown'.
  """
  cls_name = type(optimizer).__name__.lower()

  # Check common optimizer class names
  if 'adam' in cls_name and 'amsgrad' not in cls_name:
    return 'adam'
  if 'sgd' in cls_name:
    return 'sgd'
  if 'adagrad' in cls_name:
    return 'adagrad'

  # Check by slot names as fallback
  if hasattr(optimizer, 'get_slot_names'):
    slot_names = set(optimizer.get_slot_names())
    if slot_names == {'m', 'v'} or slot_names == {'m', 'v', 'vhat'}:
      return 'adam'
    if slot_names == {'momentum'}:
      return 'sgd'
    if slot_names == {'accumulator'}:
      return 'adagrad'
    if len(slot_names) == 0:
      return 'sgd'

  return 'unknown'


def _get_lr(optimizer):
  """Get the current learning rate from an optimizer, handling various types.

  Args:
    optimizer: A TF optimizer instance.

  Returns:
    A scalar Tensor or float representing the learning rate.
  """
  lr = None
  for attr in ('learning_rate', 'lr', '_learning_rate', '_lr'):
    if hasattr(optimizer, attr):
      lr = getattr(optimizer, attr)
      break

  if lr is None:
    raise ValueError("Cannot determine learning rate from optimizer "
                     "{}".format(type(optimizer).__name__))

  if callable(lr) and not isinstance(lr, tf.Tensor):
    lr = lr()
  return lr


def _get_hyper(optimizer, name, dtype=None):
  """Safely get a hyperparameter from an optimizer.

  Tries multiple access patterns to accommodate different optimizer versions
  (TF1 train.*, TF2 keras legacy, TF2 keras experimental).

  Args:
    optimizer: A TF optimizer instance.
    name: Hyperparameter name string.
    dtype: Optional dtype to cast to.

  Returns:
    The hyperparameter value as a Tensor, or None if not found.
  """
  val = None

  # Try direct attribute access
  if hasattr(optimizer, name):
    val = getattr(optimizer, name)
  elif hasattr(optimizer, '_' + name):
    val = getattr(optimizer, '_' + name)
  # Try _get_hyper method (Keras optimizer v2)
  elif hasattr(optimizer, '_get_hyper'):
    try:
      val = optimizer._get_hyper(name, dtype or tf.float32)
    except (KeyError, ValueError):
      pass

  if val is not None and dtype is not None:
    if isinstance(val, tf.Tensor):
      val = math_ops.cast(val, dtype)
    elif isinstance(val, tf.Variable):
      val = math_ops.cast(val.read_value(), dtype)
    elif hasattr(val, 'read_value'):
      # Handle distributed/PS variables
      val = math_ops.cast(val.read_value(), dtype)
    else:
      try:
        val = tf.constant(val, dtype=dtype)
      except (TypeError, NotImplementedError):
        val = math_ops.cast(tf.identity(val), dtype)

  return val


def _de_ps_apply_sgd(optimizer, values, grads, slot_values, dtype):
  """Pure-Python SGD update for PS mode.

  Implements:
    Without momentum: w = w - lr * g
    With momentum:    m = mu * m + g
                      w = w - lr * m
    With Nesterov:    w = w - lr * (mu * m_new + g)

  Args:
    optimizer: TF optimizer instance.
    values: Current parameter values, shape [num_ids, dim].
    grads: Gradient values, shape [num_ids, dim].
    slot_values: Dict mapping slot name to current slot tensor values.
    dtype: The value dtype.

  Returns:
    Tuple of (updated_values, updated_slot_values_dict).
  """
  lr = math_ops.cast(_get_lr(optimizer), dtype)

  momentum_coeff = _get_hyper(optimizer, 'momentum', dtype)
  use_nesterov = getattr(optimizer, 'nesterov', False)
  if not isinstance(use_nesterov, bool):
    use_nesterov = bool(use_nesterov)

  if momentum_coeff is not None and 'momentum' in slot_values:
    m = slot_values['momentum']
    new_m = momentum_coeff * m + grads
    if use_nesterov:
      updated_values = values - lr * (momentum_coeff * new_m + grads)
    else:
      updated_values = values - lr * new_m
    return updated_values, {'momentum': new_m}
  else:
    updated_values = values - lr * grads
    return updated_values, {}


def _de_ps_apply_adam(optimizer, values, grads, slot_values, dtype):
  """Pure-Python Adam update for PS mode.

  Implements standard Adam:
    m = beta1 * m + (1 - beta1) * g
    v = beta2 * v + (1 - beta2) * g^2
    m_hat = m / (1 - beta1^t)
    v_hat = v / (1 - beta2^t)
    w = w - lr * m_hat / (sqrt(v_hat) + epsilon)

  Args:
    optimizer: TF optimizer instance.
    values: Current parameter values, shape [num_ids, dim].
    grads: Gradient values, shape [num_ids, dim].
    slot_values: Dict mapping slot name to current slot tensor values.
    dtype: The value dtype.

  Returns:
    Tuple of (updated_values, updated_slot_values_dict).
  """
  lr = math_ops.cast(_get_lr(optimizer), dtype)

  beta_1 = _get_hyper(optimizer, 'beta_1', dtype)
  if beta_1 is None:
    beta_1 = _get_hyper(optimizer, 'beta1', dtype)
  if beta_1 is None:
    beta_1 = tf.constant(0.9, dtype=dtype)

  beta_2 = _get_hyper(optimizer, 'beta_2', dtype)
  if beta_2 is None:
    beta_2 = _get_hyper(optimizer, 'beta2', dtype)
  if beta_2 is None:
    beta_2 = tf.constant(0.999, dtype=dtype)

  epsilon = _get_hyper(optimizer, 'epsilon', dtype)
  if epsilon is None:
    epsilon = tf.constant(1e-7, dtype=dtype)

  # Get current iteration count
  if hasattr(optimizer, 'iterations'):
    t = math_ops.cast(optimizer.iterations + 1, dtype)
  elif hasattr(optimizer, '_iterations'):
    t = math_ops.cast(optimizer._iterations + 1, dtype)
  else:
    t = tf.constant(1.0, dtype=dtype)

  m = slot_values.get('m', tf.zeros_like(values))
  v = slot_values.get('v', tf.zeros_like(values))

  new_m = beta_1 * m + (1.0 - beta_1) * grads
  new_v = beta_2 * v + (1.0 - beta_2) * math_ops.square(grads)

  m_hat = new_m / (1.0 - math_ops.pow(beta_1, t))
  v_hat = new_v / (1.0 - math_ops.pow(beta_2, t))

  updated_values = values - lr * m_hat / (math_ops.sqrt(v_hat) + epsilon)

  updated_slots = {'m': new_m, 'v': new_v}

  # Handle amsgrad if applicable
  if 'vhat' in slot_values:
    vhat = slot_values['vhat']
    new_vhat = math_ops.maximum(vhat, new_v)
    updated_values = values - lr * m_hat / (math_ops.sqrt(new_vhat) + epsilon)
    updated_slots['vhat'] = new_vhat

  return updated_values, updated_slots


def _de_ps_apply_adagrad(optimizer, values, grads, slot_values, dtype):
  """Pure-Python Adagrad update for PS mode.

  Implements:
    accum = accum + g^2
    w = w - lr * g / (sqrt(accum) + epsilon)

  Args:
    optimizer: TF optimizer instance.
    values: Current parameter values, shape [num_ids, dim].
    grads: Gradient values, shape [num_ids, dim].
    slot_values: Dict mapping slot name to current slot tensor values.
    dtype: The value dtype.

  Returns:
    Tuple of (updated_values, updated_slot_values_dict).
  """
  lr = math_ops.cast(_get_lr(optimizer), dtype)

  epsilon = _get_hyper(optimizer, 'epsilon', dtype)
  if epsilon is None:
    epsilon = tf.constant(1e-7, dtype=dtype)

  accum = slot_values.get('accumulator', tf.zeros_like(values))
  new_accum = accum + math_ops.square(grads)

  updated_values = values - lr * grads / (math_ops.sqrt(new_accum) + epsilon)
  return updated_values, {'accumulator': new_accum}


_OPTIMIZER_APPLY_FNS = {
    'sgd': _de_ps_apply_sgd,
    'adam': _de_ps_apply_adam,
    'adagrad': _de_ps_apply_adagrad,
}


def apply_ps_de_update(optimizer, de_var, grad, slot_de_vars, bp_v2=False):
  """Apply gradient update to a dynamic embedding variable in PS mode.

  This function bypasses TF native optimizer ops (which assume static shapes)
  and instead performs the update entirely through hash table operations:
    1. Normalize gradients (deduplicate IndexedSlices)
    2. Lookup current values from main table and slot tables
    3. Compute updated values in pure Python
    4. Upsert results back to hash tables

  Args:
    optimizer: The TF optimizer instance (Adam, SGD, etc.).
    de_var: The `de.Variable` (main embedding hash table).
    grad: The gradient, either `IndexedSlices` or dense `Tensor`.
    slot_de_vars: Dict mapping slot name (e.g., 'm', 'v') to `de.Variable`
      for each optimizer slot.
    bp_v2: If True, use accumulation mode (accum) instead of overwrite (upsert)
      for writing back to the hash table. This helps with stale gradient issues
      in asynchronous distributed training.

  Returns:
    An op group that performs the complete update.

  Raises:
    NotImplementedError: If the optimizer type is not supported.
  """
  optimizer_type = _get_optimizer_type(optimizer)
  if optimizer_type == 'unknown' or optimizer_type not in _OPTIMIZER_APPLY_FNS:
    raise NotImplementedError(
        "Optimizer '{}' (type='{}') is not yet supported in PS mode "
        "with dynamic embedding. Currently supported optimizers: {}. "
        "Please file a feature request if you need support for this "
        "optimizer.".format(
            type(optimizer).__name__, optimizer_type,
            list(_OPTIMIZER_APPLY_FNS.keys())))

  dtype = de_var.value_dtype

  # Step 1: Normalize gradients (deduplicate IndexedSlices)
  indices, grad_values = _normalize_indexed_slices(grad)

  if indices is None:
    # Dense gradient can happen when PS strategy reduces gradients.
    # In this case, the gradient corresponds to all active IDs in the
    # TrainableWrapper/ShadowVariable. We need the caller to provide
    # the active IDs. For now, return no_op with a warning since
    # we handle this in the optimizer integration layer.
    tf_logging.warning(
        "Dense gradient received for dynamic embedding variable '{}' "
        "in PS mode. This is unusual and may indicate a problem.".format(
            de_var.name))
    return control_flow_ops.no_op()

  grad_values = math_ops.cast(grad_values, dtype)

  # Ensure indices dtype matches the de.Variable key_dtype
  indices = math_ops.cast(indices, de_var.key_dtype)

  # Step 2: Lookup current values from main table and all slot tables
  # With bp_v2, also get exists flags for accum mode
  if bp_v2:
    current_values, main_exists = de_var.lookup(indices, return_exists=True)
  else:
    current_values = de_var.lookup(indices)
    main_exists = None

  current_slots = {}
  slot_exists = {}
  for slot_name, slot_var in slot_de_vars.items():
    if bp_v2:
      sv, se = slot_var.lookup(indices, return_exists=True)
      current_slots[slot_name] = sv
      slot_exists[slot_name] = se
    else:
      current_slots[slot_name] = slot_var.lookup(indices)

  # Step 3: Compute updates using pure-Python optimizer math
  apply_fn = _OPTIMIZER_APPLY_FNS[optimizer_type]
  updated_values, updated_slots = apply_fn(optimizer, current_values,
                                           grad_values, current_slots, dtype)

  # Step 4: Write back to hash tables (slots first, then main table)
  # bp_v2=True: use accum (delta accumulation) to avoid stale gradient
  #   overwrites when multiple workers update the same key concurrently
  # bp_v2=False: use upsert (full overwrite), only safe for single-worker
  update_ops = []
  for slot_name, slot_var in slot_de_vars.items():
    if slot_name in updated_slots:
      if bp_v2:
        update_ops.append(
            slot_var.accum(indices, current_slots[slot_name],
                           updated_slots[slot_name], slot_exists[slot_name]))
      else:
        update_ops.append(slot_var.upsert(indices, updated_slots[slot_name]))

  with tf.control_dependencies(update_ops):
    if bp_v2:
      main_update = de_var.accum(indices, current_values, updated_values,
                                 main_exists)
    else:
      main_update = de_var.upsert(indices, updated_values)

  if de_var.restrict_policy is not None:
    with tf.control_dependencies([main_update]):
      restrict_op = de_var.restrict_policy.apply_update(indices)
      return restrict_op

  return main_update
