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
"""Tests for PS mode dynamic embedding optimizer implementations."""

import itertools
import numpy as np
import tensorflow as tf

from tensorflow_recommenders_addons import dynamic_embedding as de
from tensorflow_recommenders_addons.dynamic_embedding.python.ops.ps_embedding_optimizer import (
    is_ps_strategy,
    _normalize_indexed_slices,
    _get_optimizer_type,
    _de_ps_apply_sgd,
    _de_ps_apply_adam,
    _de_ps_apply_adagrad,
    apply_ps_de_update,
)


class IspsStrategyTest(tf.test.TestCase):
  """Test is_ps_strategy detection function."""

  def test_no_strategy(self):
    self.assertFalse(is_ps_strategy())

  def test_none_strategy(self):
    self.assertFalse(is_ps_strategy(None))


class NormalizeIndexedSlicesTest(tf.test.TestCase):
  """Test gradient normalization (deduplication)."""

  def test_unique_indices(self):
    grad = tf.IndexedSlices(values=tf.constant([[1.0, 2.0], [3.0, 4.0]]),
                            indices=tf.constant([0, 1]),
                            dense_shape=tf.constant([5, 2]))
    indices, values = _normalize_indexed_slices(grad)
    self.assertAllEqual(indices, [0, 1])
    self.assertAllClose(values, [[1.0, 2.0], [3.0, 4.0]])

  def test_duplicate_indices(self):
    grad = tf.IndexedSlices(values=tf.constant([[1.0, 2.0], [3.0, 4.0],
                                                [0.5, 0.5]]),
                            indices=tf.constant([0, 1, 0]),
                            dense_shape=tf.constant([5, 2]))
    indices, values = _normalize_indexed_slices(grad)
    # Index 0 should have values summed: [1.0+0.5, 2.0+0.5] = [1.5, 2.5]
    idx_np = indices.numpy()
    val_np = values.numpy()
    pos_0 = np.where(idx_np == 0)[0][0]
    pos_1 = np.where(idx_np == 1)[0][0]
    self.assertAllClose(val_np[pos_0], [1.5, 2.5])
    self.assertAllClose(val_np[pos_1], [3.0, 4.0])

  def test_dense_gradient(self):
    grad = tf.constant([[1.0, 2.0], [3.0, 4.0]])
    indices, values = _normalize_indexed_slices(grad)
    self.assertIsNone(indices)
    self.assertAllClose(values, [[1.0, 2.0], [3.0, 4.0]])


class GetOptimizerTypeTest(tf.test.TestCase):
  """Test optimizer type identification."""

  def test_adam(self):
    try:
      from tensorflow.keras.optimizers.legacy import Adam
    except ImportError:
      from tensorflow.keras.optimizers import Adam
    opt = Adam(0.001)
    self.assertEqual(_get_optimizer_type(opt), 'adam')

  def test_sgd(self):
    try:
      from tensorflow.keras.optimizers.legacy import SGD
    except ImportError:
      from tensorflow.keras.optimizers import SGD
    opt = SGD(0.01)
    self.assertEqual(_get_optimizer_type(opt), 'sgd')

  def test_adagrad(self):
    try:
      from tensorflow.keras.optimizers.legacy import Adagrad
    except ImportError:
      from tensorflow.keras.optimizers import Adagrad
    opt = Adagrad(0.01)
    self.assertEqual(_get_optimizer_type(opt), 'adagrad')


class DePsApplySgdTest(tf.test.TestCase):
  """Test pure-Python SGD update correctness."""

  def test_sgd_no_momentum(self):
    try:
      from tensorflow.keras.optimizers.legacy import SGD
    except ImportError:
      from tensorflow.keras.optimizers import SGD
    opt = SGD(learning_rate=0.1)
    values = tf.constant([[1.0, 2.0], [3.0, 4.0]])
    grads = tf.constant([[0.1, 0.2], [0.3, 0.4]])
    updated, updated_slots = _de_ps_apply_sgd(opt, values, grads, {},
                                              tf.float32)

    # w = w - lr * g = [[1-0.01, 2-0.02], [3-0.03, 4-0.04]]
    expected = values - 0.1 * grads
    self.assertAllClose(updated, expected, atol=1e-6)
    self.assertEqual(len(updated_slots), 0)


class DePsApplyAdamTest(tf.test.TestCase):
  """Test pure-Python Adam update correctness."""

  def test_adam_single_step(self):
    try:
      from tensorflow.keras.optimizers.legacy import Adam
    except ImportError:
      from tensorflow.keras.optimizers import Adam
    lr = 0.001
    beta1 = 0.9
    beta2 = 0.999
    epsilon = 1e-7
    opt = Adam(learning_rate=lr, beta_1=beta1, beta_2=beta2, epsilon=epsilon)
    # Force iteration count to 0 so t=1
    if hasattr(opt, 'iterations'):
      opt.iterations.assign(0)

    dim = 4
    values = tf.constant([[1.0, 2.0, 3.0, 4.0]], dtype=tf.float32)
    grads = tf.constant([[0.1, 0.2, 0.3, 0.4]], dtype=tf.float32)
    m_init = tf.zeros_like(values)
    v_init = tf.zeros_like(values)

    updated, updated_slots = _de_ps_apply_adam(opt, values, grads, {
        'm': m_init,
        'v': v_init
    }, tf.float32)

    # Manual computation
    t = 1.0
    new_m = beta1 * 0 + (1 - beta1) * grads.numpy()
    new_v = beta2 * 0 + (1 - beta2) * grads.numpy()**2
    m_hat = new_m / (1 - beta1**t)
    v_hat = new_v / (1 - beta2**t)
    expected = values.numpy() - lr * m_hat / (np.sqrt(v_hat) + epsilon)

    self.assertAllClose(updated, expected, atol=1e-5)
    self.assertAllClose(updated_slots['m'], new_m, atol=1e-6)
    self.assertAllClose(updated_slots['v'], new_v, atol=1e-6)


class DePsApplyAdagradTest(tf.test.TestCase):
  """Test pure-Python Adagrad update correctness."""

  def test_adagrad_single_step(self):
    try:
      from tensorflow.keras.optimizers.legacy import Adagrad
    except ImportError:
      from tensorflow.keras.optimizers import Adagrad
    lr = 0.1
    epsilon = 1e-7
    opt = Adagrad(learning_rate=lr, epsilon=epsilon)

    values = tf.constant([[1.0, 2.0], [3.0, 4.0]], dtype=tf.float32)
    grads = tf.constant([[0.5, 0.5], [1.0, 1.0]], dtype=tf.float32)
    accum_init = tf.zeros_like(values)

    updated, updated_slots = _de_ps_apply_adagrad(opt, values, grads,
                                                  {'accumulator': accum_init},
                                                  tf.float32)

    # Manual computation
    new_accum = 0 + grads.numpy()**2
    expected = values.numpy() - lr * grads.numpy() / (np.sqrt(new_accum) +
                                                      epsilon)

    self.assertAllClose(updated, expected, atol=1e-6)
    self.assertAllClose(updated_slots['accumulator'], new_accum, atol=1e-6)


class ApplyPsDeUpdateTest(tf.test.TestCase):
  """Test the end-to-end PS DE update function with real de.Variable."""

  def test_sgd_update_e2e(self):
    """Test SGD update through the full apply_ps_de_update path."""
    try:
      from tensorflow.keras.optimizers.legacy import SGD
    except ImportError:
      from tensorflow.keras.optimizers import SGD
    opt = SGD(learning_rate=0.1)

    # Create a dynamic embedding variable
    de_var = de.get_variable(
        name="test_sgd_e2e",
        key_dtype=tf.int64,
        value_dtype=tf.float32,
        dim=4,
        initializer=tf.keras.initializers.Ones(),
    )

    # Insert some initial values
    ids = tf.constant([0, 1, 2], dtype=tf.int64)
    init_values = tf.constant([[1.0, 1.0, 1.0, 1.0], [2.0, 2.0, 2.0, 2.0],
                               [3.0, 3.0, 3.0, 3.0]])
    de_var.upsert(ids, init_values)

    # Create gradient (IndexedSlices)
    grad = tf.IndexedSlices(values=tf.constant([[0.1, 0.1, 0.1, 0.1],
                                                [0.2, 0.2, 0.2, 0.2],
                                                [0.3, 0.3, 0.3, 0.3]]),
                            indices=tf.constant([0, 1, 2], dtype=tf.int64),
                            dense_shape=tf.constant([3, 4], dtype=tf.int64))

    # Apply update
    apply_ps_de_update(
        optimizer=opt,
        de_var=de_var,
        grad=grad,
        slot_de_vars={},
        bp_v2=False,
    )

    # Check results
    result = de_var.lookup(ids)
    expected = init_values - 0.1 * grad.values
    self.assertAllClose(result, expected, atol=1e-5)

  def test_adam_update_e2e(self):
    """Test Adam update through the full apply_ps_de_update path."""
    try:
      from tensorflow.keras.optimizers.legacy import Adam
    except ImportError:
      from tensorflow.keras.optimizers import Adam
    opt = Adam(learning_rate=0.001)
    if hasattr(opt, 'iterations'):
      opt.iterations.assign(0)

    # Create main de.Variable
    de_var = de.get_variable(
        name="test_adam_e2e",
        key_dtype=tf.int64,
        value_dtype=tf.float32,
        dim=4,
        initializer=tf.keras.initializers.Ones(),
    )

    # Create slot de.Variables (m and v)
    slot_m = de.get_variable(
        name="test_adam_e2e/Adam/m",
        key_dtype=tf.int64,
        value_dtype=tf.float32,
        dim=4,
        initializer=0.0,
        trainable=False,
    )
    slot_v = de.get_variable(
        name="test_adam_e2e/Adam/v",
        key_dtype=tf.int64,
        value_dtype=tf.float32,
        dim=4,
        initializer=0.0,
        trainable=False,
    )

    # Insert initial values
    ids = tf.constant([10, 20], dtype=tf.int64)
    init_values = tf.constant([[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]])
    de_var.upsert(ids, init_values)

    # Initialize slots to zero
    slot_m.upsert(ids, tf.zeros_like(init_values))
    slot_v.upsert(ids, tf.zeros_like(init_values))

    # Create gradient
    grad = tf.IndexedSlices(values=tf.constant([[0.1, 0.2, 0.3, 0.4],
                                                [0.5, 0.6, 0.7, 0.8]]),
                            indices=tf.constant([10, 20], dtype=tf.int64),
                            dense_shape=tf.constant([100, 4], dtype=tf.int64))

    # Apply update
    apply_ps_de_update(
        optimizer=opt,
        de_var=de_var,
        grad=grad,
        slot_de_vars={
            'm': slot_m,
            'v': slot_v
        },
        bp_v2=False,
    )

    # Verify m and v slots were updated
    m_result = slot_m.lookup(ids)
    v_result = slot_v.lookup(ids)

    # m should not be zero anymore
    self.assertNotAllClose(m_result, tf.zeros_like(init_values))
    # v should not be zero anymore
    self.assertNotAllClose(v_result, tf.zeros_like(init_values))

    # Main values should have been updated
    result = de_var.lookup(ids)
    self.assertNotAllClose(result, init_values)

  def test_duplicate_indices(self):
    """Test that duplicate gradient indices are handled correctly."""
    try:
      from tensorflow.keras.optimizers.legacy import SGD
    except ImportError:
      from tensorflow.keras.optimizers import SGD
    opt = SGD(learning_rate=0.1)

    de_var = de.get_variable(
        name="test_dup_idx",
        key_dtype=tf.int64,
        value_dtype=tf.float32,
        dim=2,
        initializer=tf.keras.initializers.Ones(),
    )

    ids = tf.constant([0, 1], dtype=tf.int64)
    init_values = tf.constant([[1.0, 1.0], [2.0, 2.0]])
    de_var.upsert(ids, init_values)

    # Gradient with duplicate index 0
    grad = tf.IndexedSlices(values=tf.constant([[0.1, 0.1], [0.2, 0.2],
                                                [0.3, 0.3]]),
                            indices=tf.constant([0, 1, 0], dtype=tf.int64),
                            dense_shape=tf.constant([2, 2], dtype=tf.int64))

    apply_ps_de_update(
        optimizer=opt,
        de_var=de_var,
        grad=grad,
        slot_de_vars={},
        bp_v2=False,
    )

    result = de_var.lookup(ids)
    # Index 0: sum_grad = 0.1+0.3 = 0.4, value = 1.0 - 0.1*0.4 = 0.96
    # Index 1: grad = 0.2, value = 2.0 - 0.1*0.2 = 1.98
    self.assertAllClose(result[0], [0.96, 0.96], atol=1e-5)
    self.assertAllClose(result[1], [1.98, 1.98], atol=1e-5)


if __name__ == "__main__":
  tf.test.main()
