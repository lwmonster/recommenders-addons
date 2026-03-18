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
"""Integration tests for PS mode dynamic embedding training.

These tests verify that dynamic embedding variables work correctly
under ParameterServerStrategy, which was previously broken due to
shape mismatch errors (ResourceScatterAdd / AssignSubVariableOp).
"""

import numpy as np
import tensorflow as tf

from tensorflow.python.eager import context
from tensorflow.python.framework import config as tf_config
from tensorflow.python.platform import test
from tensorflow.core.protobuf import config_pb2
from tensorflow.python.training import server_lib

from tensorflow_recommenders_addons import dynamic_embedding as de

default_cluster_config = config_pb2.ConfigProto(allow_soft_placement=False)


def _create_ps_and_worker_servers(spec):
  """Create PS and Worker servers from a ClusterSpec."""
  ps_list, worker_list = [], []
  for job_name, ip_port_list in spec.as_dict().items():
    for i, v in enumerate(ip_port_list):
      node = server_lib.Server(spec,
                               job_name=job_name,
                               task_index=i,
                               config=default_cluster_config)
      if job_name == 'ps':
        ps_list.append(node)
      elif job_name == 'worker':
        worker_list.append(node)
      else:
        raise TypeError(
            'Expecting ps or worker in cluster_spec, but get {}'.format(
                job_name))
  return ps_list, worker_list


class PSStrategyDynamicEmbeddingTest(test.TestCase):
  """Test dynamic embedding training under ParameterServerStrategy."""

  @classmethod
  def setUpClass(cls):
    """Set up PS and Worker servers once for all tests."""
    if not context.executing_eagerly():
      return

    cls.cluster_spec = tf.train.ClusterSpec({
        'ps': ['localhost:3320', 'localhost:3321'],
        'worker': ['localhost:3322', 'localhost:3323']
    })
    cls.ps_list, cls.worker_list = _create_ps_and_worker_servers(
        cls.cluster_spec)
    cls.resolver = tf.distribute.cluster_resolver.SimpleClusterResolver(
        cls.cluster_spec)
    cls.strategy = tf.distribute.experimental.ParameterServerStrategy(
        cls.resolver)
    cls.coordinator = (
        tf.distribute.experimental.coordinator.ClusterCoordinator(cls.strategy))

  def test_sgd_training(self):
    """Test that SGD optimizer works with dynamic embedding in PS mode."""
    if not context.executing_eagerly():
      self.skipTest('Only test in eager mode.')

    with self.strategy.scope():
      var = de.get_variable('ps_sgd_test',
                            dim=2,
                            initializer=0.1,
                            devices=['/job:ps/task:0', '/job:ps/task:1'])
      shadow_var = de.shadow_ops.ShadowVariable(
          var, name='ps_sgd_shadow', distribute_strategy=self.strategy)
      try:
        from tensorflow.keras.optimizers.legacy import SGD
      except ImportError:
        from tensorflow.keras.optimizers import SGD
      optimizer = SGD(learning_rate=0.01)
      optimizer = de.DynamicEmbeddingOptimizer(optimizer)

    def dist_dataset_fn():
      dataset_values = np.arange(0, 10, dtype=np.int64)
      fn = lambda x: tf.data.Dataset.from_tensor_slices(dataset_values).batch(
          4).repeat(None)
      return self.strategy.distribute_datasets_from_function(fn)

    dataset = self.coordinator.create_per_worker_dataset(dist_dataset_fn)

    @tf.function
    def step_fn(iterator):

      def replica_fn(ids):

        def loss_fn(ids):
          emb = de.shadow_ops.embedding_lookup(shadow_var, ids)
          loss = tf.reduce_mean(emb)
          return loss

        optimizer.minimize(lambda: loss_fn(ids), [shadow_var])

      return self.strategy.run(replica_fn, args=(next(iterator),))

    iterator = iter(dataset)
    for i in range(5):
      self.coordinator.schedule(step_fn, args=(iterator,))
    self.coordinator.join()

    # Verify that all 10 keys were inserted
    self.assertAllEqual(var.size(), 10)

  def test_adam_training(self):
    """Test that Adam optimizer works with dynamic embedding in PS mode."""
    if not context.executing_eagerly():
      self.skipTest('Only test in eager mode.')

    with self.strategy.scope():
      var = de.get_variable('ps_adam_test',
                            dim=4,
                            initializer=0.1,
                            devices=['/job:ps/task:0', '/job:ps/task:1'])
      shadow_var = de.shadow_ops.ShadowVariable(
          var, name='ps_adam_shadow', distribute_strategy=self.strategy)
      try:
        from tensorflow.keras.optimizers.legacy import Adam
      except ImportError:
        from tensorflow.keras.optimizers import Adam
      optimizer = Adam(learning_rate=0.001)
      optimizer = de.DynamicEmbeddingOptimizer(optimizer)

    def dist_dataset_fn():
      dataset_values = np.arange(0, 20, dtype=np.int64)
      fn = lambda x: tf.data.Dataset.from_tensor_slices(dataset_values).batch(
          5).repeat(None)
      return self.strategy.distribute_datasets_from_function(fn)

    dataset = self.coordinator.create_per_worker_dataset(dist_dataset_fn)

    @tf.function
    def step_fn(iterator):

      def replica_fn(ids):

        def loss_fn(ids):
          emb = de.shadow_ops.embedding_lookup(shadow_var, ids)
          loss = tf.reduce_mean(emb)
          return loss

        optimizer.minimize(lambda: loss_fn(ids), [shadow_var])

      return self.strategy.run(replica_fn, args=(next(iterator),))

    iterator = iter(dataset)
    for i in range(3):
      self.coordinator.schedule(step_fn, args=(iterator,))
    self.coordinator.join()

    # Verify that keys were inserted (at least some training happened)
    # The exact count depends on batching and number of steps
    self.assertGreater(var.size().numpy(), 0)

  def test_dynamic_table_growth(self):
    """Test that new IDs appearing during training don't cause crashes.

    This is the core scenario that was broken before the PS fix:
    new IDs trigger table expansion during forward pass, and the
    optimizer update must handle the expanded table correctly.
    """
    if not context.executing_eagerly():
      self.skipTest('Only test in eager mode.')

    with self.strategy.scope():
      var = de.get_variable('ps_growth_test',
                            dim=2,
                            initializer=0.5,
                            devices=['/job:ps/task:0', '/job:ps/task:1'])
      shadow_var = de.shadow_ops.ShadowVariable(
          var, name='ps_growth_shadow', distribute_strategy=self.strategy)
      try:
        from tensorflow.keras.optimizers.legacy import Adam
      except ImportError:
        from tensorflow.keras.optimizers import Adam
      optimizer = Adam(learning_rate=0.001)
      optimizer = de.DynamicEmbeddingOptimizer(optimizer)

    # Use increasing ranges of IDs to simulate table growth
    epoch_data = [
        np.arange(0, 10, dtype=np.int64),  # epoch 1: 10 IDs
        np.arange(5, 25, dtype=np.int64),  # epoch 2: 20 new IDs (5-24)
        np.arange(20, 50, dtype=np.int64),  # epoch 3: 30 new IDs (25-49)
    ]

    for epoch, data in enumerate(epoch_data):

      def make_dist_dataset_fn(d):

        def dist_dataset_fn():
          fn = lambda x: tf.data.Dataset.from_tensor_slices(d).batch(4).repeat(
              None)
          return self.strategy.distribute_datasets_from_function(fn)

        return dist_dataset_fn

      dataset = self.coordinator.create_per_worker_dataset(
          make_dist_dataset_fn(data))

      @tf.function
      def step_fn(iterator):

        def replica_fn(ids):

          def loss_fn(ids):
            emb = de.shadow_ops.embedding_lookup(shadow_var, ids)
            loss = tf.reduce_mean(emb)
            return loss

          optimizer.minimize(lambda: loss_fn(ids), [shadow_var])

        return self.strategy.run(replica_fn, args=(next(iterator),))

      iterator = iter(dataset)
      for _ in range(3):
        self.coordinator.schedule(step_fn, args=(iterator,))
      self.coordinator.join()

    # After all epochs, the table should have grown to contain some IDs
    # The exact count depends on how many batches were processed
    self.assertGreater(var.size().numpy(), 0)


if __name__ == "__main__":
  test.main()
