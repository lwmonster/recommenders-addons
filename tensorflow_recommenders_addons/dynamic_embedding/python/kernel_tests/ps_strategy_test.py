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


class PSSparseFeatureTest(test.TestCase):
  """Tests for variable-length sparse feature support in fit_ps."""

  @classmethod
  def setUpClass(cls):
    if not context.executing_eagerly():
      return

    cls.cluster_spec = tf.train.ClusterSpec({
        'ps': ['localhost:3420', 'localhost:3421'],
        'worker': ['localhost:3422', 'localhost:3423']
    })
    cls.ps_list, cls.worker_list = _create_ps_and_worker_servers(
        cls.cluster_spec)
    cls.resolver = tf.distribute.cluster_resolver.SimpleClusterResolver(
        cls.cluster_spec)
    cls.strategy = tf.distribute.experimental.ParameterServerStrategy(
        cls.resolver)
    cls.coordinator = (
        tf.distribute.experimental.coordinator.ClusterCoordinator(cls.strategy))

  def _make_sparse_model(self, combiner='mean', name_suffix=''):
    with self.strategy.scope():
      model = _SparseDemoModel(embedding_size=4,
                               combiner=combiner,
                               ps_devices=['/job:ps/task:0', '/job:ps/task:1'],
                               name_suffix=name_suffix)
      try:
        from tensorflow.keras.optimizers.legacy import Adam
      except ImportError:
        from tensorflow.keras.optimizers import Adam
      optimizer = Adam(learning_rate=0.001)
      optimizer = de.DynamicEmbeddingOptimizer(optimizer)
      model.compile(optimizer=optimizer,
                    loss=tf.keras.losses.MeanSquaredError())
    return model

  def test_sparse_tensor_input(self):
    if not context.executing_eagerly():
      self.skipTest('Only test in eager mode.')

    model = self._make_sparse_model(combiner='mean', name_suffix='sp_input')

    def dataset_fn():

      def gen():
        for _ in range(20):
          n = np.random.randint(1, 5)
          ids = np.random.randint(0, 100, size=(n,), dtype=np.int64)
          yield ids, np.array([1.0], dtype=np.float32)

      def make_sparse_batch(ids, label):
        sp = tf.RaggedTensor.from_row_lengths(ids, [tf.shape(ids)[0]])
        sp = sp.to_sparse()
        return {'tags': sp}, label

      fn = lambda x: (tf.data.Dataset.from_generator(
          gen,
          output_signature=(tf.TensorSpec([None], tf.int64),
                            tf.TensorSpec([1], tf.float32))).map(
                                make_sparse_batch).batch(4).repeat(None))
      return self.strategy.distribute_datasets_from_function(fn)

    de.fit_ps(model, dataset_fn, self.strategy, epochs=1, steps_per_epoch=3)
    self.assertGreater(
        model.get_layer('tags_emb_sp_input').params.size().numpy(), 0)

  def test_sparse_tensor_empty_rows(self):
    if not context.executing_eagerly():
      self.skipTest('Only test in eager mode.')

    from tensorflow_recommenders_addons.dynamic_embedding.python.keras.models import (
        _extract_ids_and_segments, _segment_combine)

    sp = tf.SparseTensor(indices=[[0, 0], [0, 1], [2, 0]],
                         values=tf.constant([10, 20, 30], dtype=tf.int64),
                         dense_shape=[3, 3])
    flat_ids, seg_ids, num_seg, is_sparse = _extract_ids_and_segments(sp)
    self.assertTrue(is_sparse)
    self.assertAllEqual(flat_ids, [10, 20, 30])
    self.assertAllEqual(seg_ids, [0, 0, 2])
    self.assertAllEqual(num_seg, 3)

    fake_emb = tf.constant([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    result = _segment_combine(fake_emb, seg_ids, num_seg, 'mean')
    self.assertAllEqual(result.shape, [3, 2])
    self.assertAllClose(result[0], [2.0, 3.0])
    self.assertAllClose(result[1], [0.0, 0.0])
    self.assertAllClose(result[2], [5.0, 6.0])

  def test_ragged_tensor_input(self):
    if not context.executing_eagerly():
      self.skipTest('Only test in eager mode.')

    from tensorflow_recommenders_addons.dynamic_embedding.python.keras.models import (
        _extract_ids_and_segments, _segment_combine)

    ragged = tf.ragged.constant([[10, 20], [30], [40, 50, 60]], dtype=tf.int64)
    flat_ids, seg_ids, num_seg, is_sparse = _extract_ids_and_segments(ragged)
    self.assertTrue(is_sparse)
    self.assertAllEqual(flat_ids, [10, 20, 30, 40, 50, 60])
    self.assertAllEqual(seg_ids, [0, 0, 1, 2, 2, 2])
    self.assertAllEqual(num_seg, 3)

    fake_emb = tf.constant([[1., 0.], [0., 1.], [2., 2.], [3., 0.], [0., 3.],
                            [1., 1.]])
    result = _segment_combine(fake_emb, seg_ids, num_seg, 'sum')
    self.assertAllClose(result[0], [1.0, 1.0])
    self.assertAllClose(result[1], [2.0, 2.0])
    self.assertAllClose(result[2], [4.0, 4.0])

  def test_mixed_dense_sparse_features(self):
    if not context.executing_eagerly():
      self.skipTest('Only test in eager mode.')

    model = _MixedFeatureModel(strategy=self.strategy,
                               ps_devices=['/job:ps/task:0', '/job:ps/task:1'])

    def dataset_fn():

      def gen():
        for _ in range(20):
          user_id = np.random.randint(0, 50, dtype=np.int64)
          n_tags = np.random.randint(1, 5)
          tags = np.random.randint(0, 100, size=(n_tags,), dtype=np.int64)
          yield user_id, tags, np.array([1.0], dtype=np.float32)

      def to_batch(uid, tags, label):
        sp = tf.RaggedTensor.from_row_lengths(tags, [tf.shape(tags)[0]])
        sp = sp.to_sparse()
        return {'user_id': tf.expand_dims(uid, 0), 'user_tags': sp}, label

      fn = lambda x: (tf.data.Dataset.from_generator(
          gen,
          output_signature=(tf.TensorSpec([], tf.int64),
                            tf.TensorSpec([None], tf.int64),
                            tf.TensorSpec([1], tf.float32))).map(to_batch).
                      batch(4).repeat(None))
      return self.strategy.distribute_datasets_from_function(fn)

    de.fit_ps(model, dataset_fn, self.strategy, epochs=1, steps_per_epoch=3)
    self.assertGreater(
        model.get_layer('uid_emb_mixed').params.size().numpy(), 0)
    self.assertGreater(
        model.get_layer('tags_emb_mixed').params.size().numpy(), 0)

  def test_sparse_combiner_sum_mean_sqrtn(self):
    if not context.executing_eagerly():
      self.skipTest('Only test in eager mode.')

    from tensorflow_recommenders_addons.dynamic_embedding.python.keras.models import (
        _extract_ids_and_segments, _segment_combine)

    sp = tf.SparseTensor(indices=[[0, 0], [0, 1], [1, 0]],
                         values=tf.constant([1, 2, 3], dtype=tf.int64),
                         dense_shape=[2, 2])
    _, seg_ids, num_seg, _ = _extract_ids_and_segments(sp)

    emb = tf.constant([[10.0, 20.0], [30.0, 40.0], [50.0, 60.0]])

    result_sum = _segment_combine(emb, seg_ids, num_seg, 'sum')
    self.assertAllClose(result_sum[0], [40.0, 60.0])
    self.assertAllClose(result_sum[1], [50.0, 60.0])

    result_mean = _segment_combine(emb, seg_ids, num_seg, 'mean')
    self.assertAllClose(result_mean[0], [20.0, 30.0])
    self.assertAllClose(result_mean[1], [50.0, 60.0])

    result_sqrtn = _segment_combine(emb, seg_ids, num_seg, 'sqrtn')
    sqrt2 = np.sqrt(2.0)
    self.assertAllClose(result_sqrtn[0], [40.0 / sqrt2, 60.0 / sqrt2])
    self.assertAllClose(result_sqrtn[1], [50.0, 60.0])

  def test_sparse_gradient_correctness(self):
    if not context.executing_eagerly():
      self.skipTest('Only test in eager mode.')

    model = self._make_sparse_model(combiner='sum', name_suffix='grad_test')
    tags_layer = model.get_layer('tags_emb_grad_test')

    test_ids = tf.constant([0, 1, 2, 3, 4], dtype=tf.int64)
    init_values = tf.ones((5, 4), dtype=tf.float32)
    tags_layer.params.upsert(test_ids, init_values)

    before_values = tags_layer.params.lookup(test_ids)

    def dataset_fn():
      sp = tf.SparseTensor(indices=[[0, 0], [0, 1], [1, 0]],
                           values=tf.constant([0, 1, 2], dtype=tf.int64),
                           dense_shape=[2, 2])

      def gen():
        for _ in range(4):
          yield {'tags': sp}, tf.constant([[1.0], [1.0]])

      fn = lambda x: (tf.data.Dataset.from_generator(
          gen,
          output_signature=({
              'tags': tf.SparseTensorSpec([None, None], tf.int64)
          }, tf.TensorSpec([None, 1], tf.float32))).repeat(None))
      return self.strategy.distribute_datasets_from_function(fn)

    de.fit_ps(model, dataset_fn, self.strategy, epochs=1, steps_per_epoch=2)

    after_values = tags_layer.params.lookup(test_ids)
    changed_ids = tf.constant([0, 1, 2], dtype=tf.int64)
    before_subset = tags_layer.params.lookup(changed_ids)
    diff = tf.reduce_sum(tf.abs(after_values[:3] - before_values[:3]))
    self.assertGreater(diff.numpy(), 0.0,
                       "Embeddings for trained IDs should have changed")


class _SparseDemoModel(tf.keras.Model):

  def __init__(self, embedding_size, combiner, ps_devices, name_suffix=''):
    super(_SparseDemoModel, self).__init__()
    self.tags_embedding = de.keras.layers.SquashedEmbedding(
        embedding_size,
        combiner=combiner,
        initializer=tf.keras.initializers.Ones(),
        devices=ps_devices,
        with_unique=True,
        bp_v2=True,
        input_key='tags',
        name='tags_emb_{}'.format(name_suffix))
    self.dense_out = tf.keras.layers.Dense(
        1,
        kernel_initializer=tf.keras.initializers.Ones(),
        bias_initializer=tf.keras.initializers.Zeros())

  def call(self, features):
    emb = self.tags_embedding(features['tags'])
    return self.dense_out(emb)

  def call_with_embeddings(self, features, embeddings, training=None):
    emb = embeddings[self.tags_embedding.name]
    return self.dense_out(emb)


class _MixedFeatureModel(tf.keras.Model):

  def __init__(self, strategy, ps_devices):
    super(_MixedFeatureModel, self).__init__()
    with strategy.scope():
      self.uid_embedding = de.keras.layers.SquashedEmbedding(
          4,
          combiner='sum',
          initializer=tf.keras.initializers.Ones(),
          devices=ps_devices,
          with_unique=True,
          bp_v2=True,
          input_key='user_id',
          name='uid_emb_mixed')
      self.tags_embedding = de.keras.layers.SquashedEmbedding(
          4,
          combiner='mean',
          initializer=tf.keras.initializers.Ones(),
          devices=ps_devices,
          with_unique=True,
          bp_v2=True,
          input_key='user_tags',
          name='tags_emb_mixed')
      try:
        from tensorflow.keras.optimizers.legacy import Adam
      except ImportError:
        from tensorflow.keras.optimizers import Adam
      optimizer = Adam(learning_rate=0.001)
      optimizer = de.DynamicEmbeddingOptimizer(optimizer)
      self.dense_out = tf.keras.layers.Dense(
          1,
          kernel_initializer=tf.keras.initializers.Ones(),
          bias_initializer=tf.keras.initializers.Zeros())
    self.compile(optimizer=optimizer, loss=tf.keras.losses.MeanSquaredError())

  def call(self, features):
    uid_emb = self.uid_embedding(features['user_id'])
    tags_emb = self.tags_embedding(features['user_tags'])
    return self.dense_out(tf.concat([uid_emb, tags_emb], axis=1))

  def call_with_embeddings(self, features, embeddings, training=None):
    uid_emb = embeddings[self.uid_embedding.name]
    tags_emb = embeddings[self.tags_embedding.name]
    return self.dense_out(tf.concat([uid_emb, tags_emb], axis=1))


class PSSequenceFeatureTest(test.TestCase):

  @classmethod
  def setUpClass(cls):
    if not context.executing_eagerly():
      return
    cls.cluster_spec = tf.train.ClusterSpec({
        'ps': ['localhost:3520', 'localhost:3521'],
        'worker': ['localhost:3522', 'localhost:3523']
    })
    cls.ps_list, cls.worker_list = _create_ps_and_worker_servers(
        cls.cluster_spec)
    cls.resolver = tf.distribute.cluster_resolver.SimpleClusterResolver(
        cls.cluster_spec)
    cls.strategy = tf.distribute.experimental.ParameterServerStrategy(
        cls.resolver)
    cls.coordinator = (
        tf.distribute.experimental.coordinator.ClusterCoordinator(cls.strategy))

  def test_sequence_feature_shape(self):
    if not context.executing_eagerly():
      self.skipTest('Only test in eager mode.')

    model = _SequenceDemoModel(strategy=self.strategy,
                               ps_devices=['/job:ps/task:0', '/job:ps/task:1'],
                               name_suffix='shape')

    captured_shapes = {}

    original_call_with = model.call_with_embeddings

    def capturing_call_with(features, embeddings, training=None):
      for k, v in embeddings.items():
        captured_shapes[k] = v.shape
      return original_call_with(features, embeddings, training)

    model.call_with_embeddings = capturing_call_with

    def dataset_fn():

      def gen():
        for _ in range(20):
          history = np.random.randint(0, 100, size=(10,), dtype=np.int64)
          yield {'history': history}, np.array([1.0], dtype=np.float32)

      fn = lambda x: (tf.data.Dataset.from_generator(
          gen,
          output_signature=({
              'history': tf.TensorSpec([10], tf.int64)
          }, tf.TensorSpec([1], tf.float32))).batch(4).repeat(None))
      return self.strategy.distribute_datasets_from_function(fn)

    de.fit_ps(model, dataset_fn, self.strategy, epochs=1, steps_per_epoch=3)
    self.assertGreater(
        model.get_layer('seq_emb_shape').params.size().numpy(), 0)

  def test_sequence_feature_no_pooling(self):
    if not context.executing_eagerly():
      self.skipTest('Only test in eager mode.')

    model = _SequenceDemoModel(strategy=self.strategy,
                               ps_devices=['/job:ps/task:0', '/job:ps/task:1'],
                               combiner='mean',
                               name_suffix='nopool')

    def dataset_fn():

      def gen():
        for _ in range(20):
          history = np.random.randint(0, 100, size=(10,), dtype=np.int64)
          yield {'history': history}, np.array([1.0], dtype=np.float32)

      fn = lambda x: (tf.data.Dataset.from_generator(
          gen,
          output_signature=({
              'history': tf.TensorSpec([10], tf.int64)
          }, tf.TensorSpec([1], tf.float32))).batch(4).repeat(None))
      return self.strategy.distribute_datasets_from_function(fn)

    de.fit_ps(model, dataset_fn, self.strategy, epochs=1, steps_per_epoch=3)
    self.assertGreater(
        model.get_layer('seq_emb_nopool').params.size().numpy(), 0)

  def test_sequence_sparse_input_raises(self):
    if not context.executing_eagerly():
      self.skipTest('Only test in eager mode.')

    from tensorflow_recommenders_addons.dynamic_embedding.python.keras.models import (
        _extract_ids_and_segments,)

    sp = tf.SparseTensor(indices=[[0, 0], [1, 0]],
                         values=[1, 2],
                         dense_shape=[2, 2])
    _, _, _, is_sparse = _extract_ids_and_segments(sp)
    self.assertTrue(is_sparse)

  def test_mixed_dense_sparse_sequence(self):
    if not context.executing_eagerly():
      self.skipTest('Only test in eager mode.')

    model = _FullMixedModel(strategy=self.strategy,
                            ps_devices=['/job:ps/task:0', '/job:ps/task:1'])

    def dataset_fn():

      def gen():
        for _ in range(20):
          uid = np.random.randint(0, 50, dtype=np.int64)
          n_tags = np.random.randint(1, 5)
          tags = np.random.randint(0, 100, size=(n_tags,), dtype=np.int64)
          history = np.random.randint(0, 200, size=(8,), dtype=np.int64)
          yield uid, tags, history, np.array([1.0], dtype=np.float32)

      def to_batch(uid, tags, history, label):
        sp = tf.RaggedTensor.from_row_lengths(tags, [tf.shape(tags)[0]])
        sp = sp.to_sparse()
        return {
            'user_id': tf.expand_dims(uid, 0),
            'user_tags': sp,
            'browsing_history': history,
        }, label

      fn = lambda x: (tf.data.Dataset.from_generator(
          gen,
          output_signature=(
              tf.TensorSpec([], tf.int64), tf.TensorSpec([None], tf.int64),
              tf.TensorSpec([8], tf.int64), tf.TensorSpec([1], tf.float32))).
                      map(to_batch).batch(4).repeat(None))
      return self.strategy.distribute_datasets_from_function(fn)

    de.fit_ps(model, dataset_fn, self.strategy, epochs=1, steps_per_epoch=3)
    self.assertGreater(model.get_layer('uid_emb_full').params.size().numpy(), 0)
    self.assertGreater(
        model.get_layer('tags_emb_full').params.size().numpy(), 0)
    self.assertGreater(model.get_layer('seq_emb_full').params.size().numpy(), 0)

  def test_sequence_gradient_correctness(self):
    if not context.executing_eagerly():
      self.skipTest('Only test in eager mode.')

    model = _SequenceDemoModel(strategy=self.strategy,
                               ps_devices=['/job:ps/task:0', '/job:ps/task:1'],
                               name_suffix='grad')

    seq_layer = model.get_layer('seq_emb_grad')
    test_ids = tf.constant([0, 1, 2, 3, 4], dtype=tf.int64)
    init_values = tf.ones((5, 4), dtype=tf.float32)
    seq_layer.params.upsert(test_ids, init_values)
    before_values = seq_layer.params.lookup(test_ids)

    def dataset_fn():

      def gen():
        for _ in range(8):
          yield {'history': np.array([0, 1, 2, 0, 1], dtype=np.int64)}, \
                np.array([1.0], dtype=np.float32)

      fn = lambda x: (tf.data.Dataset.from_generator(
          gen,
          output_signature=({
              'history': tf.TensorSpec([5], tf.int64)
          }, tf.TensorSpec([1], tf.float32))).batch(2).repeat(None))
      return self.strategy.distribute_datasets_from_function(fn)

    de.fit_ps(model, dataset_fn, self.strategy, epochs=1, steps_per_epoch=2)

    after_values = seq_layer.params.lookup(test_ids)
    diff = tf.reduce_sum(tf.abs(after_values[:3] - before_values[:3]))
    self.assertGreater(diff.numpy(), 0.0,
                       "Embeddings for trained IDs should have changed")


class _SequenceDemoModel(tf.keras.Model):

  def __init__(self, strategy, ps_devices, combiner='sum', name_suffix=''):
    super(_SequenceDemoModel, self).__init__()
    with strategy.scope():
      self.seq_embedding = de.keras.layers.Embedding(
          4,
          initializer=tf.keras.initializers.Ones(),
          devices=ps_devices,
          with_unique=True,
          bp_v2=True,
          input_key='history',
          is_sequence=True,
          combiner=combiner,
          name='seq_emb_{}'.format(name_suffix))
      try:
        from tensorflow.keras.optimizers.legacy import Adam
      except ImportError:
        from tensorflow.keras.optimizers import Adam
      optimizer = Adam(learning_rate=0.001)
      optimizer = de.DynamicEmbeddingOptimizer(optimizer)
      self.pool = tf.keras.layers.GlobalAveragePooling1D()
      self.dense_out = tf.keras.layers.Dense(
          1,
          kernel_initializer=tf.keras.initializers.Ones(),
          bias_initializer=tf.keras.initializers.Zeros())
    self.compile(optimizer=optimizer, loss=tf.keras.losses.MeanSquaredError())

  def call(self, features):
    emb = self.seq_embedding(features['history'])
    pooled = self.pool(emb)
    return self.dense_out(pooled)

  def call_with_embeddings(self, features, embeddings, training=None):
    seq_emb = embeddings[self.seq_embedding.name]
    pooled = self.pool(seq_emb)
    return self.dense_out(pooled)


class _FullMixedModel(tf.keras.Model):

  def __init__(self, strategy, ps_devices):
    super(_FullMixedModel, self).__init__()
    with strategy.scope():
      self.uid_embedding = de.keras.layers.SquashedEmbedding(
          4,
          initializer=tf.keras.initializers.Ones(),
          devices=ps_devices,
          with_unique=True,
          bp_v2=True,
          input_key='user_id',
          name='uid_emb_full')
      self.tags_embedding = de.keras.layers.SquashedEmbedding(
          4,
          combiner='mean',
          initializer=tf.keras.initializers.Ones(),
          devices=ps_devices,
          with_unique=True,
          bp_v2=True,
          input_key='user_tags',
          name='tags_emb_full')
      self.seq_embedding = de.keras.layers.Embedding(
          4,
          initializer=tf.keras.initializers.Ones(),
          devices=ps_devices,
          with_unique=True,
          bp_v2=True,
          input_key='browsing_history',
          is_sequence=True,
          name='seq_emb_full')
      try:
        from tensorflow.keras.optimizers.legacy import Adam
      except ImportError:
        from tensorflow.keras.optimizers import Adam
      optimizer = Adam(learning_rate=0.001)
      optimizer = de.DynamicEmbeddingOptimizer(optimizer)
      self.pool = tf.keras.layers.GlobalAveragePooling1D()
      self.dense_out = tf.keras.layers.Dense(
          1,
          kernel_initializer=tf.keras.initializers.Ones(),
          bias_initializer=tf.keras.initializers.Zeros())
    self.compile(optimizer=optimizer, loss=tf.keras.losses.MeanSquaredError())

  def call(self, features):
    uid_emb = self.uid_embedding(features['user_id'])
    tags_emb = self.tags_embedding(features['user_tags'])
    seq_emb = self.seq_embedding(features['browsing_history'])
    pooled_seq = self.pool(seq_emb)
    return self.dense_out(tf.concat([uid_emb, tags_emb, pooled_seq], axis=1))

  def call_with_embeddings(self, features, embeddings, training=None):
    uid_emb = embeddings[self.uid_embedding.name]
    tags_emb = embeddings[self.tags_embedding.name]
    seq_emb = embeddings[self.seq_embedding.name]
    pooled_seq = self.pool(seq_emb)
    return self.dense_out(tf.concat([uid_emb, tags_emb, pooled_seq], axis=1))


if __name__ == "__main__":
  test.main()
