"""End-to-end test: variable-length sparse feature (movie_genres) in PS training.

Adds movie_genres as a SparseTensor feature alongside existing dense features
(user_id, movie_id) to validate fit_ps sparse support.
"""
import os
import sys
import time
import signal
import multiprocessing

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow_recommenders_addons import dynamic_embedding as de

try:
  from tensorflow.keras.optimizers.legacy import Adam
except ImportError:
  from tensorflow.keras.optimizers import Adam


class SparseFeatureModel(tf.keras.Model):

  def __init__(self, devices, embedding_size=16):
    super(SparseFeatureModel, self).__init__()
    init = tf.keras.initializers.RandomNormal(0.0, 0.5)

    self.user_embedding = de.keras.layers.SquashedEmbedding(
        embedding_size,
        initializer=init,
        devices=devices,
        with_unique=True,
        bp_v2=True,
        input_key='user_id',
        name='user_emb')

    self.movie_embedding = de.keras.layers.SquashedEmbedding(
        embedding_size,
        initializer=init,
        devices=devices,
        with_unique=True,
        bp_v2=True,
        input_key='movie_id',
        name='movie_emb')

    self.genre_embedding = de.keras.layers.SquashedEmbedding(
        embedding_size,
        combiner='mean',
        initializer=init,
        devices=devices,
        with_unique=True,
        bp_v2=True,
        input_key='movie_genres',
        name='genre_emb')

    self.dnn1 = tf.keras.layers.Dense(64, activation='relu')
    self.dnn2 = tf.keras.layers.Dense(16, activation='relu')
    self.out = tf.keras.layers.Dense(5, activation='softmax')

  def call(self, features):
    user_id = tf.reshape(features['user_id'], (-1, 1))
    movie_id = tf.reshape(features['movie_id'], (-1, 1))
    u = self.user_embedding(user_id)
    m = self.movie_embedding(movie_id)
    g = self.genre_embedding(features['movie_genres'])
    x = tf.concat([u, m, g], axis=1)
    x = self.dnn1(x)
    x = self.dnn2(x)
    return self.out(x)

  def call_with_embeddings(self, features, embeddings, training=None):
    u = embeddings['user_emb']
    m = embeddings['movie_emb']
    g = embeddings['genre_emb']
    x = tf.concat([u, m, g], axis=1)
    x = self.dnn1(x)
    x = self.dnn2(x)
    return self.out(x)


def make_dataset_fn(batch_size=32):

  def dataset_fn():
    ds = tfds.load('movielens/1m-ratings', split='train')

    def process(x):
      features = {
          'user_id': tf.strings.to_number(x['user_id'], tf.int64),
          'movie_id': tf.strings.to_number(x['movie_id'], tf.int64),
          'movie_genres': tf.cast(x['movie_genres'], tf.int64),
      }
      label = tf.one_hot(tf.cast(x['user_rating'] - 1, dtype=tf.int64), 5)
      return features, label

    ds = ds.map(process)
    ds = ds.shuffle(4096, reshuffle_each_iteration=False)
    ds = ds.ragged_batch(batch_size)

    def to_sparse_genres(features, label):
      ragged_genres = features['movie_genres']
      sparse_genres = ragged_genres.to_sparse()
      features = {
          'user_id': features['user_id'],
          'movie_id': features['movie_id'],
          'movie_genres': sparse_genres,
      }
      return features, label

    ds = ds.map(to_sparse_genres)
    return ds

  return dataset_fn


def run_ps(task_id, cluster_config):
  os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
  import tensorflow as tf
  cluster_spec = tf.train.ClusterSpec(cluster_config)
  server = tf.distribute.Server(cluster_spec,
                                protocol='grpc',
                                job_name='ps',
                                task_index=task_id)
  server.join()


def run_worker(task_id, cluster_config):
  os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
  import tensorflow as tf
  cluster_spec = tf.train.ClusterSpec(cluster_config)
  server = tf.distribute.Server(cluster_spec,
                                protocol='grpc',
                                job_name='worker',
                                task_index=task_id)
  server.join()


def run_chief(cluster_config, epochs=3, steps_per_epoch=20, batch_size=32):
  cluster_spec = tf.train.ClusterSpec(cluster_config)
  resolver = tf.distribute.cluster_resolver.SimpleClusterResolver(
      cluster_spec, task_type='chief', task_id=0)
  strategy = tf.distribute.experimental.ParameterServerStrategy(resolver)

  ps_devices = [
      '/job:ps/replica:0/task:{}/device:CPU:0'.format(i)
      for i in range(len(cluster_config['ps']))
  ]

  with strategy.scope():
    model = SparseFeatureModel(devices=ps_devices, embedding_size=16)
    optimizer = Adam(1E-3)
    optimizer = de.DynamicEmbeddingOptimizer(optimizer)
    model.compile(optimizer=optimizer,
                  loss=tf.keras.losses.MeanSquaredError(),
                  metrics=[tf.keras.metrics.AUC(num_thresholds=200)])

  print('\n=== Starting PS training with sparse feature (movie_genres) ===')
  history = de.fit_ps(model,
                      make_dataset_fn(batch_size),
                      strategy,
                      epochs=epochs,
                      steps_per_epoch=steps_per_epoch)

  print('\n=== Training complete ===')
  for k, v in history.items():
    print(f'  {k}: {v}')

  genre_size = model.get_layer('genre_emb').params.size().numpy()
  user_size = model.get_layer('user_emb').params.size().numpy()
  movie_size = model.get_layer('movie_emb').params.size().numpy()
  print(f'\n  genre_emb table size: {genre_size}')
  print(f'  user_emb table size:  {user_size}')
  print(f'  movie_emb table size: {movie_size}')

  assert genre_size > 0, 'genre_emb should have entries'
  assert user_size > 0, 'user_emb should have entries'
  assert movie_size > 0, 'movie_emb should have entries'

  loss_values = history.get('loss', [])
  if len(loss_values) >= 2:
    if loss_values[-1] < loss_values[0]:
      print('\n  ✓ Loss decreased: {:.4f} → {:.4f}'.format(
          loss_values[0], loss_values[-1]))
    else:
      print('\n  ⚠ Loss did not decrease: {:.4f} → {:.4f}'.format(
          loss_values[0], loss_values[-1]))

  print('\n=== PASS: Sparse feature PS training works correctly ===\n')


def main():
  cluster_config = {
      'chief': ['localhost:3330'],
      'ps': ['localhost:3320', 'localhost:3321'],
      'worker': ['localhost:3331'],
  }

  procs = []
  for i in range(len(cluster_config['ps'])):
    p = multiprocessing.Process(target=run_ps, args=(i, cluster_config))
    p.daemon = True
    p.start()
    procs.append(p)
    time.sleep(0.5)

  for i in range(len(cluster_config['worker'])):
    p = multiprocessing.Process(target=run_worker, args=(i, cluster_config))
    p.daemon = True
    p.start()
    procs.append(p)
    time.sleep(0.5)

  time.sleep(1)

  try:
    run_chief(cluster_config, epochs=3, steps_per_epoch=20, batch_size=32)
  finally:
    for p in procs:
      p.terminate()
      p.join(timeout=5)


if __name__ == '__main__':
  main()
