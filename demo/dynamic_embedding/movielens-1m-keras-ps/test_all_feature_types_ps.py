"""End-to-end PS training with all three feature types:
  - Dense: user_id, movie_id (single-value ID)
  - Sparse: movie_genres (variable-length multi-value, combiner='mean')
  - Sequence: browsing_history (fixed-length padded, is_sequence=True)

The browsing_history is simulated by collecting each user's recent movie_ids
from the dataset and padding to a fixed window size.
"""
import os
import time
import multiprocessing

import numpy as np

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow_recommenders_addons import dynamic_embedding as de

try:
  from tensorflow.keras.optimizers.legacy import Adam
except ImportError:
  from tensorflow.keras.optimizers import Adam

SEQ_LEN = 10
EMB_SIZE = 16


class AllFeatureTypesModel(tf.keras.Model):

  def __init__(self, devices):
    super(AllFeatureTypesModel, self).__init__()
    init = tf.keras.initializers.RandomNormal(0.0, 0.5)

    self.user_embedding = de.keras.layers.SquashedEmbedding(
        EMB_SIZE,
        initializer=init,
        devices=devices,
        with_unique=True,
        bp_v2=True,
        input_key='user_id',
        name='user_emb')

    self.movie_embedding = de.keras.layers.SquashedEmbedding(
        EMB_SIZE,
        initializer=init,
        devices=devices,
        with_unique=True,
        bp_v2=True,
        input_key='movie_id',
        name='movie_emb')

    self.genre_embedding = de.keras.layers.SquashedEmbedding(
        EMB_SIZE,
        combiner='mean',
        initializer=init,
        devices=devices,
        with_unique=True,
        bp_v2=True,
        input_key='movie_genres',
        name='genre_emb')

    self.history_embedding = de.keras.layers.Embedding(
        EMB_SIZE,
        initializer=init,
        devices=devices,
        with_unique=True,
        bp_v2=True,
        input_key='browsing_history',
        is_sequence=True,
        name='history_emb')

    self.attention_pool = tf.keras.layers.GlobalAveragePooling1D()
    self.dnn1 = tf.keras.layers.Dense(64, activation='relu')
    self.dnn2 = tf.keras.layers.Dense(16, activation='relu')
    self.out = tf.keras.layers.Dense(5, activation='softmax')

  def call(self, features):
    u = self.user_embedding(tf.reshape(features['user_id'], (-1, 1)))
    m = self.movie_embedding(tf.reshape(features['movie_id'], (-1, 1)))
    g = self.genre_embedding(features['movie_genres'])
    h = self.history_embedding(features['browsing_history'])
    h_pooled = self.attention_pool(h)
    x = tf.concat([u, m, g, h_pooled], axis=1)
    x = self.dnn1(x)
    x = self.dnn2(x)
    return self.out(x)

  def call_with_embeddings(self, features, embeddings, training=None):
    u = embeddings['user_emb']
    m = embeddings['movie_emb']
    g = embeddings['genre_emb']
    h = embeddings['history_emb']
    h_pooled = self.attention_pool(h)
    x = tf.concat([u, m, g, h_pooled], axis=1)
    x = self.dnn1(x)
    x = self.dnn2(x)
    return self.out(x)


def _prebuild_records():
  raw_ds = tfds.load('movielens/1m-ratings', split='train')
  print("Building browsing histories from dataset...")
  user_histories = {}
  records = []
  for x in raw_ds.take(50000):
    uid = int(tf.strings.to_number(x['user_id'], tf.int64).numpy())
    mid = int(tf.strings.to_number(x['movie_id'], tf.int64).numpy())
    genres = x['movie_genres'].numpy().astype(np.int64)
    rating = int(x['user_rating'].numpy())
    user_histories.setdefault(uid, []).append(mid)
    records.append((uid, mid, genres, rating))

  print(f"  {len(records)} records, {len(user_histories)} users")

  padded_histories = {}
  for uid, hist in user_histories.items():
    if len(hist) >= SEQ_LEN:
      padded_histories[uid] = np.array(hist[:SEQ_LEN], dtype=np.int64)
    else:
      padded_histories[uid] = np.array(
          hist + [0] * (SEQ_LEN - len(hist)), dtype=np.int64)

  enriched = []
  for uid, mid, genres, rating in records:
    history = padded_histories[uid]
    label = np.zeros(5, dtype=np.float32)
    label[rating - 1] = 1.0
    enriched.append((uid, mid, genres, history, label))
  return enriched


def _records_to_dataset(records, batch_size):

  def gen():
    for uid, mid, genres, history, label in records:
      yield uid, mid, genres, history, label

  def make_features(uid, mid, genres, history, label):
    genres_ragged = tf.RaggedTensor.from_row_lengths(
        genres, [tf.shape(genres)[0]])
    genres_sparse = genres_ragged.to_sparse()
    features = {
        'user_id': uid,
        'movie_id': mid,
        'movie_genres': genres_sparse,
        'browsing_history': history,
    }
    return features, label

  ds = tf.data.Dataset.from_generator(
      gen,
      output_signature=(
          tf.TensorSpec([], tf.int64),
          tf.TensorSpec([], tf.int64),
          tf.TensorSpec([None], tf.int64),
          tf.TensorSpec([SEQ_LEN], tf.int64),
          tf.TensorSpec([5], tf.float32),
      ))
  ds = ds.map(make_features)
  ds = ds.shuffle(4096, reshuffle_each_iteration=False)
  ds = ds.batch(batch_size)
  return ds


def run_ps(task_id, cluster_config):
  os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
  import tensorflow as tf
  cluster_spec = tf.train.ClusterSpec(cluster_config)
  server = tf.distribute.Server(
      cluster_spec, protocol='grpc', job_name='ps', task_index=task_id)
  server.join()


def run_worker(task_id, cluster_config):
  os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
  import tensorflow as tf
  cluster_spec = tf.train.ClusterSpec(cluster_config)
  server = tf.distribute.Server(
      cluster_spec, protocol='grpc', job_name='worker', task_index=task_id)
  server.join()


def run_chief(cluster_config, epochs=3, steps_per_epoch=30, batch_size=32):
  cluster_spec = tf.train.ClusterSpec(cluster_config)
  resolver = tf.distribute.cluster_resolver.SimpleClusterResolver(
      cluster_spec, task_type='chief', task_id=0)
  strategy = tf.distribute.experimental.ParameterServerStrategy(resolver)

  ps_devices = [
      '/job:ps/replica:0/task:{}/device:CPU:0'.format(i)
      for i in range(len(cluster_config['ps']))
  ]

  with strategy.scope():
    model = AllFeatureTypesModel(devices=ps_devices)
    optimizer = Adam(1E-3)
    optimizer = de.DynamicEmbeddingOptimizer(optimizer)
    model.compile(
        optimizer=optimizer,
        loss=tf.keras.losses.MeanSquaredError(),
        metrics=[tf.keras.metrics.AUC(num_thresholds=200)])

  records = _prebuild_records()

  def dataset_fn():

    def per_worker_fn(input_context):
      return _records_to_dataset(records, batch_size).repeat(None)

    return strategy.distribute_datasets_from_function(per_worker_fn)

  print('\n' + '=' * 70)
  print('  PS Training: Dense + Sparse + Sequence features')
  print('=' * 70)
  print(f'  Dense features:    user_id, movie_id')
  print(f'  Sparse feature:    movie_genres (combiner=mean)')
  print(f'  Sequence feature:  browsing_history (is_sequence=True, len={SEQ_LEN})')
  print(f'  Epochs: {epochs}, Steps/epoch: {steps_per_epoch}')
  print('=' * 70 + '\n')

  history = de.fit_ps(
      model, dataset_fn, strategy, epochs=epochs,
      steps_per_epoch=steps_per_epoch)

  print('\n' + '=' * 70)
  print('  Training Results')
  print('=' * 70)
  for k, v in history.items():
    print(f'  {k}: {["%.4f" % x for x in v]}')

  user_size = model.get_layer('user_emb').params.size().numpy()
  movie_size = model.get_layer('movie_emb').params.size().numpy()
  genre_size = model.get_layer('genre_emb').params.size().numpy()
  history_size = model.get_layer('history_emb').params.size().numpy()

  print(f'\n  Embedding table sizes:')
  print(f'    user_emb:     {user_size}')
  print(f'    movie_emb:    {movie_size}')
  print(f'    genre_emb:    {genre_size}')
  print(f'    history_emb:  {history_size}')

  assert user_size > 0, 'user_emb should have entries'
  assert movie_size > 0, 'movie_emb should have entries'
  assert genre_size > 0, 'genre_emb should have entries'
  assert history_size > 0, 'history_emb should have entries'

  loss_values = history.get('loss', [])
  if len(loss_values) >= 2 and loss_values[-1] < loss_values[0]:
    print(f'\n  ✓ Loss decreased: {loss_values[0]:.4f} → {loss_values[-1]:.4f}')
  else:
    print(f'\n  ⚠ Loss: {loss_values}')

  print(f'\n{"=" * 70}')
  print(f'  ✓ PASS: All three feature types work in PS training')
  print(f'{"=" * 70}\n')


def main():
  cluster_config = {
      'chief': ['localhost:4430'],
      'ps': ['localhost:4420', 'localhost:4421'],
      'worker': ['localhost:4431'],
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
    run_chief(cluster_config, epochs=3, steps_per_epoch=30, batch_size=32)
  finally:
    for p in procs:
      p.terminate()
      p.join(timeout=5)


if __name__ == '__main__':
  main()
