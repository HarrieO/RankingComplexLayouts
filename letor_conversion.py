"""Converts MNIST data to TFRecords file format with Example protos."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
import sys
import random

import numpy as np
import tensorflow as tf

from tensorflow.contrib.learn.python.learn.datasets import mnist

FLAGS = None


def _int64_feature(value):
  return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def _int64_list(value_list):
  return tf.train.Feature(int64_list=tf.train.Int64List(value=value_list))

def _float_feature(value):
  return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))

def _float_list(value_list):
  return tf.train.Feature(float_list=tf.train.FloatList(value=value_list))


def _bytes_feature(value):
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def convert_to(data_set, name):
  """Converts a dataset to tfrecords."""
  images = data_set.images
  labels = data_set.labels
  num_examples = data_set.num_examples

  if images.shape[0] != num_examples:
    raise ValueError('Images size %d does not match label size %d.' %
                     (images.shape[0], num_examples))
  rows = images.shape[1]
  cols = images.shape[2]
  depth = images.shape[3]

  filename = os.path.join(FLAGS.directory, name + '.tfrecords')
  print('Writing', filename)
  writer = tf.python_io.TFRecordWriter(filename)
  for index in range(num_examples):
    image_raw = images[index].tostring()
    example = tf.train.Example(features=tf.train.Features(feature={
        'height': _int64_feature(rows),
        'width': _int64_feature(cols),
        'depth': _int64_feature(depth),
        'label': _int64_feature(int(labels[index])),
        # 'image_raw': _bytes_feature(image_raw)
      }))
    print('Example:', example)
    writer.write(example.SerializeToString())
  writer.close()


def main(unused_argv):
  # # Get the data.
  # data_sets = mnist.read_data_sets(FLAGS.directory,
  #                                  dtype=tf.uint8,
  #                                  reshape=False,
  #                                  validation_size=FLAGS.validation_size)

  # # Convert to Examples and write the result to TFRecords.
  # convert_to(data_sets.train, 'train')
  # convert_to(data_sets.validation, 'validation')
  # convert_to(data_sets.test, 'test')
  train_queries, train_doclists, train_labels, train_feat = _read_file(FLAGS.input_folder + '/train.txt')
  vali_queries, vali_doclists, vali_labels, vali_feat = _read_file(FLAGS.input_folder + '/vali.txt')
  test_queries, test_doclists, test_labels, test_feat = _read_file(FLAGS.input_folder + '/test.txt')

  features_to_keep = train_feat & vali_feat

  for name, queries, doclists, labels, shards in [
          ('train', train_queries, train_doclists, train_labels, FLAGS.train_shards),
          ('vali', vali_queries, vali_doclists, vali_labels, FLAGS.vali_shards),
          ('test', test_queries, test_doclists, test_labels, FLAGS.test_shards),
          ]:
    writers = []
    for i in range(shards):
      writers.append(
        tf.python_io.TFRecordWriter(FLAGS.output_folder + '/%s.%d-of-%d.tfrecord' % (name, i , shards))
        )
    max_n_doc = 0
    for qid, index in queries.items():
      query_feat = {}
      for fid in features_to_keep:
        query_feat[fid] = []

      # cutoff = int(np.random.uniform(4, 11))
      # labels[index] = labels[index][:cutoff]
      # doclists[index] = doclists[index][:cutoff]

      np_labels = np.array(labels[index])
      n_docs = len(labels[index])
      max_n_doc = max(max_n_doc, n_docs)
      # print(qid, 'n doc:', len(doclists[index]), 'labels', np_labels[np_labels > 0])
      for doc in doclists[index]:
        for fid in features_to_keep:
          query_feat[fid].append(doc.get(fid,0.))

      features = {}
      features['qid'] = _int64_list([int(qid)]*len(labels[index]))
      features['label'] = _int64_list(labels[index])
      features['n_docs'] = _int64_list([n_docs])
      # print("%s n_docs: %d" % (name, n_docs))
      for fid in features_to_keep:
        assert len(query_feat[fid]) == n_docs
        min_v = min(query_feat[fid])
        normalized = [x - min_v for x in query_feat[fid]]
        max_v = max(normalized)
        if max_v == 0:
          max_v = 1.
        normalized = [x/max_v for x in normalized]
        features[fid] = _float_list(normalized)
      example = tf.train.Example(features=tf.train.Features(feature=features))
      random.choice(writers).write(example.SerializeToString())
    print('%s total queries:' % name, len(queries))
    print('%s max n docs:' % name, max_n_doc)
    [writer.close() for writer in writers]

  with open(FLAGS.output_folder +'/features.txt', 'w') as f:
    for fid in features_to_keep:
      f.write(fid + '\n')

def _read_file(path, filter_non_uniq=True):
  '''
  Read letor file and returns dict for qid to indices, labels for queries
  and list of doclists of features per doc per query.
  '''
  current_qid = None
  queries = {}
  queryIndex = 0
  doclists = []
  labels = []
  all_features = set()

  feat_bounds = {}

  for line in open(path, 'r'):
    info = line[:line.find('#')].split()

    qid = info[1].split(':')[1]
    label = int(info[0])
    if qid not in queries:
      queryIndex = len(queries)
      queries[qid] = queryIndex
      doclists.append([])
      labels.append([])
      current_qid = qid
    elif qid != current_qid:
      queryIndex = queries[qid]
      current_qid = qid

    featureDict = {}
    for pair in info[2:]:
      featid, feature = pair.split(':')
      all_features.add(featid)
      feat_value = float(feature)
      featureDict[featid] = feat_value
      if featid in feat_bounds:
        feat_bounds[featid] = (min(feat_bounds[featid][0], feat_value),
                               max(feat_bounds[featid][1], feat_value))
      else:
        feat_bounds[featid] = (feat_value, feat_value)
    doclists[queryIndex].append(featureDict)
    labels[queryIndex].append(label)

  if filter_non_uniq:
    unique_features = set()
    for featid in all_features:
      if feat_bounds[featid][0] < feat_bounds[featid][1]:
        unique_features.add(featid)
    return queries, doclists, labels, unique_features
  else:
    return queries, doclists, labels, all_features

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument(
      '--input_folder',
      type=str,
      default='/Users/hroosterhuis/ILPS/datasets/NP2003/Fold1/',
      help='Directory to with input lerot data.'
  )
  parser.add_argument(
      '--output_folder',
      type=str,
      default='/Users/hroosterhuis/ILPS/datasets/TFRecords/NP2003/Fold1/',
      help='Directory to with input lerot data.'
  )
  parser.add_argument(
      '--train_shards',
      type=int,
      default=5,
      help='Number of shards to store data in.'
  )
  parser.add_argument(
      '--vali_shards',
      type=int,
      default=1,
      help='Number of shards to store data in.'
  )
  parser.add_argument(
      '--test_shards',
      type=int,
      default=1,
      help='Number of shards to store data in.'
  )
  FLAGS, unparsed = parser.parse_known_args()
  tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)