import tensorflow as tf
import baseline_utils as bsu

def get_features(params, input_dir):
  max_n_docs = params['max_docs']
  features = {'qid': tf.FixedLenSequenceFeature(shape=[], dtype=tf.int64,
                                                allow_missing=True,
                                                default_value=-1),
              'label': tf.FixedLenSequenceFeature(shape=[], dtype=tf.int64,
                                                  allow_missing=True),
              'n_docs': tf.FixedLenFeature(shape=[1], dtype=tf.int64)}

  with open(input_dir + '/features.txt', 'r') as f:
    for fid in f:
      features[fid.strip()] = tf.FixedLenSequenceFeature(
        shape=[], dtype=tf.float32, default_value=0., allow_missing=True)

  # feature_columns = []
  # for featid in features:
  #   if featid not in ['qid', 'label', 'n_docs']:
  #     feature_columns.append(
  #       tf.contrib.layers.real_valued_column(
  #         featid, dimension=1, default_value=0, dtype=tf.float32
  #         )
  #       )
  return features#, feature_columns

def get_letor_examples(params, input_dir, features, num_threads=3):
  batched_examples = tf.contrib.learn.read_batch_examples(
    file_pattern=input_dir + params['partition'] + '.*-*.tfrecord',
    batch_size=params['read_batch'],
    reader=tf.TFRecordReader,
    randomize_input=not params['evaluation'],
    num_epochs=None,
    queue_capacity=10000,
    num_threads=num_threads,
    read_batch_size=1,
    parse_fn=None,
    name=None,
    seed=None
  )

  examples = tf.parse_example(batched_examples, features=features)

  feat_ids = sorted(int(x) for x in features.keys()
                    if x not in ['qid', 'label', 'n_docs'])
  
  doc_tensors = tf.stack([examples[str(x)] for x in feat_ids], axis=2)

  min_values = tf.reduce_min(doc_tensors, axis=1)
  doc_tensors = doc_tensors-min_values[:,None,:]

  max_values = tf.reduce_max(doc_tensors, axis=1)
  safe_max = tf.where(tf.equal(max_values, 0.),
                      tf.ones_like(max_values),
                      max_values)
  doc_tensors = doc_tensors/safe_max[:,None,:]

  # examples = bsu.spread_out_documents(None, examples)

  result = {
    'qid':    tf.cast(examples['qid'], tf.int32),
    'label':  tf.cast(examples['label'], tf.int32),
    'n_docs': tf.cast(examples['n_docs'], tf.int32),
    'doc_tensors': doc_tensors
  }

  for key, values in examples.items():
    if key == 'label':
      tf.summary.histogram("label/input", values)
    else:
      tf.summary.histogram("input/%s" % key, values)

  return result, result['label']