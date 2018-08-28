import tensorflow as tf
import model_utils as mu
import rnn_utils as ru
import numpy as np
from tensorflow.python.ops import variable_scope
from tensorflow.contrib import layers

def get_sigmoid_layer(params, sub_name, label_network, inference, reuse_variable_scope):
  parent_scope = params['model_name'] + sub_name
  # avoiding non_local parameter scope in python 2.7
  class c:
    reuse_var_scope = reuse_variable_scope

  hidden_state_size = params['hidden_state_size']
  hidden_dropout = params['hidden_dropout']
  if inference or label_network or hidden_dropout < 1:
    keep_prob = 1.
  else:
    keep_prob = hidden_dropout

  def sigmoid_layer(hidden_state, input_vector):
    l2_scale = params['l2_scale']
    with variable_scope.variable_scope(parent_scope,
        values=(hidden_state, input_vector),
        reuse=c.reuse_var_scope) as gru_scope:

      # In case function is called more than once.
      c.reuse_var_scope = True

      sigmoid_input = tf.concat([hidden_state, input_vector], axis=1)

      return layers.fully_connected(
        sigmoid_input, hidden_state_size,
        activation_fn=tf.sigmoid,
        biases_initializer=None,
        trainable=not label_network)

  return sigmoid_layer

def model(params, examples, labels):
  serp_len = params['serp_len']
  doc_emb_size = params['doc_emb'][-1]
  hidden_state_size = params['hidden_state_size']
  docs = examples['doc_tensors']
  batch_size = docs.shape[0].value
  batch_max_docs = tf.shape(docs)[1]
  docs_per_query = examples['n_docs']

  result = {
    'docs_per_query': docs_per_query,
    }

  doc_emb = mu._shared_doc_embeddings(docs, params,
                                      '/main/doc_emb')


  hidden_init = tf.zeros([batch_size, hidden_state_size])
  

  rnn_fn = get_sigmoid_layer(params, '/main/gru',
                                  label_network=False,
                                  inference=False,
                                  reuse_variable_scope=False)

  batch_ind = tf.range(batch_size, dtype=tf.int64)[:, None]

  hidden_state = hidden_init
  n_doc_filter = tf.sequence_mask(docs_per_query[:, 0], batch_max_docs)
  doc_filter = tf.where(n_doc_filter,
        tf.zeros([batch_size, batch_max_docs]),
        tf.fill([batch_size, batch_max_docs], np.NINF))
   #tf.zeros([n_docs, hidden_state_size])
  serp_labels = []
  serp_ind = []
  probs = []
  for i in range(serp_len):
    hidden_states = tf.tile(hidden_state[:, None, :], [1, batch_max_docs, 1])
    score_input = tf.concat([hidden_states, doc_emb], axis=2)
    scores = mu._create_subnetwork(score_input,
                                   params,
                                   subnetwork_name='/main/scoring',
                                   label_network=False,
                                   reuse_variable_scope=i>0,
                                   inference=False)[:, :, 0]
    seq_mask = tf.less(i, docs_per_query[:,0])
    if params['evaluation']:
      sampled = tf.argmax(scores + doc_filter, axis=1)[:, None]
    else:
      sampled = tf.multinomial((scores + doc_filter), 1)
      sampled = tf.where(seq_mask,
                         sampled,
                         tf.zeros_like(sampled))
    serp_ind.append(sampled[:, 0])
      # sampled = tf.Print(sampled, [tf.shape(scores + doc_filter)], 'shape: ')
      # sampled = tf.Print(sampled, [tf.reduce_sum(scores, axis=1)], 'scores: ')
      # sampled = tf.Print(sampled, [tf.reduce_sum(doc_filter, axis=1)], 'filter: ')
      # sampled = tf.Print(sampled, [tf.reduce_sum(scores + doc_filter, axis=1)], 'sum: ')
      # sampled = tf.Print(sampled, [docs_per_query[:,0]], 'docs_per_query: ')
      # sampled = tf.Print(sampled, [sampled[:,0]], 'sampled: ')

    gather_ind = tf.concat([batch_ind, sampled], axis=1)
    sampled_scores = tf.gather_nd(scores, gather_ind)
    exp_scores = tf.exp(scores + doc_filter)
    exp_scores = tf.where(tf.less(exp_scores, 0.001),
                          exp_scores + 0.001,
                          exp_scores)
    denom = tf.reduce_sum(exp_scores, axis=1)

    doc_filter += tf.one_hot(sampled[:, 0], batch_max_docs,
                             on_value=np.NINF, off_value=0.)
    select_doc = tf.gather_nd(doc_emb, gather_ind)
    hidden_state = rnn_fn(hidden_state, select_doc)

    select_labels = tf.gather_nd(labels, gather_ind)
    serp_labels.append(tf.where(
        seq_mask,
        select_labels,
        tf.zeros([batch_size], dtype=tf.int32),
      ))
    probs.append(tf.where(
        seq_mask,
        sampled_scores - tf.log(denom),
        tf.zeros([batch_size]),
      ))
    # probs[-1] = tf.Print(probs[-1], [denom], 'denom %d:' % i)
    # probs[-1] = tf.Print(probs[-1], [tf.exp(np.NINF)], 'test %d:' % i)

    # probs[-1] = tf.Print(probs[-1], [sampled], 'sampled %d:' % i)
    # probs[-1] = tf.Print(probs[-1], [serp_labels[-1]], 'labels %d:' % i)
    # probs[-1] = tf.Print(probs[-1], [probs[-1]], 'probs %d:' % i)


  result['labels'] = tf.stack(serp_labels, axis=1)
  result['probs'] = tf.stack(probs, axis=1)
  result['serp_ind'] = tf.stack(serp_ind, axis=1)


  # result['probs'] = tf.Print(result['probs'], [result['serp_ind']], 'serp_ind: ')
  # result['probs'] = tf.Print(result['probs'], [tf.exp(result['probs'])], 'prob: ')

  return result