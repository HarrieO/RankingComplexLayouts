import tensorflow as tf
import numpy as np
import model_utils as mu
import rnn_utils as ru

def naive_model(params, examples, labels, epsilon):
  serp_len = params['serp_len']
  doc_emb_size = params['doc_emb'][-1]
  docs = mu._get_doc_tensors(examples, params, 'main')
  result = {}

  n_docs = tf.shape(docs)[0]
  ind_range = tf.cast(tf.range(n_docs), dtype=tf.int64)
  result['docs_per_query'] = n_docs

  score_filter = tf.zeros([n_docs, 1], dtype=tf.float32)
  neginf = tf.add(score_filter,np.NINF)

  doc_emb = mu._shared_doc_embeddings(docs, params,
                                      '/main/doc_emb',
                                      inference=True)

  serp_docs = []
  serp_labels = []
  doc_input = doc_emb
  for i in range(serp_len):
    scores = mu._create_subnetwork(doc_input, params,
                                   '/main/score/pos_%d' % i,
                                   reuse_variable_scope=False,
                                   inference=True)
    tf.summary.scalar('policy/scores/pos_%d' % i, tf.reduce_mean(scores))
    action, max_ind = mu.select_eps_greedy_action(scores,
                                                  epsilon,
                                                  score_filter)

    select_doc = tf.gather(docs, action)
    serp_labels.append(
      tf.cond(
        tf.less(i, n_docs),
        lambda: tf.gather(labels, action, axis=0),
        lambda: tf.constant([[0]], dtype=tf.int64),
      )
    )
    serp_docs.append(select_doc)

    if i > 0:
      result['max_doc_%d' % i] = tf.gather(docs, max_ind)

    if i < serp_len-1:
      score_filter = tf.where(tf.equal(ind_range, action),
                              neginf,
                              score_filter)
      select_doc_input = tf.gather(doc_emb, action)
      tiled = tf.tile(select_doc_input, [n_docs, 1])
      doc_input = tf.concat([tiled, doc_input], axis=1)
  result['serp'] = tf.stack(serp_docs, axis=1)
  # result['max_docs'] = tf.stack(max_docs, axis=1)
  result['labels'] = tf.concat(serp_labels, axis=1)
  tf.summary.histogram("label/output", result['labels'])
  return result

# positional bag of words
def pbow_model(params, examples, labels, epsilon):
  serp_len = params['serp_len']
  doc_emb_size = params['doc_emb'][-1]
  hidden_state_size = params['hidden_state_size']
  docs = mu._get_doc_tensors(examples, params, 'main')
  result = {}

  n_docs = tf.shape(docs)[0]
  result['docs_per_query'] = n_docs

  score_filter = tf.zeros([n_docs, 1], dtype=tf.float32)
  neginf = tf.add(score_filter,np.NINF)
  ind_range = tf.cast(tf.range(n_docs), dtype=tf.int64)

  doc_emb = mu._shared_doc_embeddings(docs, params,
                                      '/main/doc_emb',
                                      inference=True)

  pbow = tf.zeros([1, hidden_state_size])
  doc_i = tf.zeros([n_docs,1])
  doc_pos = tf.concat([doc_i, docs], axis=1)
  doc_emb = mu._shared_doc_embeddings(doc_pos, params,
                                      '/main/doc_emb',
                                      inference=True)
  doc_input = doc_emb
  for i in range(serp_len):
    scores = mu._create_subnetwork(doc_input, params, '/main/score',
                                   reuse_variable_scope=i>0,
                                   inference=True)
    tf.summary.scalar('policy/scores/pos_%d' % i, tf.reduce_mean(scores))
    action, max_ind = mu.select_eps_greedy_action(scores,
                                                  epsilon,
                                                  score_filter)

    select_doc = tf.gather(docs, action)
    select_doc_input = tf.gather(doc_emb, action)
    result['label_%d' % i] = tf.cond(
      tf.less(i, n_docs),
      lambda: tf.gather(labels, action, axis=0),
      lambda: tf.constant([[0]], dtype=tf.int64),
      )
    result['doc_%d' % i] = select_doc

    if i > 0:
      result['max_doc_%d' % i] = tf.gather(docs, max_ind)

    if i < serp_len-1:
      score_filter = tf.where(tf.equal(ind_range, action),
                              neginf,
                              score_filter)

      pbow += select_doc_input
      doc_i += 1
      doc_pos = tf.concat([doc_i, docs], axis=1)
      doc_emb = mu._shared_doc_embeddings(doc_pos, params,
        '/main/doc_emb', reuse_variable_scope=True,
        inference=True)
      doc_input = pbow + doc_emb
  return result

# RNN with Gated Recurrent Unit (GRU)
def gru_model(params, examples, labels, epsilon):
  serp_len = params['serp_len']
  doc_emb_size = params['doc_emb'][-1]
  hidden_state_size = params['hidden_state_size']
  # docs = mu._get_doc_tensors(examples, params, 'main')
  docs = examples['doc_tensors']
  batch_size = docs.shape[0].value
  batch_max_docs = tf.shape(docs)[1]
  docs_per_query = examples['n_docs']

  # if params['context_input']:
  #   to_shuffle = tf.concat([tf.cast(labels, tf.float32), docs], axis=1)
  #   shuffled = tf.random_shuffle(to_shuffle)
  #   labels = tf.cast(tf.slice(shuffled, [0, 0], [-1, 1]), tf.int64)
  #   docs = tf.slice(shuffled, [0, 1], [-1, -1])
  assert not params['context_input'], 'Context not supported for GRU.'

  result = {
    'docs_per_query': docs_per_query,
    }

  doc_emb = mu._shared_doc_embeddings(docs, params,
                                      '/main/doc_emb',
                                      inference=True)


  hidden_init = tf.zeros([batch_size, hidden_state_size])
  # if params['context_input']:
  #   context_gru_fn = ru.get_gru_layer(params, '/main/gru/context',
  #                                     label_network=False,
  #                                     inference=True,
  #                                     reuse_variable_scope=False)
  #   context_init = hidden_init
  #   context = tf.scan(context_gru_fn,
  #                     tf.expand_dims(doc_emb, axis=1), context_init)

  #   hidden_init = tf.gather(context, n_docs-1)

  gru_fn = ru.get_gru_layer(params, '/main/gru',
                                  label_network=False,
                                  inference=True,
                                  reuse_variable_scope=False)

  policy = mu.EpsilonGreedy(epsilon, batch_size, batch_max_docs, docs_per_query)
  hidden_state = hidden_init
   #tf.zeros([n_docs, hidden_state_size])
  serp = []
  serp_labels = []
  serp_ind = []
  for i in range(serp_len):
    hidden_states = tf.tile(hidden_state[:, None, :], [1, batch_max_docs, 1])
    score_input = tf.concat([hidden_states, doc_emb], axis=2)
    scores = mu._create_subnetwork(score_input,
                                   params,
                                   subnetwork_name='/main/scoring',
                                   label_network=False,
                                   reuse_variable_scope=i>0,
                                   inference=True)

    tf.summary.scalar('policy/scores/pos_%d' % i, tf.reduce_mean(scores))
    action = policy.choose(scores)
    serp_ind.append(action)

    nd_ind = tf.stack([tf.range(batch_size, dtype=tf.int64), action], axis=1)
    select_doc = tf.gather_nd(docs, nd_ind)
    select_labels = tf.gather_nd(labels, nd_ind)[:, None]
    
    serp_labels.append(tf.where(
      tf.less(i, docs_per_query),
      select_labels,
      tf.zeros([batch_size, 1], dtype=tf.int32),
      ))
    serp.append(select_doc)

    if i < serp_len-1:
      select_emb = tf.gather_nd(doc_emb, nd_ind)
      hidden_state = gru_fn(hidden_state, select_emb)


  result['serp'] = tf.stack(serp, axis=1)
  result['serp_ind'] = tf.stack(serp_ind, axis=1)
  result['serp_ind'] = tf.Print(result['serp_ind'], serp_ind, 'serp_ind: ')
  result['labels'] = tf.concat(serp_labels, axis=1)
  tf.summary.histogram("label/output", result['labels'])

  # if params['context_input']:
  max_docs = params['max_docs']
  padding = tf.convert_to_tensor([[0, 0], [0, max_docs-batch_max_docs], [0, 0]])
  padded_docs = tf.pad(docs, padding, "CONSTANT")
  padded_docs = tf.reshape(padded_docs, [batch_size, max_docs, docs.shape[2].value])
  result['docs'] = padded_docs

  return result