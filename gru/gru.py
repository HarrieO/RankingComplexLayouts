import tensorflow as tf
import model_utils as mu
import rnn_utils as ru
import numpy as np

def model(params, examples, labels, epsilon):
  serp_len = params['serp_len']
  doc_emb_size = params['doc_emb'][-1]
  hidden_state_size = params['hidden_state_size']
  docs = examples['doc_tensors']
  batch_size = docs.shape[0].value
  batch_max_docs = tf.shape(docs)[1]
  docs_per_query = examples['n_docs']

  if params['context_input']:
    to_shuffle = tf.concat([tf.cast(labels[:, :, None], tf.float32), docs], axis=2)
    shuffled = tf.random_shuffle(tf.transpose(to_shuffle, [1, 0, 2]))
    shuffled = tf.transpose(shuffled, [1, 0, 2])
    labels = tf.cast(tf.slice(shuffled, [0, 0, 0], [-1, -1, 1]), tf.int32)
    labels = labels[:, :, 0]
    docs = tf.slice(shuffled, [0, 0, 1], [-1, -1, -1])
  # assert not params['context_input'], 'Context not supported for GRU.'

  result = {
    'docs_per_query': docs_per_query,
    }

  doc_emb = mu._shared_doc_embeddings(docs, params,
                                      '/main/doc_emb',
                                      inference=True)


  hidden_init = tf.zeros([batch_size, hidden_state_size])
  if params['context_input']:
    context_gru_fn = ru.get_gru_layer(params, '/main/gru/context',
                                      label_network=False,
                                      inference=True,
                                      reuse_variable_scope=False)
    scan_input = tf.transpose(doc_emb, [1, 0, 2])
    context = tf.scan(context_gru_fn, scan_input, hidden_init)

    ind_nd = tf.concat([docs_per_query-1, tf.range(batch_size)[:, None]], axis=1)
    hidden_init = tf.gather_nd(context, ind_nd)

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


    action = policy.choose(scores)
    serp_ind.append(action)

    nd_ind = tf.stack([tf.range(batch_size, dtype=tf.int64), action], axis=1)
    select_doc = tf.gather_nd(docs, nd_ind)
    select_labels = tf.gather_nd(labels, nd_ind)[:, None]
    tf.summary.scalar('policy/scores/pos_%d' % i,
                      tf.reduce_mean(tf.gather_nd(scores, nd_ind)))

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
  result['labels'] = tf.concat(serp_labels, axis=1)
  tf.summary.histogram("label/output", result['labels'])

  # if params['context_input']:
  max_docs = params['max_docs']
  padding = tf.convert_to_tensor([[0, 0], [0, max_docs-batch_max_docs], [0, 0]])
  padded_docs = tf.pad(docs, padding, "CONSTANT")
  padded_docs = tf.reshape(padded_docs, [batch_size, max_docs, docs.shape[2].value])
  result['docs'] = padded_docs

  return result

def max_train_docs(params, replay, hidden_states, doc_col):
  serp_len = params['serp_len']
  max_n_docs = params['max_docs']
  batch_size = params['replay_batch']
  serp_ind = replay['serp_ind']
  docs_per_query = replay['docs_per_query']

  score_filter = tf.one_hot(serp_ind[:, :-1], max_n_docs,
                            on_value=np.NINF, off_value=0.)
  score_filter = tf.cumsum(score_filter, axis=1)

  n_doc_filter = tf.sequence_mask(docs_per_query[:, 0], max_n_docs)
  score_filter += tf.where(n_doc_filter,
                           tf.zeros([batch_size, max_n_docs]),
                           tf.fill([batch_size, max_n_docs], np.NINF))[:, None, :]

  hidden_states = tf.transpose(hidden_states[1:, :, :], [1, 0, 2])

  tiled_states = tf.tile(hidden_states[:, :, None, :], [1, 1, max_n_docs, 1])
  tiled_docs = tf.tile(doc_col[:, None, :, :], [1, serp_len-1, 1, 1])
  score_input = tf.concat([tiled_states, tiled_docs], axis=3)

  scores = mu._create_subnetwork(score_input,
                                 params,
                                 subnetwork_name='/main/scoring',
                                 label_network=False,
                                 reuse_variable_scope=True,
                                 inference=False)[:, :, :, 0]
  return tf.argmax(scores + score_filter, axis=2)

def get_label_scores(params, replay, max_train_ind):
  serp_len = params['serp_len']
  batch_size = replay['serp'].shape[0]
  hidden_state_size = params['hidden_state_size']
  docs_per_query = replay['docs_per_query']
  doc_col = replay['docs']
  batch_ind = tf.tile(tf.range(batch_size, dtype=tf.int64)[:, None], [1, serp_len-1])
  max_ind = tf.stack([tf.reshape(batch_ind, [-1]), tf.reshape(max_train_ind, [-1])], axis=1)
  max_docs = tf.gather_nd(doc_col, max_ind)
  max_docs = tf.reshape(max_docs, [batch_size, serp_len-1, -1])

  max_emb = mu._shared_doc_embeddings(max_docs, params,
                                        '/label/doc_emb',
                                        inference=True,
                                        label_network=True,
                                        reuse_variable_scope=False)

  doc_emb = mu._shared_doc_embeddings(replay['serp'][:, :-1], params,
                                            '/label/doc_emb',
                                            inference=True,
                                            label_network=True,
                                            reuse_variable_scope=True)


  gru = ru.get_gru_layer(params, '/label/gru',
                               label_network=True,
                               inference=True,
                               reuse_variable_scope=False)

  init_hidden = tf.zeros([batch_size, hidden_state_size])
  if params['context_input']:
    emb_col = mu._shared_doc_embeddings(doc_col, params,
                                        '/label/doc_emb',
                                        inference=True,
                                        label_network=True,
                                        reuse_variable_scope=True)
    context_gru_fn = ru.get_gru_layer(params, '/label/gru/context',
                                      label_network=True,
                                      inference=True,
                                      reuse_variable_scope=False)
    scan_input = tf.transpose(emb_col, [1, 0, 2])
    context = tf.scan(context_gru_fn, scan_input, init_hidden)

    ind_nd = tf.concat([docs_per_query-1, tf.range(batch_size)[:, None]], axis=1)
    init_hidden = tf.gather_nd(context, ind_nd)

  serp_emb = tf.transpose(doc_emb, [1, 0, 2])
  hidden_states = tf.scan(gru, serp_emb, init_hidden)
  hidden_states = tf.transpose(hidden_states, [1, 0, 2])

  score_input = tf.concat([hidden_states, max_emb], axis=2)
  scores = mu._create_subnetwork(score_input,
                                 params,
                                 subnetwork_name='/label/scoring',
                                 label_network=True,
                                 reuse_variable_scope=False,
                                 inference=True)
  return tf.stop_gradient(scores)[:, :, 0]


def loss(params, replay, rewards, doc_rewards):
  serp_len = params['serp_len']
  visible_dropout = params['visible_dropout']
  docs_per_query = replay['docs_per_query']
  batch_docs = replay['serp']
  max_n_docs = params['max_docs']
  batch_size = params['replay_batch']
  hidden_state_size = params['hidden_state_size']
  doc_level_rewards = params['doc_rewards']

  drop_col = tf.nn.dropout(replay['docs'], visible_dropout)

  doc_col = mu._shared_doc_embeddings(drop_col, params,
                                        '/main/doc_emb',
                                        inference=False,
                                        reuse_variable_scope=True)

  init_hidden = tf.zeros([batch_size, hidden_state_size])
  if params['context_input']:
    context_gru_fn = ru.get_gru_layer(params, '/main/gru/context',
                                      label_network=False,
                                      inference=False,
                                      reuse_variable_scope=True)
    scan_input = tf.transpose(doc_col, [1, 0, 2])
    context = tf.scan(context_gru_fn, scan_input, init_hidden)

    ind_nd = tf.concat([docs_per_query-1, tf.range(batch_size)[:, None]], axis=1)
    init_hidden = tf.gather_nd(context, ind_nd)

  drop_docs = tf.nn.dropout(batch_docs, visible_dropout)

  doc_emb = mu._shared_doc_embeddings(drop_docs, params,
                                      '/main/doc_emb',
                                      label_network=False,
                                      inference=False,
                                      reuse_variable_scope=True)

  serp_emb = tf.transpose(doc_emb, [1, 0, 2])
  gru = ru.get_gru_layer(params, '/main/gru',
                                 label_network=False,
                                 inference=False,
                                 reuse_variable_scope=True)

  hidden_states = tf.scan(gru, serp_emb[:-1, :, :], init_hidden)
  hidden_states = tf.concat([init_hidden[None, :, :], hidden_states], axis=0)
  score_input = tf.concat([hidden_states, serp_emb], axis=2)

  scores = mu._create_subnetwork(score_input,
                                 params,
                                 subnetwork_name='/main/scoring',
                                 label_network=False,
                                 reuse_variable_scope=True,
                                 inference=False)


  scores = tf.transpose(scores, [1,0,2])[:, :, 0]

  if not doc_level_rewards:
    unfiltered_mc_loss = (rewards-scores)**2.
  else:
    cum_rewards = tf.cumsum(doc_rewards, axis=1, reverse=True)
    unfiltered_mc_loss = (rewards-scores)**2.

  max_train_ind = max_train_docs(params, replay, hidden_states, doc_col)
  label_scores = get_label_scores(params, replay, max_train_ind)
  if not doc_level_rewards:
    q_values = tf.concat([label_scores, rewards], axis=1)

    end_mask = tf.equal(docs_per_query-1, tf.range(serp_len)[None,:])
    reward_tile = tf.tile(rewards, [1, serp_len])
    q_values = tf.where(end_mask, reward_tile, q_values)

    unfiltered_dqn_loss = (scores - q_values)**2.
  else:
    zero_end = tf.zeros([batch_size, 1])
    q_values = tf.concat([label_scores, zero_end], axis=1)
    end_mask = tf.equal(docs_per_query-1, tf.range(serp_len)[None,:])
    q_values = tf.where(end_mask, tf.zeros_like(q_values), q_values)
    q_values += doc_rewards

    unfiltered_dqn_loss = (scores - q_values)**2.

  mask = tf.squeeze(tf.sequence_mask(docs_per_query, serp_len), axis=1)
  query_denom = tf.cast(docs_per_query[:, 0], tf.float32)
  query_denom = tf.minimum(query_denom, serp_len)
  query_denom = tf.maximum(query_denom, tf.ones_like(query_denom))

  filtered_mc_loss = tf.where(mask,
                              unfiltered_mc_loss,
                              tf.zeros_like(unfiltered_mc_loss))
  mc_loss = tf.reduce_mean(tf.reduce_sum(filtered_mc_loss, axis=1)/query_denom)

  filtered_dqn_loss = tf.where(mask,
                               unfiltered_dqn_loss,
                               tf.zeros_like(unfiltered_dqn_loss))
  dqn_loss = tf.reduce_mean(tf.reduce_sum(filtered_dqn_loss, axis=1)/query_denom)

  # dqn_loss = tf.Print(dqn_loss, [filtered_dqn_loss[0, j] for j in range(10)], 'dqn loss: ')
  # dqn_loss = tf.Print(dqn_loss, [query_denom[0]], 'denom: ')
  # dqn_loss = tf.Print(dqn_loss, [dqn_loss], 'total loss: ')


  tf.summary.scalar('monte_carlo/loss', mc_loss)
  tf.summary.scalar('DQN/loss', dqn_loss)

  tf.summary.scalar('DQN/double_max_scores', tf.reduce_mean(label_scores))

  return mc_loss, dqn_loss