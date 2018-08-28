import tensorflow as tf
import numpy as np
import model_utils as mu
import rnn_utils as ru

class PositionEpsilonGreedy:
  def __init__(self, serp_len, epsilon, batch_size, batch_max_docs, docs_per_query):
    self.serp_len = serp_len
    self.epsilon = epsilon
    self.n_docs = batch_max_docs
    self.batch_size = batch_size
    self.pos_filter = tf.zeros([batch_size, serp_len])
    n_doc_filter = tf.sequence_mask(docs_per_query[:, 0], batch_max_docs)
    self.doc_filter = tf.where(n_doc_filter,
          tf.zeros([batch_size, batch_max_docs]),
          tf.fill([batch_size, batch_max_docs], np.NINF))

  def max_doc_ind(self, scores):
    return tf.argmax(scores + self.doc_filter, axis=1)

  def max_pos(self, scores):
    return tf.argmax(scores + self.pos_filter, axis=1)

  def choose_doc(self, scores):
    max_ind = self.max_doc_ind(scores[:, :, 0])
    noise = tf.random_uniform([self.batch_size, self.n_docs])
    noise_ind = self.max_doc_ind(noise)

    random_doc = tf.greater_equal(tf.random_uniform([self.batch_size]),
                                   self.epsilon)
    action_ind = tf.where(random_doc,
                     max_ind,
                     noise_ind)

    cur_ind = tf.one_hot(action_ind, self.n_docs,
                         on_value=np.NINF, off_value=0.)
    self.doc_filter += cur_ind
  
    return action_ind

  def choose_pos(self, scores):
    max_pos = self.max_pos(scores)
    noise = tf.random_uniform([self.batch_size, self.serp_len])
    noise_pos = self.max_pos(noise)
    
    random_pos = tf.greater_equal(tf.random_uniform([self.batch_size]),
                                  self.epsilon)

    action_pos = tf.where(random_pos, max_pos, noise_pos)

    # action_pos = tf.Print(action_pos, [max_pos, noise_pos, action_pos], 'pos: ')

    cur_pos = tf.one_hot(action_pos, self.serp_len,
                         on_value=np.NINF, off_value=0.)
    self.pos_filter += cur_pos
  
    return action_pos

def mean_summary(params, name, values, stats_ops):
    if params['evaluation']:
      mean, update = tf.metrics.mean(tf.reduce_mean(values))
      stats_ops.append(update)
    else:
      mean = tf.reduce_mean(values)
    tf.summary.scalar(name, mean)

def model(params, examples, labels, epsilon, stats_ops):
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
                                      '/main/doc_emb',
                                      inference=True)

  hidden_init = tf.zeros([batch_size, hidden_state_size])

  gru_fn = ru.get_gru_layer(params, '/main/gru',
                            label_network=False,
                            inference=True,
                            reuse_variable_scope=False)

  policy = PositionEpsilonGreedy(serp_len, epsilon, batch_size,
                                 batch_max_docs, docs_per_query)

  hidden_state = hidden_init
  serp = []
  serp_pos = []
  serp_labels = []
  serp_ind = []
  for i in range(serp_len):
    hidden_states = tf.tile(hidden_state[:, None, :], [1, batch_max_docs, 1])
    score_input = tf.concat([hidden_states, doc_emb], axis=2)
    doc_scores = mu._create_subnetwork(score_input,
                                       params,
                                       subnetwork_name='/main/scoring/doc',
                                       label_network=False,
                                       reuse_variable_scope=i>0,
                                       inference=True,
                                       n_output=1)

    action_ind = policy.choose_doc(doc_scores)

    ind_nd = tf.stack([tf.range(batch_size, dtype=tf.int64), action_ind],
                      axis=1)

    select_doc = tf.gather_nd(docs, ind_nd)

    serp.append(select_doc) 
    serp_ind.append(action_ind)

    select_emb = tf.gather_nd(doc_emb, ind_nd)
    pos_input = tf.concat([hidden_state, select_emb], axis=1)
    pos_scores = mu._create_subnetwork(pos_input,
                                       params,
                                       subnetwork_name='/main/scoring/pos',
                                       label_network=False,
                                       reuse_variable_scope=i>0,
                                       inference=True,
                                       n_output=10)
    # pos_scores = tf.Print(pos_scores, [pos_scores[0,x] for x in range(10)], 'scores %d: ' % i)

    mean_summary(params, 'policy_%d/doc' % i,
                 tf.gather_nd(doc_scores, ind_nd), stats_ops)
    for j in range(serp_len):
      mean_summary(params, 'policy_%d/pos_%d' % (i, j),
                   pos_scores[:, j], stats_ops)

    action_pos = policy.choose_pos(pos_scores)
    # if i == 0:
    #   action_pos = tf.Print(action_pos, [pos_scores[0,x] for x in range(10)], 'pos_scores: ')
    #   action_pos = tf.Print(action_pos, [action_pos], 'pos: ')

    in_doc = tf.less(i, docs_per_query[:, 0])
    serp_labels.append(tf.where(
      in_doc,
      tf.gather_nd(labels, ind_nd),
      tf.zeros([batch_size], dtype=tf.int32),
      ))
    serp_pos.append(tf.where(
      in_doc,
      action_pos,
      tf.fill([batch_size], tf.cast(serp_len, dtype=tf.int64)),
      ))

    if i < serp_len-1:
      a_pos = tf.cast(action_pos, tf.float32)[:, None]
      gru_input = tf.concat([select_emb, a_pos], axis=1)
      hidden_state = gru_fn(hidden_state, gru_input)

  pos_order = tf.stack(serp_pos, axis=1)
  _, order_ind = tf.nn.top_k(-pos_order, serp_len)
  unordered_labels = tf.stack(serp_labels, axis=1)
  batch_ind_nd = tf.tile(tf.range(batch_size)[:, None], [1, serp_len])
  order_ind_nd = tf.stack([tf.reshape(batch_ind_nd, [-1]),
                           tf.reshape(order_ind, [-1])],
                           axis=1)
  ordered_labels = tf.gather_nd(unordered_labels, order_ind_nd)
  ordered_labels = tf.reshape(ordered_labels, [batch_size, serp_len])
  
  result['serp'] = tf.stack(serp, axis=1)
  result['serp_ind'] = tf.stack(serp_ind, axis=1)
  result['labels'] = ordered_labels
  result['select_order_labels'] = unordered_labels
  result['pos_order'] = pos_order

  max_docs = params['max_docs']
  padding = tf.convert_to_tensor([[0, 0], [0, max_docs-batch_max_docs], [0, 0]])
  padded_docs = tf.pad(docs, padding, "CONSTANT")
  padded_docs = tf.reshape(padded_docs, [batch_size, max_docs, docs.shape[2].value])
  result['docs'] = padded_docs
  return result

def max_train_doc_pos(params, hidden_states, serp_emb, serp_len, doc_col,
                     serp_ind, pos_order, docs_per_query, max_n_docs):
  max_n_docs = params['max_docs']
  hidden_state_size = params['hidden_state_size']
  batch_size = hidden_states.shape[0]

  doc_states = tf.tile(hidden_states[:, 1:, None, :], [1, 1, max_n_docs, 1])
  max_col = tf.tile(doc_col[:, None, :, :], [1, serp_len-1, 1, 1])
  doc_input = tf.concat([doc_states, max_col], axis=3)

  doc_scores = mu._create_subnetwork(doc_input,
                                     params,
                                     subnetwork_name='/main/scoring/doc',
                                     label_network=False,
                                     reuse_variable_scope=True,
                                     inference=False)[:, :, :, 0]

  doc_filter = tf.one_hot(serp_ind[:, :-1], max_n_docs,
                          on_value=np.NINF, off_value=0.)
  doc_filter = tf.cumsum(doc_filter, axis=1)
  n_doc_filter = tf.sequence_mask(docs_per_query, max_n_docs)
  n_doc_filter = tf.where(n_doc_filter,
                        tf.zeros(n_doc_filter.shape),
                        tf.fill(n_doc_filter.shape, np.NINF))
  max_doc_ind = tf.argmax(doc_scores + doc_filter + n_doc_filter, axis=2)

  pos_input = tf.concat([hidden_states, serp_emb], axis=2)
  pos_scores = mu._create_subnetwork(pos_input,
                                     params,
                                     subnetwork_name='/main/scoring/pos',
                                     label_network=False,
                                     reuse_variable_scope=True,
                                     inference=False,
                                     n_output=10)

  pos_filter = tf.one_hot(pos_order[:,:-1], serp_len,
                          on_value=np.NINF, off_value=0.)
  pos_filter = tf.cumsum(pos_filter, axis=1)
  pos_filter = tf.concat([tf.zeros([batch_size, 1, serp_len]),
                          pos_filter], axis=1)

  max_pos = tf.argmax(pos_scores + pos_filter, axis=2)
  
  return max_doc_ind, max_pos

def get_label_scores(params, replay, max_doc_ind, max_pos):
  serp_len = params['serp_len']
  all_docs = replay['docs']
  batch_docs = replay['serp']
  batch_pos = replay['pos_order']
  max_n_docs = params['max_docs']
  batch_size = all_docs.shape[0]
  hidden_state_size = params['hidden_state_size']

  init_hidden = tf.zeros([batch_size, hidden_state_size])

  doc_emb = mu._shared_doc_embeddings(batch_docs, params,
                                      '/label/doc_emb',
                                      label_network=True,
                                      inference=True,
                                      reuse_variable_scope=False)

  batch_ind_nd = tf.tile(tf.range(batch_size, dtype=tf.int64)[:, None], [1, serp_len-1])
  doc_ind_nd = tf.stack([tf.reshape(batch_ind_nd, [-1]),
                         tf.reshape(max_doc_ind, [-1]),
                        ], axis=1)
  max_docs = tf.gather_nd(all_docs, doc_ind_nd)
  max_docs = tf.reshape(max_docs, [batch_size, serp_len-1, all_docs.shape[2]])

  max_emb = mu._shared_doc_embeddings(max_docs, params,
                                      '/label/doc_emb',
                                      label_network=True,
                                      inference=True,
                                      reuse_variable_scope=True)

  serp_emb = tf.transpose(doc_emb, [1, 0, 2])
  gru = ru.get_gru_layer(params, '/label/gru',
                               label_network=True,
                               inference=True,
                               reuse_variable_scope=False)

  pos = tf.cast(batch_pos, tf.float32)[:, :-1, None]
  pos = tf.transpose(pos, [1, 0, 2])
  gru_input = tf.concat([serp_emb[:-1, :, :], pos], axis=2)
  hidden_states = tf.scan(gru, gru_input, init_hidden)
  hidden_states = tf.transpose(hidden_states, [1, 0, 2])

  score_input = tf.concat([hidden_states, max_emb], axis=2)
  doc_scores = mu._create_subnetwork(score_input,
                                     params,
                                     subnetwork_name='/label/scoring/doc',
                                     label_network=True,
                                     inference=True,
                                     reuse_variable_scope=False)[:,:,0]

  pos_states = tf.concat([init_hidden[:, None, :], hidden_states], axis=1)
  pos_input = tf.concat([pos_states, doc_emb], axis=2)
  pos_scores = mu._create_subnetwork(pos_input,
                                     params,
                                     subnetwork_name='/label/scoring/pos',
                                     label_network=True,
                                     inference=True,
                                     reuse_variable_scope=False,
                                     n_output=10)

  batch_ind_nd = tf.tile(tf.range(batch_size, dtype=tf.int64)[:, None], [1, serp_len])
  serp_ind_nd = tf.tile(tf.range(serp_len, dtype=tf.int64)[None, :], [batch_size, 1])
  pos_ind_nd = tf.stack([tf.reshape(batch_ind_nd, [-1]),
                         tf.reshape(serp_ind_nd, [-1]),
                         tf.reshape(max_pos, [-1]),
                        ],axis=1)
  pos_scores = tf.gather_nd(pos_scores, pos_ind_nd)
  pos_scores = tf.reshape(pos_scores, [batch_size, serp_len])

  return doc_scores, pos_scores

def loss(params, replay, rewards, doc_rewards):
  serp_len = params['serp_len']
  visible_dropout = params['visible_dropout']
  docs_per_query = replay['docs_per_query']
  batch_docs = replay['serp']
  batch_pos = replay['pos_order']
  max_n_docs = params['max_docs']
  batch_size = batch_docs.shape[0]
  hidden_state_size = params['hidden_state_size']
  doc_level_rewards = params['doc_rewards']

  mask = tf.squeeze(tf.sequence_mask(docs_per_query, serp_len), axis=1)

  init_hidden = tf.zeros([batch_size, hidden_state_size])

  drop_col = tf.nn.dropout(replay['docs'], visible_dropout)

  doc_col = mu._shared_doc_embeddings(drop_col, params,
                                        '/main/doc_emb',
                                        inference=False,
                                        reuse_variable_scope=True)

  drop_docs = tf.nn.dropout(batch_docs, visible_dropout)

  doc_emb = mu._shared_doc_embeddings(drop_docs, params,
                                      '/main/doc_emb',
                                      inference=False,
                                      reuse_variable_scope=True)

  serp_emb = tf.transpose(doc_emb, [1, 0, 2])
  gru = ru.get_gru_layer(params, '/main/gru',
                               label_network=False,
                               inference=False,
                               reuse_variable_scope=True)

  pos = tf.cast(batch_pos, tf.float32)[:, :, None]
  pos = tf.transpose(pos, [1, 0, 2])
  gru_input = tf.concat([serp_emb, pos], axis=2)
  hidden_states = tf.scan(gru, gru_input, init_hidden)
  hidden_states = tf.concat([init_hidden[None, :, :],
                            hidden_states[:-1, :, :]], axis=0)
  hidden_states = tf.transpose(hidden_states, [1, 0, 2])

  score_input = tf.concat([hidden_states, doc_emb], axis=2)
  doc_scores = mu._create_subnetwork(score_input,
                                     params,
                                     subnetwork_name='/main/scoring/doc',
                                     label_network=False,
                                     reuse_variable_scope=True,
                                     inference=False)[:, :, 0]
  pos_scores = mu._create_subnetwork(score_input,
                                     params,
                                     subnetwork_name='/main/scoring/pos',
                                     label_network=False,
                                     reuse_variable_scope=True,
                                     inference=False,
                                     n_output=serp_len)

  batch_pos_filtered = tf.where(mask,
                                batch_pos,
                                tf.zeros_like(batch_pos))
  batch_ind_nd = tf.tile(tf.range(batch_size, dtype=tf.int64)[:, None], [1, serp_len])
  serp_ind_nd = tf.tile(tf.range(serp_len, dtype=tf.int64)[:, None], [batch_size, 1])
  pos_ind_nd = tf.stack([tf.reshape(batch_ind_nd, [-1]),
                         tf.reshape(serp_ind_nd, [-1]),
                         tf.reshape(batch_pos_filtered, [-1]),
                        ], axis=1)
  pos_scores = tf.gather_nd(pos_scores, pos_ind_nd)
  pos_scores = tf.reshape(pos_scores, [batch_size, serp_len])

  if not doc_level_rewards:
    unfiltered_mc_loss = (rewards-pos_scores)**2 + (rewards-doc_scores)**2
  else:
    cum_rewards = tf.cumsum(doc_rewards, axis=1, reverse=True)
    unfiltered_mc_loss = (cum_rewards-pos_scores)**2 + (cum_rewards-doc_scores)**2

  max_doc_ind, max_pos = max_train_doc_pos(params, hidden_states,
                                        doc_emb, serp_len, doc_col,
                                        replay['serp_ind'], batch_pos,
                                        docs_per_query, max_n_docs)

  label_doc_scores, q_pos_values = get_label_scores(params, replay, max_doc_ind, max_pos)

  if not doc_level_rewards:
    q_doc_values = tf.concat([label_doc_scores, rewards], axis=1)
    end_mask = tf.equal(docs_per_query-1,
                        tf.range(serp_len)[None, :])
    reward_tile = tf.tile(rewards, [1, serp_len])
    q_doc_values = tf.where(end_mask, reward_tile, q_doc_values)
  else:
    zero_end = tf.zeros([batch_size, 1])
    q_doc_values = tf.concat([label_doc_scores, zero_end], axis=1)
    end_mask = tf.equal(docs_per_query-1,
                        tf.range(serp_len)[None, :])
    q_doc_values = tf.where(end_mask, tf.zeros_like(q_doc_values), q_doc_values)
    q_doc_values += doc_rewards


    # q_doc_values = tf.Print(q_doc_values, [batch_pos[0,x] for x in range(10)], 'pos: ')
    # q_doc_values = tf.Print(q_doc_values, [pos_scores[0,x] for x in range(10)], 'pos_scores: ')
    # q_doc_values = tf.Print(q_doc_values, [q_doc_values[0,x] for x in range(10)], 'q-values: ')
    # q_doc_values = tf.Print(q_doc_values, [doc_rewards[0,x] for x in range(10)], 'doc_rewards: ')


  unfiltered_doc_loss = (doc_scores - q_pos_values)**2
  unfiltered_pos_loss = (pos_scores - q_doc_values)**2
  unfiltered_dqn_loss = unfiltered_doc_loss + unfiltered_pos_loss

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

  tf.summary.scalar('monte_carlo/loss', mc_loss)

  tf.summary.scalar('DQN/loss', dqn_loss)

  tf.summary.scalar('DQN/max_doc_scores', tf.reduce_mean(label_doc_scores))
  tf.summary.scalar('DQN/max_pos_scores', tf.reduce_mean(q_pos_values))

  return mc_loss, dqn_loss