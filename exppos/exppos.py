import tensorflow as tf
import numpy as np
import model_utils as mu
import rnn_utils as ru

class PositionEpsilonGreedy:
  def __init__(self, serp_len, epsilon, n_docs, split_explore=True):
    self.serp_len = serp_len
    self.epsilon = epsilon
    self.n_docs = n_docs
    self.split_explore = split_explore
    self.original_doc_filter = tf.zeros([n_docs])
    self.original_pos_filter = tf.zeros([1, serp_len])
    self.doc_filter = self.original_doc_filter
    self.pos_filter = self.original_pos_filter

    # s_mask = tf.sequence_mask(n_docs, serp_len)
    # short_filter = tf.where(s_mask,
    #          tf.zeros_like(s_mask, dtype=tf.float32),
    #          tf.fill(tf.shape(s_mask), np.NINF))
    # self.pos_filter += tf.expand_dims(short_filter, axis=0)

  def max_doc_ind(self, scores):
    return tf.argmax(scores + self.doc_filter, axis=0)

  def max_pos(self, scores):
    values, indices =  tf.nn.top_k(scores + self.pos_filter, k=1)
    return tf.squeeze(values, axis=1), tf.squeeze(indices, axis=1)

  def choose(self, scores):
    pos_values, max_pos_ind = self.max_pos(scores)
    max_ind = self.max_doc_ind(pos_values)
    max_pos = tf.gather(max_pos_ind, max_ind)

    noise_scores = tf.random_uniform([1, self.serp_len])
    _, noise_pos_ind = self.max_pos(noise_scores)
    noise_values = tf.random_uniform(tf.shape(pos_values))
    noise_ind = self.max_doc_ind(noise_values)
    noise_pos = noise_pos_ind[0]

    random_doc = tf.greater_equal(tf.random_uniform([]),
                                   self.epsilon)
    if self.split_explore:
      random_pos = tf.greater_equal(tf.random_uniform([]),
                                   self.epsilon)
    else:
      random_pos = random_doc

    action_ind = tf.cond(random_doc,
                     lambda: max_ind,
                     lambda: noise_ind)
    action_ind = tf.reshape(action_ind, [1])
    action_pos = tf.cond(random_pos,
                     lambda: max_pos,
                     lambda: noise_pos)
    action_pos = tf.reshape(action_pos, [1])

    cur_pos = tf.one_hot(action_pos, self.serp_len,
                         on_value=np.NINF, off_value=0.)
    self.pos_filter += cur_pos

    cur_ind = tf.one_hot(action_ind, self.n_docs,
                         on_value=np.NINF, off_value=0.)
    self.doc_filter += tf.squeeze(cur_ind, axis=0)
  
    return action_ind, action_pos

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
  docs = mu._get_doc_tensors(examples, params, 'main')

  result = {}

  n_docs = tf.shape(docs)[0]
  result['docs_per_query'] = n_docs

  doc_emb = mu._shared_doc_embeddings(docs, params,
                                      '/main/doc_emb',
                                      inference=True)

  hidden_init = tf.zeros([1, hidden_state_size])

  gru_fn = ru.get_gru_layer(params, '/main/gru',
                            label_network=False,
                            inference=True,
                            reuse_variable_scope=False)

  policy = PositionEpsilonGreedy(serp_len, epsilon, n_docs)

  hidden_state = hidden_init #tf.zeros([n_docs, hidden_state_size])
  serp = []
  serp_pos = []
  serp_labels = []
  serp_ind = []
  for i in range(serp_len):
    hidden_states = tf.tile(hidden_state, [n_docs, 1])
    score_input = tf.concat([hidden_states, doc_emb], axis=1)
    scores = mu._create_subnetwork(score_input,
                                   params,
                                   subnetwork_name='/main/scoring',
                                   label_network=False,
                                   reuse_variable_scope=i>0,
                                   inference=True)
    for j in range(serp_len):
      mean_summary(params, 'policy_%d/pos_%d' % (i, j), scores[:,j], stats_ops)

    action_ind, action_pos = policy.choose(scores)
    select_doc = tf.gather(docs, action_ind)

    serp.append(select_doc) 
    serp_ind.append(action_ind)

    in_doc = tf.less(i, n_docs)
    serp_labels.append(tf.cond(
      in_doc,
      lambda: tf.gather(labels, action_ind, axis=0),
      lambda: tf.constant([[0]], dtype=tf.int64),
      ))
    serp_labels[-1].set_shape([1, 1])
    serp_pos.append(tf.cond(
      in_doc,
      lambda: tf.expand_dims(action_pos, axis=1),
      lambda: tf.constant([[serp_len]], dtype=tf.int32),
      ))
    serp_pos[-1].set_shape([1, 1])


    if i < serp_len-1:
      a_pos = tf.expand_dims(tf.cast(action_pos, tf.float32), axis=1)
      a_doc = tf.gather(doc_emb, action_ind)
      gru_input = tf.concat([a_doc, a_pos], axis=1)
      hidden_state = gru_fn(hidden_state, gru_input)

  pos_order = tf.concat(serp_pos, axis=1)

  order_ind = tf.nn.top_k(-pos_order, serp_len)[1]
  # order_ind.set_shape()
  unordered_labels = tf.squeeze(tf.concat(serp_labels, axis=1), axis=0)
  ordered_labels = tf.gather(unordered_labels, order_ind)

  result['serp'] = tf.stack(serp, axis=1)
  result['serp_ind'] = tf.stack(serp_ind, axis=1)
  result['serp_doc'] = tf.stack(serp_ind, axis=1)
  result['labels'] = ordered_labels
  result['select_order_labels'] = unordered_labels[None, :]
  # pos_order = tf.Print(pos_order, [unordered_labels[i] for i in range(10)], 'unordered: ')
  # pos_order = tf.Print(pos_order, [pos_order[0, i] for i in range(10)], 'reranking: ')
  # pos_order = tf.Print(pos_order, [result['labels'][0, i] for i in range(10)], 'ordered: ')
  # pos_order = tf.Print(pos_order, [n_docs], '                        ')
  result['pos_order'] = pos_order

  
  # tf.summary.histogram("label/output", result['labels'])

  # if params['context_input']:
  max_docs = params['max_docs']
  padding = tf.convert_to_tensor([[0, max_docs-n_docs], [0, 0]])
  padded_docs = tf.pad(docs, padding, "CONSTANT")
  padded_docs = tf.reshape(padded_docs, [1, max_docs, docs.shape[1].value])
  result['docs'] = padded_docs
  return result

def max_train_filter(params, hidden_states, serp_len, doc_col,
                     serp_ind, pos_order, docs_in_query, max_n_docs):
  max_n_docs = params['max_docs']
  hidden_states = tf.transpose(hidden_states, [1, 0, 2])
  max_states = tf.expand_dims(hidden_states[:, :serp_len-1, :], axis=2)
  max_states = tf.tile(max_states, [1, 1, max_n_docs, 1])
  max_col = tf.expand_dims(doc_col, axis=1)
  max_col = tf.tile(max_col, [1, serp_len-1, 1, 1])
  score_input = tf.concat([max_states, max_col], axis=3)

  scores = mu._create_subnetwork(score_input,
                                 params,
                                 subnetwork_name='/main/scoring',
                                 label_network=False,
                                 reuse_variable_scope=True,
                                 inference=False)
  pos_filter = tf.expand_dims(pos_order[:, :-1], axis=2)
  pos_filter = tf.one_hot(pos_filter, serp_len,
                          on_value=np.NINF, off_value=0.)
  pos_filter = tf.cumsum(pos_filter, axis=1)
  doc_filter = tf.one_hot(serp_ind[:, :-1], max_n_docs,
                          on_value=np.NINF, off_value=0.)
  doc_filter = tf.expand_dims(tf.cumsum(doc_filter, axis=1), axis=3)
  n_doc_filter = tf.sequence_mask(docs_in_query, max_n_docs)
  n_doc_filter = tf.where(n_doc_filter,
                          tf.zeros_like(n_doc_filter, dtype=tf.float32),
                          tf.fill(n_doc_filter.shape, np.NINF))
  doc_filter = doc_filter + n_doc_filter[:, :, :, None]
  score_filter = pos_filter + doc_filter

  pos_scores, pos_ind = tf.nn.top_k(scores + score_filter, k=1)
  doc_ind = tf.argmax(pos_scores, axis=2)[:, :, 0]

  unfilter_pos = tf.one_hot(pos_ind[:, :, :, 0], serp_len)
  doc_filter = tf.one_hot(doc_ind, max_n_docs)

  return doc_filter[:,:,:,None] * unfilter_pos

def get_label_scores(params, replay):
  serp_len = params['serp_len']
  all_docs = replay['docs']
  batch_docs = replay['serp']
  batch_pos = replay['pos_order']
  max_n_docs = params['max_docs']
  n_docs = batch_docs.shape[0]
  hidden_state_size = params['hidden_state_size']

  init_hidden = tf.zeros([n_docs, hidden_state_size])

  doc_col = mu._shared_doc_embeddings(all_docs, params,
                                      '/label/doc_emb',
                                      label_network=True,
                                      inference=True,
                                      reuse_variable_scope=False)

  doc_emb = mu._shared_doc_embeddings(batch_docs[:,:-1,:], params,
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
  gru_input = tf.concat([serp_emb, pos], axis=2)
  hidden_states = tf.scan(gru, gru_input, init_hidden)
  hidden_states = tf.transpose(hidden_states, [1, 0, 2])

  tiled_states = tf.tile(hidden_states[:,:,None,:], [1,1,max_n_docs,1])
  tiled_docs = tf.tile(doc_col[:,None,:,:], [1,serp_len-1,1,1])

  score_input = tf.concat([tiled_states, tiled_docs], axis=3)
  return mu._create_subnetwork(score_input,
                               params,
                               subnetwork_name='/label/scoring',
                               label_network=True,
                               inference=True,
                               reuse_variable_scope=False)


def loss(params, replay, rewards):
  serp_len = params['serp_len']
  visible_dropout = params['visible_dropout']
  docs_in_query = replay['docs_per_query']
  batch_docs = replay['serp']
  batch_pos = replay['pos_order']
  max_n_docs = params['max_docs']
  n_docs = batch_docs.shape[0]
  hidden_state_size = params['hidden_state_size']

  drop_col = tf.nn.dropout(replay['docs'], visible_dropout)

  doc_col = mu._shared_doc_embeddings(drop_col, params,
                                        '/main/doc_emb',
                                        inference=False,
                                        reuse_variable_scope=True)
  init_hidden = tf.zeros([n_docs, hidden_state_size])

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

  pos = tf.expand_dims(tf.cast(batch_pos, tf.float32), axis=2)
  pos = tf.transpose(pos, [1, 0, 2])
  gru_input = tf.concat([serp_emb, pos], axis=2)
  hidden_states = tf.scan(gru, gru_input, init_hidden)
  score_states = tf.concat([init_hidden[None, :, :],
                             hidden_states[:-1, :, :]], axis=0)
  score_input = tf.concat([score_states, serp_emb], axis=2)
  pos_scores = mu._create_subnetwork(score_input,
                                 params,
                                 subnetwork_name='/main/scoring',
                                 label_network=False,
                                 reuse_variable_scope=True,
                                 inference=False)
  pos_scores = tf.transpose(pos_scores, [1, 0, 2])
  pos_filter = tf.one_hot(batch_pos, serp_len)

  scores = tf.reduce_sum(pos_scores * pos_filter, axis=2)
  unfiltered_mc_loss = (rewards-scores)**2

  max_filter = max_train_filter(params, hidden_states, serp_len,
                                doc_col, replay['serp_ind'],
                                batch_pos, docs_in_query, max_n_docs)

  label_scores = get_label_scores(params, replay)

  double_max_scores = tf.reduce_sum(max_filter*label_scores, axis=[2,3])
  q_values = tf.concat([double_max_scores, rewards], axis=1)

  end_mask = tf.equal(docs_in_query-1,
                      tf.range(serp_len)[None, :])
  reward_tile = tf.tile(rewards, [1, serp_len])
  q_values = tf.where(end_mask, reward_tile, q_values)

  unfiltered_dqn_loss = (scores - q_values)**2

  doc_denom = tf.cast(tf.reduce_sum(docs_in_query), tf.float32)
  mask = tf.squeeze(tf.sequence_mask(docs_in_query, serp_len), axis=1)

  filtered_mc_loss = tf.where(mask,
                           unfiltered_mc_loss,
                           tf.zeros_like(unfiltered_mc_loss))
  mc_loss = tf.reduce_sum(filtered_mc_loss)/doc_denom

  filtered_dqn_loss = tf.where(mask,
                               unfiltered_dqn_loss,
                               tf.zeros_like(unfiltered_dqn_loss))
  dqn_loss = tf.reduce_sum(filtered_dqn_loss)/doc_denom

  # dqn_loss = tf.Print(dqn_loss, [scores[0,i] for i in range(10)], 'Scores:')
  # dqn_loss = tf.Print(dqn_loss, [q_values[0,i] for i in range(10)], 'Q_values:')
  # dqn_loss = tf.Print(dqn_loss, [unfiltered_dqn_loss[0,i] for i in range(10)], 'Loss:')
  # dqn_loss = tf.Print(dqn_loss, [n_docs], 'DQN:')
  # mc_loss = tf.Print(mc_loss, [n_docs], 'MC:')

  tf.summary.scalar('monte_carlo/loss', mc_loss)
  tf.summary.scalar('DQN/loss', dqn_loss)

  tf.summary.scalar('DQN/double_max_scores', tf.reduce_mean(double_max_scores))

  return mc_loss, dqn_loss