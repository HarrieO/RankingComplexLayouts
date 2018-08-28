import tensorflow as tf
import numpy as np

def log2(x):
  numerator = tf.log(tf.cast(x, tf.float32))
  denominator = tf.log(tf.constant(2, dtype=numerator.dtype))
  return numerator / denominator

def get_DCG_discounts(params, docs_per_query, labels):
  new_discounts = {}
  perfect_pos = {}
  serp_len = params['serp_len']
  all_discounts = params['all_discounts']
  batch_size = tf.shape(labels)[0]

  len_discounts = {}
  for name, d_ind in all_discounts.items():
    len_discounts[name] = [np.zeros(serp_len)]
    for i in range(serp_len):
      cutoff_ind = np.array([x for x in d_ind if x <= i])
      cutoff_dis = 1./(np.log2(cutoff_ind+2))
      padded_dis = np.concatenate([cutoff_dis, np.zeros(serp_len-i-1)])
      len_discounts[name].append(padded_dis)
    len_discounts[name] = tf.convert_to_tensor(np.array(len_discounts[name]), dtype=tf.float32)

  cut_len = tf.minimum(serp_len, tf.cast(docs_per_query, tf.int32))
  for name in all_discounts:
    new_discounts[name] = tf.gather(len_discounts[name],
                                    cut_len[:, 0])
  return new_discounts

def calculate_ideal_rewards(params, labels, all_discounts):
  ideal_rewards = {}
  serp_len = params['serp_len']
  batch_size = labels.shape[0]
  padding = tf.zeros([labels.shape[0],
                      tf.maximum(serp_len-tf.shape(labels)[1], 0)],
                      dtype=tf.int32)
  padded_labels = tf.concat([labels, padding], axis=1)

  top_labels, _ = tf.nn.top_k(padded_labels, k=serp_len)
  nominators = tf.cast(2**top_labels - 1, tf.float32)
  ideal_reward = tf.reduce_sum(nominators*all_discounts['ndcg'],
                               axis=1, keep_dims=True)

  for name, discounts in all_discounts.items():
    ideal_rewards[name] = ideal_reward
  return ideal_rewards

def calculate_DCG_reward(params, replay, stats_ops=[], ideal_reward=None):
  def mean_helper(values):
    if params['evaluation']:
      mean, update = tf.metrics.mean(tf.reduce_mean(values))
      stats_ops.append(update)
      return mean
    else:
      return tf.reduce_mean(values)
  rewards = 0.
  for i in range(params['serp_len']):
    labels = replay['label_%d' % i]
    nominators = tf.cast(2**labels - 1, tf.float32)
    denominators = tf.log(
      tf.cast(i + 2, tf.float32))/tf.log(tf.constant(2., dtype=tf.float32))
    rewards += nominators/denominators
    tf.summary.scalar('label/pos_%d' % i, mean_helper(tf.cast(labels, tf.float32)))
    tf.summary.scalar('exp_label/pos_%d' % i, mean_helper(tf.cast(nominators, tf.float32)))
  tf.summary.scalar('reward', mean_helper(rewards))
  if ideal_reward is not None:
    denom = tf.where(ideal_reward==0.,
                     tf.ones_like(ideal_reward),
                     ideal_reward)
    tf.summary.scalar('normalized_reward', mean_helper(rewards/denom))
  return rewards

def calculate_custom_discount_reward(params, replay, stats_ops=[]):
  def mean_helper(values):
    if params['evaluation']:
      mean, update = tf.metrics.mean(tf.reduce_mean(values))
      stats_ops.append(update)
      return mean
    else:
      return tf.reduce_mean(values)

  cur_reward = params['discount']
  rewards = {}
  doc_rewards = {}

  labels = tf.cast(replay['labels'], tf.float32)

  if params['model'] == 'seppos':
    serp_len = params['serp_len']
    batch_size = labels.shape[0]

    docs_per_query = replay['docs_per_query']
    pos_order = tf.cast(replay['pos_order'], tf.int32)

    first_sort = tf.nn.top_k(-pos_order, k=serp_len)[1]
    second_sort = tf.nn.top_k(-first_sort, k=serp_len)[1]
    # pos_order = tf.minimum(pos_order, serp_len-1)
    pos_labels = tf.cast(replay['select_order_labels'], tf.float32)
    pos_nominators = 2**pos_labels - 1

    to_few_docs = tf.less(docs_per_query, serp_len)[:,0]

    safe_order = tf.where(to_few_docs, second_sort, pos_order)

    batch_ind = tf.tile(tf.range(batch_size)[:, None], [1, serp_len])
    pos_ind_nd = tf.stack([tf.reshape(batch_ind, [-1]),
                           tf.reshape(safe_order, [-1])], axis=1)

  nominators = 2**labels - 1
  for name in params['all_discounts']:
    discounts = replay['discounts/%s' % name]
    doc_rewards[name] = discounts*nominators
    rewards[name] = tf.reduce_sum(discounts*nominators, axis=1, keep_dims=True)

    if params['model'] == 'seppos':

      pos_discounts = tf.gather_nd(discounts, pos_ind_nd)
      pos_discounts = tf.reshape(pos_discounts, [batch_size, serp_len])

      doc_rewards[name] = pos_discounts*pos_nominators

    ideal_reward = replay['ideal_rewards/%s' % name]

    denom = tf.where(tf.equal(ideal_reward, 0),
                     tf.ones_like(ideal_reward),
                     ideal_reward)
    norm_reward = mean_helper(rewards[name]/denom)
    tf.summary.scalar('normalized_rewards/%s' % name, norm_reward)
    if name == cur_reward:
      tf.summary.scalar('normalized_reward', norm_reward)

  for i in range(params['serp_len']):
    label_i = tf.gather(labels, i, axis=1)
    nom_i = tf.gather(nominators, i, axis=1)
    tf.summary.scalar('label/pos_%d' % i, mean_helper(label_i))
    tf.summary.scalar('exp_label/pos_%d' % i,
                      mean_helper(nom_i))
    if 'pos_order' in replay:
      tf.summary.scalar('pos_order/pos_%d' % i, mean_helper(replay['pos_order'][:, i]))
      label_order = tf.cast(replay['select_order_labels'], tf.float32)
      nom_order = 2**label_order - 1
      tf.summary.scalar('label_order2/label_%d' % i, mean_helper(label_order[:, i]))
      tf.summary.scalar('label_order/exp_label_%d' % i, mean_helper(nom_order[:, i]))

  tf.summary.scalar('reward', mean_helper(rewards[cur_reward]))

  # per_name = 'perfect_pos/%s' % cur_reward
  # pos_order = tf.cast(replay['pos_order'], tf.float32)
  # match = tf.cast(tf.equal(pos_order, replay[per_name]), tf.float32)
  # rewards = tf.reduce_sum(match, axis=1, keep_dims=True)
  # tf.summary.scalar('pos_reward', mean_helper(rewards))

  # # serp_len = params['serp_len']
  # # rewards = tf.Print(rewards, [replay[per_name][0, i] for i in range(serp_len)], 'perfect:   ')
  # # rewards = tf.Print(rewards, [pos_order[0, i] for i in range(serp_len)], 'pos_order: ')
  # # rewards = tf.Print(rewards, [rewards[0, 0]], 'rewards: ')
  # return rewards


  # rewards[cur_reward] = tf.Print(rewards[cur_reward], [labels[0, j] for j in range(10)], 'labels %s:' % cur_reward)
  # rewards[cur_reward] = tf.Print(rewards[cur_reward], [rewards[cur_reward][0]], 'reward %s:' % cur_reward)

  return rewards[cur_reward], doc_rewards[cur_reward]