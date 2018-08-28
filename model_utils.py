import tensorflow as tf
import six
import numpy as np
from tensorflow.contrib import layers
from tensorflow.python.ops import partitioned_variables
from tensorflow.python.ops import variable_scope
from tensorflow.python.summary import summary
from tensorflow.contrib.layers import l2_regularizer

def _get_feature_dict(features):
  if isinstance(features, dict):
    return features
  return {"": features}

def _add_layer_summary(value, tag):
  summary.scalar("%s/fraction_of_zero_values" % tag, tf.nn.zero_fraction(value))
  summary.histogram("%s/activation" % tag, value)

def _get_doc_tensors(features, params, subnetwork_name='',
                     reuse_variable_scope=False):
  assert False, 'Only batch reading supported.'
  # feature_columns = params.get("feature_columns")
  # config = params.get("config")
  # model_name = params.get("model_name")
  # l2_scale = params['l2_scale']
  # num_ps_replicas = config.num_ps_replicas if config else 0
  # input_layer_partitioner = params.get("input_layer_partitioner") or (
  #     partitioned_variables.min_max_variable_partitioner(
  #         max_partitions=num_ps_replicas,
  #         min_slice_size=64 << 20))

  # if not feature_columns:
  #   raise ValueError(
  #       "feature_columns must be defined.")

  # features = _get_feature_dict(features)

  # parent_scope = model_name
  # if subnetwork_name:
  #   parent_scope = parent_scope + '/' + subnetwork_name

  # partitioner = (
  #     partitioned_variables.min_max_variable_partitioner(
  #     max_partitions=num_ps_replicas))
  # with variable_scope.variable_scope(
  #     parent_scope,
  #     values=tuple(six.itervalues(features)),
  #     partitioner=partitioner,
  #     regularizer=l2_regularizer(l2_scale),
  #     reuse=reuse_variable_scope):
  #   with variable_scope.variable_scope(
  #       "input_from_feature_columns",
  #       values=tuple(six.itervalues(features)),
  #       partitioner=input_layer_partitioner) as input_scope:
  #     net = layers.input_from_feature_columns(
  #           columns_to_tensors=features,
  #           feature_columns=feature_columns,
  #           weight_collections=[parent_scope],
  #           scope=input_scope)

  # min_feat = tf.reduce_min(net, axis=0, keep_dims=True)
  # max_feat = tf.reduce_max(net, axis=0, keep_dims=True)
  # denom_feat = max_feat-min_feat
  # net = (net-min_feat)/tf.where(denom_feat>0., denom_feat, tf.ones_like(denom_feat))
  # # for i in range(net.shape[1]):
  # #   tf.summary.histogram("normalized/%d" % i, net[:,i])
  # return net

def _shared_doc_embeddings(doc_tensors,
                           params,
                           subnetwork_name='',
                           label_network=False,
                           reuse_variable_scope=False,
                           inference=False):
  hidden_units = params.get("doc_emb")
  activation_fn = params.get("activation_fn") or tf.nn.relu
  config = params.get("config")
  model_name = params.get("model_name")
  l2_scale = params['l2_scale']
  hidden_dropout = params['hidden_dropout']
  num_ps_replicas = config.num_ps_replicas if config else 0
  input_layer_partitioner = params.get("input_layer_partitioner") or (
      partitioned_variables.min_max_variable_partitioner(
          max_partitions=num_ps_replicas,
          min_slice_size=64 << 20))

  parent_scope = model_name + subnetwork_name

  partitioner = (
      partitioned_variables.min_max_variable_partitioner(
      max_partitions=num_ps_replicas))
  with variable_scope.variable_scope(
      parent_scope,
      values=(doc_tensors,),
      regularizer=l2_regularizer(l2_scale),
      partitioner=partitioner,
      reuse=reuse_variable_scope):
    
    net = doc_tensors

    for layer_id, num_hidden_units in enumerate(hidden_units):
      with variable_scope.variable_scope(
          "shared_document_layer_%d" % layer_id,
          values=(net,)) as hidden_layer_scope:
        net = layers.fully_connected(
            net,
            num_hidden_units,
            activation_fn=activation_fn,
            variables_collections=[parent_scope],
            scope=hidden_layer_scope,
            trainable=not label_network)
      # _add_layer_summary(net, hidden_layer_scope.name)
  return net

def _create_subnetwork(doc_tensors,
                       params,
                       subnetwork_name='',
                       label_network=False,
                       reuse_variable_scope=False,
                       inference=False,
                       n_output=1):
  hidden_units = params.get("hidden_units")
  activation_fn = params.get("activation_fn") or tf.nn.relu
  config = params.get("config")
  model_name = params.get("model_name")
  hidden_dropout = params['hidden_dropout']
  l2_scale = params['l2_scale']
  if params['model'] == 'exppos':
    n_output = params['serp_len']
  # else:
  #   n_output = n_output
  num_ps_replicas = config.num_ps_replicas if config else 0
  input_layer_partitioner = params.get("input_layer_partitioner") or (
      partitioned_variables.min_max_variable_partitioner(
          max_partitions=num_ps_replicas,
          min_slice_size=64 << 20))

  parent_scope = model_name + subnetwork_name

  if hidden_units is None:
    raise ValueError(
        "hidden_units must be defined.")

  partitioner = (
      partitioned_variables.min_max_variable_partitioner(
      max_partitions=num_ps_replicas))
  with variable_scope.variable_scope(
      parent_scope,
      values=(doc_tensors,),
      partitioner=partitioner,
      regularizer=l2_regularizer(l2_scale),
      reuse=reuse_variable_scope):
    
    net = doc_tensors
    if not (inference or label_network) and hidden_dropout < 1:
      net = tf.nn.dropout(net, hidden_dropout)

    for layer_id, num_hidden_units in enumerate(hidden_units):
      with variable_scope.variable_scope(
          "hiddenlayer_%d" % layer_id,
          values=(net,)) as hidden_layer_scope:
        net = layers.fully_connected(
            net,
            num_hidden_units,
            activation_fn=activation_fn,
            variables_collections=[parent_scope],
            scope=hidden_layer_scope,
            trainable=not label_network)
        if not (inference or label_network) and hidden_dropout < 1:
          net = tf.nn.dropout(net, hidden_dropout)

    with variable_scope.variable_scope(
        "logits",
        values=(net,)) as logits_scope:

      if label_network:
        w_init = tf.zeros_initializer()
      else:
        w_init = layers.xavier_initializer()

      logits = layers.fully_connected(
          net,
          n_output,
          activation_fn=None,
          variables_collections=[parent_scope],
          weights_initializer=w_init,
          trainable=not label_network,
          scope=logits_scope)
  return logits

def select_eps_greedy_action(scores, epsilon, score_filter):
  max_ind = tf.argmax(scores + score_filter, axis=0)

  noise = tf.random_uniform(tf.shape(scores)) + score_filter
  max_ind_noise = tf.argmax(noise, axis=0)

  random_cond = tf.greater(tf.random_uniform([]), epsilon)
  action = tf.cond(random_cond,
                   lambda: max_ind,
                   lambda: max_ind_noise)
  action.set_shape([1])
  max_ind.set_shape([1])
  return action, max_ind

class EpsilonGreedy:
  def __init__(self, epsilon, batch_size, max_n_docs, docs_per_query):
    self.batch_size = batch_size
    self.max_n_docs = max_n_docs
    self.epsilon = epsilon
    n_doc_filter = tf.sequence_mask(docs_per_query[:, 0], max_n_docs)
    self.score_filter = tf.where(n_doc_filter,
          tf.zeros([batch_size, max_n_docs]),
          tf.fill([batch_size, max_n_docs], np.NINF))

  def max_ind(self, scores):
    return tf.argmax(scores[:, :, 0] + self.score_filter, axis=1)

  def choose(self, scores):
    max_ind = self.max_ind(scores)
    noise_ind = self.max_ind(tf.random_uniform(tf.shape(scores)))

    random_cond = tf.greater(tf.random_uniform([self.batch_size]),
                             self.epsilon)
    action = tf.where(random_cond, max_ind, noise_ind)
    action.set_shape(max_ind.shape)

    self.score_filter += tf.one_hot(action, self.max_n_docs,
                                    on_value=np.NINF, off_value=0.)
    return action
