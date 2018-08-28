import tensorflow as tf
import six
import model_utils as mu
from tensorflow.python.ops import variable_scope
from tensorflow.contrib import layers
from tensorflow.contrib.layers import l2_regularizer

def update_reset_node(input_vector, hidden_state, scope, keep_prob,
                      activation_fn, hidden_state_size, label_network):
  # update_gate
  if keep_prob < 1:    
    hidden_state = tf.nn.dropout(hidden_state, keep_prob)
    input_vector = tf.nn.dropout(input_vector, keep_prob)
  with variable_scope.variable_scope(
          "input_update",
          values=(input_vector,)) as input_scope:
    input_update = layers.fully_connected(
        input_vector, hidden_state_size,
        activation_fn=None,
        scope=input_scope,
        # Only single bias necessary because of sum.
        biases_initializer=None,
        trainable=not label_network)
  with variable_scope.variable_scope(
          "hidden_update",
          values=(input_vector,)) as update_scope:
    hidden_update = layers.fully_connected(
        hidden_state, hidden_state_size,
        activation_fn=None,
        scope=update_scope,
        trainable=not label_network)
  return activation_fn(input_update + hidden_update)


def get_gru_layer(params, sub_name, label_network, inference, reuse_variable_scope):
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

  def gru_layer(hidden_state, input_vector):
    l2_scale = params['l2_scale']
    with variable_scope.variable_scope(parent_scope,
        values=(hidden_state, input_vector),
        regularizer=l2_regularizer(l2_scale),
        reuse=c.reuse_var_scope) as gru_scope:

      # In case function is called more than once.
      c.reuse_var_scope = True

      with variable_scope.variable_scope("update_vector",
          values=(input_vector,)) as update_scope:
        update_vector = update_reset_node(input_vector, hidden_state,
                                          gru_scope, keep_prob, tf.sigmoid,
                                          hidden_state_size, label_network)
      with variable_scope.variable_scope("reset_vector",
          values=(input_vector,)) as update_scope:
        reset_vector = update_reset_node(input_vector, hidden_state,
                                         gru_scope, keep_prob, tf.sigmoid,
                                         hidden_state_size, label_network)
      with variable_scope.variable_scope("candidate",
          values=(input_vector,)) as update_scope:
        candidate = update_reset_node(input_vector, hidden_state*reset_vector,
                                      gru_scope, keep_prob, tf.tanh,
                                      hidden_state_size, label_network)
      return (update_vector * hidden_state +
              (tf.ones_like(update_vector)-update_vector)*candidate)

  return gru_layer

def get_gru_score_layer(params, sub_name, label_network, inference, reuse_variable_scope):
  class c:
    reuse_var_scope = reuse_variable_scope
  gru = get_gru_layer(params, sub_name, label_network,
                      inference, reuse_variable_scope)
  def gru_score_layer(hidden_state_score, input_vector):
    hidden_state, _ = hidden_state_score
    if not params['compact_gru']:
      net = tf.concat([hidden_state, input_vector], axis=-1)
      score = mu._create_subnetwork(net, params,
                         subnetwork_name=sub_name + '/scoring',
                         label_network=label_network,
                         reuse_variable_scope=c.reuse_var_scope,
                         inference=inference)
    next_hidden_state = gru(hidden_state, input_vector)
    if params['compact_gru']:
      score = mu._create_subnetwork(next_hidden_state, params,
                         subnetwork_name=sub_name + '/scoring',
                         label_network=label_network,
                         reuse_variable_scope=c.reuse_var_scope,
                         inference=inference)
    c.reuse_var_scope = True
    return next_hidden_state, score
  return gru_score_layer
