import tensorflow as tf
import model_utils as mu
import rnn_utils as ru
import numpy as np

def get_pos_loss(params, prev_loss, pos, docs_in_query, scores, max_q, rewards):
  if prev_loss is None:
    loss, q_loss, monte_carlo_loss = ([],[],[])
  else:
    loss, q_loss, monte_carlo_loss = prev_loss

  q_pos_loss = (scores - max_q)**2
  mc_pos_loss = (scores - rewards)**2
  # Filter shorter queries.
  q_pos_loss = tf.where(tf.less(pos, docs_in_query),
                        q_pos_loss,
                        tf.zeros_like(q_pos_loss))
  mc_pos_loss = tf.where(tf.less(pos, docs_in_query),
                         mc_pos_loss,
                         tf.zeros_like(mc_pos_loss))

  q_pos_loss = tf.reduce_mean(q_pos_loss)
  mc_pos_loss = tf.reduce_mean(mc_pos_loss)

  tf.summary.scalar('scores/pos_%d' % pos,
                    tf.reduce_mean(scores))
  tf.summary.scalar('q_loss/pos_%d' % pos,
                    q_pos_loss)
  tf.summary.scalar('monte_carlo/loss_pos_%d' % pos,
                    mc_pos_loss)

  q_loss.append(q_pos_loss)
  monte_carlo_loss.append(mc_pos_loss)
  if params['update'] == 'monte_carlo':
    loss.append(mc_pos_loss)
  else:
    loss.append(q_pos_loss)
  return loss, q_loss, monte_carlo_loss

def calculate_naive_loss(params, replay, rewards):
  serp_len = params['serp_len']
  visible_dropout = params['visible_dropout']
  docs_in_query = replay['docs_per_query']
  batch_docs = replay['serp']
  n_docs = tf.shape(batch_docs)[0]

  drop_docs = tf.nn.dropout(batch_docs, visible_dropout)

  doc_emb = mu._shared_doc_embeddings(drop_docs, params,
                                      '/main/doc_emb',
                                      inference=True,
                                      reuse_variable_scope=True)

  main_serp = tf.gather(doc_emb, 1, axis=1)
  main_scores = []
  for i in range(serp_len):
    main_scores.append(mu._create_subnetwork(main_serp, params,
                                   '/main/score/pos_%d' % i,
                                   reuse_variable_scope=True,
                                   label_network=False,
                                   inference=False))
    if i < serp_len - 1:
      main_serp = tf.concat([
                    main_serp,
                    tf.gather(doc_emb, 1, axis=1),
                    ], axis=1)

  scores = tf.concat(main_scores, axis=1)
  unfiltered_mc_loss = (rewards-scores)**2

  ind = tf.expand_dims(tf.range(serp_len),0)
  mask = tf.less(ind, docs_in_query)
  filtered_mc_loss = tf.where(mask,
                           unfiltered_mc_loss,
                           tf.zeros_like(unfiltered_mc_loss))
  mc_loss = tf.reduce_mean(filtered_mc_loss)

  # tf.summary.scalar('q_loss/loss', mean_losses[1])
  tf.summary.scalar('monte_carlo/loss', mc_loss)

  tf.summary.scalar('loss', mc_loss)

  return (mc_loss, mc_loss, mc_loss)

  # # Create embeddings from documents.
  # doc_i = tf.zeros([batch_size, 1])
  # main_emb = []
  # label_emb = []
  # max_emb = []
  # for i in range(serp_len):
  #   main_emb.append(mu._shared_doc_embeddings(
  #                   docs[i],
  #                   params, '/main/doc_emb',
  #                   reuse_variable_scope=True,
  #                   inference=False))

  #   label_emb.append(mu._shared_doc_embeddings(
  #                     docs[i],
  #                     params, '/label/doc_emb',
  #                     reuse_variable_scope=i>0,
  #                     label_network=True,
  #                     inference=True))
  #   # Documents for max choices, index is one behind.
  #   if i < serp_len - 1:
  #     max_emb.append(mu._shared_doc_embeddings(
  #                       max_docs[i],
  #                       params, '/label/doc_emb',
  #                       reuse_variable_scope=True,
  #                       label_network=True,
  #                       inference=True))

  # main_serp = main_emb[0]
  # label_serp = label_emb[0] 
  # losses = None
  # for i in range(serp_len-1):
  #   scores = mu._create_subnetwork(main_serp, params,
  #                                  '/main/score/pos_%d' % i,
  #                                  reuse_variable_scope=True,
  #                                  label_network=False,
  #                                  inference=False)

  #   max_serp = tf.concat([max_emb[i], label_serp], axis=1)
  #   max_scores = mu._create_subnetwork(max_serp,
  #                                      params,
  #                                      '/label/score/pos_%d' % (i + 1),
  #                                      reuse_variable_scope=False,
  #                                      label_network=True,
  #                                      inference=False)

  #   losses = get_pos_loss(params, losses, i, docs_in_query,
  #                         scores, max_scores, rewards)

  #   main_serp = tf.concat([main_emb[i], main_serp], axis=1)
  #   if i < serp_len - 2:
  #     label_serp = tf.concat([label_emb[i], label_serp], axis=1)

  # scores = mu._create_subnetwork(main_serp, params,
  #                                '/main/score/pos_%d' % (serp_len-1),
  #                                reuse_variable_scope=True,
  #                                label_network=False,
  #                                inference=False)
  # losses = get_pos_loss(params, losses,
  #                       serp_len-1, docs_in_query,
  #                       scores, rewards, rewards)

  # mean_losses = [tf.reduce_mean(loss) for loss in losses]

  # tf.summary.scalar('q_loss/loss', mean_losses[1])
  # tf.summary.scalar('monte_carlo/loss', mean_losses[2])

  # tf.summary.scalar('loss', mean_losses[0])
  # return mean_losses

def calculate_pbow_loss(params, replay, rewards):
  serp_len = params['serp_len']
  visible_dropout = params['visible_dropout']
  batch_size = tf.shape(replay['doc_0'])[0]
  docs_in_query = replay['docs_per_query']

  # Gather documents from replay.
  docs = [replay['doc_%d' % i] for i in range(serp_len)]
  drop_docs = [tf.nn.dropout(x, visible_dropout) for x in docs]
  max_docs = [replay['max_doc_%d' % i] for i in range(1, serp_len)]

  # Create embeddings from documents.
  doc_i = tf.zeros([batch_size, 1])
  main_emb = []
  label_emb = []
  max_emb = []
  for i in range(serp_len):
    main_pos_doc = tf.concat([doc_i, drop_docs[i]], axis=1)
    main_emb.append(mu._shared_doc_embeddings(
                    main_pos_doc,
                    params, '/main/doc_emb',
                    reuse_variable_scope=True,
                    inference=False))

    label_pos_doc = tf.concat([doc_i, docs[i]], axis=1)
    label_emb.append(mu._shared_doc_embeddings(label_pos_doc,
                      params, '/label/doc_emb',
                      reuse_variable_scope=i>0,
                      label_network=True,
                      inference=True))
    doc_i += 1
    # Documents for max choices, index is one behind.
    if i < serp_len - 1:
      max_pos_doc = tf.concat([doc_i, max_docs[i]], axis=1)
      max_emb.append(mu._shared_doc_embeddings(max_pos_doc,
                        params, '/label/doc_emb',
                        reuse_variable_scope=True,
                        label_network=True,
                        inference=True))

  serp_pbow = main_emb[0]
  label_pbow = label_emb[0] 
  losses = None
  for i in range(serp_len-1):
    scores = mu._create_subnetwork(serp_pbow, params, '/main/score',
                                   reuse_variable_scope=True,
                                   label_network=False,
                                   inference=False)

    max_scores = mu._create_subnetwork(label_pbow + max_emb[i],
                                       params,
                                       '/label/score',
                                       reuse_variable_scope=i>0,
                                       label_network=True,
                                       inference=False)

    losses = get_pos_loss(params, losses, i, docs_in_query,
                          scores, max_scores, rewards)

    serp_pbow += main_emb[i+1]
    if i < serp_len - 2:
      label_pbow += label_emb[i+1]

  scores = mu._create_subnetwork(serp_pbow, params,
                                 '/main/score',
                                 reuse_variable_scope=True,
                                 label_network=False,
                                 inference=False)
  losses = get_pos_loss(params, losses,
                        serp_len-1, docs_in_query,
                        scores, rewards, rewards)

  mean_losses = [tf.reduce_mean(loss) for loss in losses]

  tf.summary.scalar('q_loss/loss', mean_losses[1])
  tf.summary.scalar('monte_carlo/loss', mean_losses[2])

  tf.summary.scalar('loss', mean_losses[0])
  return mean_losses

def max_train_filter(gru, hidden_states, serp_len, doc_col,
                     serp_ind, docs_in_query, max_n_docs):
  hidden_states = tf.transpose(hidden_states, [1, 0, 2])
  max_states = tf.expand_dims(hidden_states[:, :-1, :], axis=2)
  max_col = tf.expand_dims(doc_col, axis=1)
  train_scores = tf.squeeze(gru((max_states, None), max_col)[1], axis=3)
  score_filter = tf.one_hot(serp_ind[:, :-1], max_n_docs,
                            on_value=np.NINF, off_value=0.)

  score_filter = tf.cumsum(score_filter, axis=1)

  n_doc_filter = tf.sequence_mask(docs_in_query, max_n_docs)
  n_doc_filter = tf.where(n_doc_filter,
                          tf.zeros_like(n_doc_filter, dtype=tf.float32),
                          tf.fill(n_doc_filter.shape, np.NINF))
  score_filter += n_doc_filter

  argmax_s = tf.argmax(train_scores + score_filter, axis=2)

  return tf.one_hot(argmax_s, max_n_docs)

def get_label_scores(params, replay):
  serp_len = params['serp_len']
  batch_size = tf.shape(replay['serp'])[0]
  hidden_state_size = params['hidden_state_size']

  doc_col = mu._shared_doc_embeddings(replay['docs'], params,
                                        '/label/doc_emb',
                                        inference=True,
                                        label_network=True,
                                        reuse_variable_scope=False)

  doc_emb = mu._shared_doc_embeddings(replay['serp'], params,
                                            '/label/doc_emb',
                                            inference=True,
                                            label_network=True,
                                            reuse_variable_scope=True)

  init_scores = tf.zeros([batch_size, 1])
  init_hidden = tf.zeros([batch_size, hidden_state_size])

  serp_emb = tf.transpose(doc_emb, [1, 0, 2])
  gru = ru.get_gru_score_layer(params, '/label/gru',
                               label_network=True,
                               inference=True,
                               reuse_variable_scope=False)

  hidden_states, _ = tf.scan(gru, serp_emb, (init_hidden, init_scores))

  hidden_states = tf.transpose(hidden_states, [1, 0, 2])
  hidden_states = hidden_states[:, :-1, None, :]

  max_col = tf.expand_dims(doc_col, axis=1)
  label_scores = tf.squeeze(gru((hidden_states, None), max_col)[1], axis=3)

  return tf.stop_gradient(label_scores)


def calculate_gru_loss(params, replay, rewards):
  serp_len = params['serp_len']
  visible_dropout = params['visible_dropout']
  docs_in_query = replay['docs_per_query']
  batch_docs = replay['serp']
  max_n_docs = params['max_docs']
  n_docs = tf.shape(batch_docs)[0]
  hidden_state_size = params['hidden_state_size']

  drop_col = tf.nn.dropout(replay['docs'], visible_dropout)

  doc_col = mu._shared_doc_embeddings(drop_col, params,
                                        '/main/doc_emb',
                                        inference=False,
                                        reuse_variable_scope=True)
  # if params['context_input']:
  #   gru = ru.get_gru_layer(params, '/main/gru/collection/',
  #                                label_network=False,
  #                                inference=False,
  #                                reuse_variable_scope=False)
  #   input_col = tf.transpose(doc_col, [1, 0, 2])

  #   hidden_col = tf.scan(gru, input_col, init_hidden)

  #   hidden_col = tf.transpose(hidden_col, [1, 0, 2])

  #   indices = tf.stack([tf.range(n_docs), tf.squeeze(docs_in_query-1 ,axis=1)], axis=1)
  #   init_hidden = tf.gather_nd(hidden_col, indices)
  # else:
  init_hidden = tf.zeros([n_docs, hidden_state_size])

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

  init_scores = tf.zeros([n_docs, 1])
  hidden_states = tf.scan(gru, serp_emb[:-1, :, :], init_hidden)
  hidden_states = tf.concat([init_hidden[None, :, :], hidden_states], axis=0)
  scores = mu._create_subnetwork(score_input,
                                   params,
                                   subnetwork_name='/main/scoring',
                                   label_network=False,
                                   reuse_variable_scope=i>0,
                                   inference=True)


  scores = tf.squeeze(tf.transpose(scores, [1,0,2]), axis=2)
  unfiltered_mc_loss = (rewards-scores)**2.

  max_filter = max_train_filter(gru, hidden_states, serp_len,
                                doc_col, replay['serp_ind'],
                                docs_in_query, max_n_docs)

  label_scores = get_label_scores(params, replay)

  double_max_scores = tf.reduce_sum(max_filter*label_scores, axis=2)
  q_values = tf.concat([double_max_scores, rewards], axis=1)

  end_mask = tf.equal(docs_in_query-1,
                      tf.expand_dims(tf.range(serp_len), axis=0))
  reward_tile = tf.tile(rewards, [1, serp_len])
  q_values = tf.where(end_mask, reward_tile, q_values)

  unfiltered_dqn_loss = (scores - q_values)**2.

  doc_denom = tf.cast(tf.reduce_sum(tf.minimum(docs_in_query, serp_len)), tf.float32)
  mask = tf.squeeze(tf.sequence_mask(docs_in_query, serp_len), axis=1)


  filtered_mc_loss = tf.where(mask,
                           unfiltered_mc_loss,
                           tf.zeros_like(unfiltered_mc_loss))
  mc_loss = tf.reduce_sum(filtered_mc_loss)/doc_denom

  filtered_dqn_loss = tf.where(mask,
                               unfiltered_dqn_loss,
                               tf.zeros_like(unfiltered_dqn_loss))

  dqn_loss = tf.reduce_sum(filtered_dqn_loss)/doc_denom

  # tf.summary.scalar('q_loss/loss', mean_losses[1])

  tf.summary.scalar('monte_carlo/loss', mc_loss)
  tf.summary.scalar('DQN/loss', dqn_loss)

  filtered_double_max = tf.where(mask[:,:-1],
                                 double_max_scores,
                                 tf.zeros_like(double_max_scores))
  double_max_denom = doc_denom - tf.cast(n_docs, tf.float32)
  double_max_mean = tf.reduce_sum(filtered_double_max)/double_max_denom
  tf.summary.scalar('DQN/double_max_scores', double_max_mean)

  return mc_loss, dqn_loss