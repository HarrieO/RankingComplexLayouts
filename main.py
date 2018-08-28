import argparse
import letorinput as letorin
import losses
import models
import numpy as np
import rewards
import tensorflow as tf
import gru.gru as gru
import exppos.exppos as exppos
import seppos.seppos as seppos

from tensorflow.contrib.training import wait_for_new_checkpoint

parser = argparse.ArgumentParser(description='Baseline Run')
parser.add_argument('--model_dir', type=str, default=None,
                    help='Directory to store/load model.')
parser.add_argument('--summary_dir', type=str, default=None,
                    help='Directory to store/load summaries.')
parser.add_argument('--input_dir', type=str, required=True,
                    help='Directory where input is found '
                      '(features.txt, [train, vali, test].tfrecord).')
parser.add_argument('--steps', type=int, default=10000,
                    help='')
parser.add_argument('--eval', action='store_true',
                    help='')
parser.add_argument('--eval_steps', type=int, default=500,
                    help='')
parser.add_argument('--dataset', type=str, required=True,
                    help='')
parser.add_argument('--partition', type=str, default='train',
                    help='')
# parser.add_argument('--serp_len', type=int, default=10,
#                     help='')
parser.add_argument('--discount', type=str, default='ndcg',
                    help='')
parser.add_argument('--model', type=str, required=True, help='')
parser.add_argument('--learning_rate', type=float, default=0.01)
parser.add_argument('--epsilon_decay', type=float, default=0.9999)
parser.add_argument('--steps_per_transfer', type=int, default=1000)
parser.add_argument('--min_replay', type=int, default=1000)
parser.add_argument('--update', type=str, default='dqn')
parser.add_argument('--doc_rewards', action='store_true',
                    help='')


args = parser.parse_args()

params = {
  'hidden_units': [128],
  'hidden_state_size': 256,
  'model': args.model,
  'model_name': 'dnn',
  'update': args.update,
  # 'serp_len': args.serp_len,
  'serp_len': 10,
  'evaluation': args.eval,
  'partition': args.partition,
  'steps': args.steps,
  'eval_steps': args.eval_steps,
  'doc_emb': [128],
  'steps_per_transfer': args.steps_per_transfer,
  'visible_dropout': 1.,
  'hidden_dropout': 1.,
  'l2_scale': 0.,
  'learning_rate': args.learning_rate,
  'epsilon_decay': args.epsilon_decay,
  'discount': args.discount,
  'context_input': False,
  'all_discounts': {
    'ndcg': list(range(10)),
    'hill': [8,6,4,2,0,1,3,5,7,9],
    'reverse': [9,8,7,6,5,4,3,2,1,0],
  },
  'compact_gru': True,
  'min_replay': args.min_replay,
  'read_batch': 64,
  'replay_batch': 64,
  'doc_rewards': args.doc_rewards,
}

# 'ndcg': [1./(np.log2(i+2)) for i in range(10)],
# 'wave': [1./(np.log2(i+2)) for i in [4,3,2,1,0,5,6,7,8,9]],
# 'hill': [1./(np.log2(i+2)) for i in [8,5,3,1,0,2,4,6,7,9]],
# 'even': [1./(np.log2(i+2)) for i in [0,5,1,6,2,7,3,8,4,9]],
# 'uneven': [1./(np.log2(i+2)) for i in [5,0,6,1,7,2,8,3,9,4]],
# 'reverse': [1./(np.log2(i+2)) for i in [9,8,7,6,5,4,3,2,1,0]],
# 'skip': [1./(np.log2(i+2)) for i in [7,8,9,0,1,2,3,4,5,6]],
# 'cup': [1./(np.log2(i+2)) for i in [0,2,4,6,8,9,7,5,3,1]],

if args.dataset == 'NP2003':
  params['train_size'] = 90
  params['vali_size'] = 30
  params['test_size'] = 30
  params['max_docs'] = 1000
elif args.dataset == 'MSLR30':
  params['train_size'] = 18919
  params['vali_size'] = 6306
  params['test_size'] = 6306
  params['max_docs'] = 1251
elif args.dataset == 'WEBSCOPE':
  params['train_size'] = 19944
  params['vali_size'] = 2994
  params['test_size'] = 6983
  params['max_docs'] = 139
elif args.dataset == 'istella':
  params['train_size'] = 23219
  params['vali_size'] = 9799 # vali set is actually empty use train
  params['test_size'] = 9799
  params['max_docs'] = 439

if params['evaluation']:
  params['train_size'] = params['vali_size']

feat = letorin.get_features(params, args.input_dir)

n_read_threads = 3
n_policy_threads = 5
if args.eval:
  params['visible_dropout'] = 0.
  params['hidden_dropout'] = 0.
  params['read_batch'] = 50
  n_policy_threads = 1

if args.dataset == 'istella' and params['partition'] == 'vali':
  params['partition'] = 'train'
  examples, labels = letorin.get_letor_examples(params, args.input_dir,
                                                feat, num_threads=n_read_threads)
  params['partition'] = 'vali'
else:
  examples, labels = letorin.get_letor_examples(params, args.input_dir,
                                                feat, num_threads=n_read_threads)

global_step = tf.Variable(0, trainable=False, name='global_step')

epsilon = tf.Variable(1.)
tf.summary.scalar('epsilon', epsilon)
degrade_epsilon = tf.assign(epsilon, tf.maximum(epsilon*params['epsilon_decay'], 0.05))
turn_off_epsilon = tf.assign(epsilon, 0)

eval_update_ops = []

if params['model'] == 'pbow':
  to_enqueue = models.pbow_model(params, examples, labels, epsilon)
elif params['model'] == 'naive':
  to_enqueue = models.naive_model(params, examples, labels, epsilon)
elif params['model'] == 'gru':
  to_enqueue = gru.model(params, examples, labels, epsilon)
elif params['model'] == 'exppos':
  to_enqueue = exppos.model(params, examples, labels, epsilon, eval_update_ops)
elif params['model'] == 'seppos':
  to_enqueue = seppos.model(params, examples, labels, epsilon, eval_update_ops)
else:
  raise ValueError('Unknown model: %s' % params['model'])

# Cutoff discounts
discounts = rewards.get_DCG_discounts(params, to_enqueue['docs_per_query'],
                                      labels)

# Ideal rewards
ideal_rewards = rewards.calculate_ideal_rewards(params, labels, discounts)
for name, v in ideal_rewards.items():
  to_enqueue['ideal_rewards/%s' % name] = v
  to_enqueue['discounts/%s' % name] = discounts[name]
  to_enqueue['step_added'] = [[global_step]]*params['read_batch']

transfer_step = tf.Variable(0, name='transfer_step')
update_transfer_step = tf.assign(transfer_step, transfer_step+1,
                                 name='increment_transfer_step')
tf.summary.scalar('transfer_step', transfer_step)

if params['evaluation']:
  # rewards = rewards.calculate_NDCG_reward(params,
  #               to_enqueue, stats_ops=eval_update_ops)
  rewards, _ = rewards.calculate_custom_discount_reward(params,
                to_enqueue, stats_ops=eval_update_ops)
  mean_reward, mean_update = tf.metrics.mean(tf.reduce_mean(rewards))
  eval_update = tf.group(mean_update, *eval_update_ops)
else:
  replay = tf.train.shuffle_batch(
      to_enqueue,
      params['replay_batch'],
      params['min_replay'] + n_policy_threads*params['replay_batch'],
      params['min_replay'],
      allow_smaller_final_batch=False,
      enqueue_many=True,
      name='ExperienceReplay',
      num_threads=n_policy_threads,
  )

  tf.summary.scalar('DQN/replay/step_added', tf.reduce_mean(replay['step_added']))
  tf.summary.scalar('DQN/replay/step_delay', tf.reduce_mean(replay['step_added']) - global_step)

    # rewards = rewards.calculate_NDCG_reward(params, replay)
  rewards, doc_rewards = rewards.calculate_custom_discount_reward(params, replay)

  # loss = losses.calculate_loss(params, replay, rewards)
  if params['model'] == 'naive':
    all_losses = losses.calculate_naive_loss(params, replay, rewards)
  elif params['model'] == 'pbow':
    all_losses = losses.calculate_pbow_loss(params, replay, rewards)
  elif params['model'] == 'gru':
    mc_loss, dqn_loss = gru.loss(params, replay, rewards, doc_rewards)
  elif params['model'] == 'exppos':
    mc_loss, dqn_loss = exppos.loss(params, replay, rewards)
  elif params['model'] == 'seppos':
    mc_loss, dqn_loss = seppos.loss(params, replay, rewards, doc_rewards)
  else:
    raise ValueError('Unknown model: %s' % params['model'])

  if params['update'] == 'monte_carlo':
    loss = mc_loss
  elif params['update'] == 'dqn':

    # loss = tf.cond(global_step <= params['steps_per_transfer'],
    #                lambda: mc_loss,
    #                lambda: dqn_loss)
    loss = dqn_loss
  else:
    raise ValueError('Unknown update: %s' % params['update'])


  opt = tf.train.AdamOptimizer(learning_rate=params['learning_rate'])
  # opt = tf.train.AdagradOptimizer(learning_rate=params['learning_rate'])
  # opt = tf.train.GradientDescentOptimizer(learning_rate=params['learning_rate'])

  opt_op = opt.minimize(loss, global_step=global_step)


  train_prefix = params['model_name'] + '/main/'
  label_prefix = params['model_name'] + '/label/'
  train_variables = {}
  for var in tf.trainable_variables():
    if var.name.startswith(train_prefix):
      train_variables[var.name[len(train_prefix):]] = var

  pairs = []
  for var in tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES):
    if var.name.startswith(label_prefix):
      common_name = var.name[len(label_prefix):]
      pairs.append(tf.assign(var, train_variables[common_name]))
  transfer_label_weights = tf.group(update_transfer_step, *pairs)

merged_summary = tf.summary.merge_all()

init = tf.group(tf.global_variables_initializer(),
                tf.local_variables_initializer())

sum_path = args.summary_dir + '/' + params['partition']
if params['partition'] == 'train' and params['evaluation']:
  sum_path = args.summary_dir + '/overfit'
writer = tf.summary.FileWriter(sum_path)
saver = tf.train.Saver()

if not params['evaluation']:

  # Train supervisor to handle starting sessions.
  sv = tf.train.Supervisor(logdir=args.model_dir,
                           summary_writer=writer,
                           save_summaries_secs=120,
                           global_step=global_step,
                           # manual saving since learning is fast
                           save_model_secs=0)

  # At this point the model will be instantiated and actually ran.
  with sv.managed_session() as sess:

    # Continue from a previous saved checkpoint, if it exists.
    checkpoint = tf.train.latest_checkpoint(args.model_dir)
    if checkpoint:
      print('Loading checkpoint', checkpoint)
      saver.restore(sess, checkpoint)
    else:
      print('No existing checkpoint found.')
      sess.run(init)

    # Check the current state of the network.
    i = sess.run([global_step])[0]
    if i == 0:
      saver.save(sess, args.model_dir + '/model.ckpt',
                 global_step=i)
    print('Running %d steps.' % (params['steps'] - i))
    while i < params['steps']:
      if i > 1 and i % params['steps_per_transfer'] == 1:
        sess.run([transfer_label_weights])
      i, l_i = sess.run([global_step, loss, opt_op, degrade_epsilon])[:2]
      # print("%d %f" % (i, l_i))
      # Evaluation will be performed on saved checkpoints
      # only. Since learning goes very fast, we save often.
      if i % params['eval_steps'] == 0 or i == params['steps']:
        saver.save(sess, args.model_dir + '/model.ckpt', global_step=i)
else:
  print('Evaluating on %s' % params['partition'])
  # For each checkpoint the entire dataset is evaluated.
  steps_per_eval = params['%s_size' % params['partition']]
  checkpoint = None
  # Basic session since we will only manually save summaries.
  with tf.Session() as sess:
    coord = tf.train.Coordinator()
    # Queue runners will take care of reading data in seperate threads.
    threads = tf.train.start_queue_runners(coord=coord)
    while True:
      checkpoint = wait_for_new_checkpoint(args.model_dir,
                                           checkpoint,
                                           seconds_to_sleep=1,
                                           timeout=1200)
      if checkpoint is None:
        print('No checkpoint found for 20 min, exiting evaluation.')
        break
      # Init for variables that are not part of checkpoint,
      # in this case the ones used for metrics.
      sess.run(init)
      # Restore a checkpoint saved by the training run.
      saver.restore(sess, checkpoint)
      sess.run(turn_off_epsilon)
      # Update the metrics for every element in the dataset.
      batch_steps = int(np.ceil(steps_per_eval/float(params['read_batch'])))
      for i in range(batch_steps):
        sess.run([eval_update])
      # Get the resulting metrics.
      cur_step, cur_reward, cur_summary = sess.run([global_step, mean_reward, merged_summary])
      # Pass the summary to the writer, which stores it for Tensorboard.
      writer.add_summary(cur_summary, global_step=cur_step)
      writer.flush()

      print('Step %d: %.02f' % (cur_step, cur_reward))
      if cur_step == params['steps']:
        break

    coord.request_stop()
    coord.join(threads)


# init = tf.group(tf.global_variables_initializer(),
#                 tf.local_variables_initializer())
# with tf.Session() as sess:
#   coord = tf.train.Coordinator()
#   # Queue runners will take care of reading data in seperate threads.
#   threads = tf.train.start_queue_runners(coord=coord)
  
#   sess.run(init)
    

#   coord.request_stop()
#   coord.join(threads)
#   exit()
