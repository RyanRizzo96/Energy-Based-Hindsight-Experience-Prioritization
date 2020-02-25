from collections import OrderedDict

import numpy as np
import tensorflow as tf
from tensorflow.contrib.staging import StagingArea

from baselines import logger
from baselines.her.util import (
    import_function, store_args, flatten_grads, transitions_in_episode_batch, convert_episode_to_batch_major)
from baselines.her.normalizer import Normalizer
from baselines.her.replay_buffer import ReplayBuffer, ReplayBufferEnergy, PrioritizedReplayBuffer
from baselines.common.mpi_adam import MpiAdam
from baselines.common import tf_util
from baselines.common.schedules import LinearSchedule


def dims_to_shapes(input_dims):
    return {key: tuple([val]) if val > 0 else tuple() for key, val in input_dims.items()}


global DEMO_BUFFER 


class DDPG(object):
    @store_args
    def __init__(self, input_dims, buffer_size, hidden, layers, network_class, polyak, batch_size,
                 Q_lr, pi_lr, norm_eps, norm_clip, action_scale, action_l2, clip_obs, scope, T,
                 rollout_batch_size, subtract_goals, relative_goals, clip_pos_returns, clip_return,
                 bc_loss, q_filter, num_demo, demo_batch_size, prm_loss_weight, aux_loss_weight,
                 sample_transitions, gamma, temperature, prioritization, alpha, beta0, beta_iters,
                 total_timesteps, rank_method, reuse=False, **kwargs):
        """
            Implementation of DDPG that is used in combination with Hindsight Experience Replay (HER).
            Added functionality to use demonstrations for training to Overcome exploration problem.
        Args:
            :param input_dims (dict of ints): dimensions for the observation (o), the goal (g), and the
                actions (u)
            :param buffer_size (int): number of transitions that are stored in the replay buffer
            :param hidden (int): number of units in the hidden layers
            :param layers (int): number of hidden layers
            :param network_class (str): the network class that should be used (e.g. 'baselines.her.ActorCritic')
            :param polyak (float): coefficient for Polyak-averaging of the target network
            :param batch_size (int): batch size for training
            :param Q_lr (float): learning rate for the Q (critic) network
            :param pi_lr (float): learning rate for the pi (actor) network
            :param norm_eps (float): a small value used in the normalizer to avoid numerical instabilities
            :param norm_clip (float): normalized inputs are clipped to be in [-norm_clip, norm_clip]
            :param action_scale(float): maximum action magnitude, i.e. actions are in [-max_u, max_u]
            :param action_l2 (float): coefficient for L2 penalty on the actions
            :param clip_obs (float): clip observations before normalization to be in [-clip_obs, clip_obs]
            :param scope (str): the scope used for the TensorFlow graph
            :param T (int): the time horizon for rollouts
            :param rollout_batch_size (int): number of parallel rollouts per DDPG agent
            :param subtract_goals (function): function that subtracts goals from each other
            :param relative_goals (boolean): whether or not relative goals should be fed into the network
            :param clip_pos_returns (boolean): whether or not positive returns should be clipped
            :param clip_return (float): clip returns to be in [-clip_return, clip_return]
            :param sample_transitions (function) function that samples from the replay buffer
            :param gamma (float): gamma used for Q learning updates
            :param reuse (boolean): whether or not the networks should be reused
            :param bc_loss: whether or not the behavior cloning loss should be used as an auxilliary loss
            :param q_filter: whether or not a filter on the q value update should be used when training with demonstartions
            :param num_demo: Number of episodes in to be used in the demonstration buffer
            :param demo_batch_size: number of samples to be used from the demonstrations buffer, per mpi thread
            :param prm_loss_weight: Weight corresponding to the primary loss
            :param aux_loss_weight: Weight corresponding to the auxilliary loss also called the cloning loss
        """
        if self.clip_return is None:
            self.clip_return = np.inf

        self.create_actor_critic = import_function(self.network_class)  # points to actor_critic.py

        self.input_dims = input_dims

        input_shapes = dims_to_shapes(input_dims)
        self.dimo = input_dims['o']
        self.dimg = input_dims['g']
        self.dimu = input_dims['u']

        self.critic_loss_episode = []
        self.actor_loss_episode = []
        self.critic_loss_avg = []
        self.actor_loss_avg = []
        
        # Energy based parameters
        self.prioritization = prioritization
        self.temperature = temperature
        self.rank_method = rank_method

        # Prepare staging area for feeding data to the model.
        stage_shapes = OrderedDict()
        for key in sorted(self.input_dims.keys()):
            if key.startswith('info_'):
                continue
            stage_shapes[key] = (None, *input_shapes[key])
        for key in ['o', 'g']:
            stage_shapes[key + '_2'] = stage_shapes[key]
        stage_shapes['r'] = (None,)
        self.stage_shapes = stage_shapes

        # Create network.
        with tf.variable_scope(self.scope):
            self.staging_tf = StagingArea(
                dtypes=[tf.float32 for _ in self.stage_shapes.keys()],
                shapes=list(self.stage_shapes.values()))
            self.buffer_ph_tf = [
                tf.placeholder(tf.float32, shape=shape) for shape in self.stage_shapes.values()]
            self.stage_op = self.staging_tf.put(self.buffer_ph_tf)

            self._create_network(reuse=reuse)  # Creates DDPG agent

        # Configure the replay buffer.
        buffer_shapes = {key: (self.T-1 if key != 'o' else self.T, *input_shapes[key])
                         for key, val in input_shapes.items()}
        buffer_shapes['g'] = (buffer_shapes['g'][0], self.dimg)
        buffer_shapes['ag'] = (self.T, self.dimg)

        buffer_size = (self.buffer_size // self.rollout_batch_size) * self.rollout_batch_size
        
        if self.prioritization == 'energy':
            self.buffer = ReplayBufferEnergy(buffer_shapes, buffer_size, self.T, self.sample_transitions, 
                                            self.prioritization)
        elif self.prioritization == 'tderror':
            self.buffer = PrioritizedReplayBuffer(buffer_shapes, buffer_size, self.T, self.sample_transitions, alpha)
            if beta_iters is None:
                beta_iters = total_timesteps
            self.beta_schedule = LinearSchedule(beta_iters, initial_p=beta0, final_p=1.0)
        else:
            self.buffer = ReplayBuffer(buffer_shapes, buffer_size, self.T, self.sample_transitions)

    def _random_action(self, n):
        return np.random.uniform(low=-self.action_scale, high=self.action_scale, size=(n, self.dimu))

    def _preprocess_og(self, o, ag, g):
        if self.relative_goals:  # no self.relative_goals
            print("self.relative_goals: ", self.relative_goals)
            g_shape = g.shape
            g = g.reshape(-1, self.dimg)
            ag = ag.reshape(-1, self.dimg)
            g = self.subtract_goals(g, ag)
            g = g.reshape(*g_shape)

        # Clip (limit) the values in an array.
        o = np.clip(o, -self.clip_obs, self.clip_obs)
        g = np.clip(g, -self.clip_obs, self.clip_obs)

        return o, g

    # Not used
    def step(self, obs):
        actions = self.get_actions(obs['observation'], obs['achieved_goal'], obs['desired_goal'])
        return actions, None, None, None

    def get_actions(self, o, ag, g, noise_eps=0., random_eps=0., use_target_net=False,
                    compute_Q=False):

        o, g = self._preprocess_og(o, ag, g)

        # Use target network use main network
        policy = self.target if use_target_net else self.main

        # values to compute
        policy_weights = [policy.actor_tf]

        if compute_Q:
            policy_weights += [policy.critic_with_actor_tf]

        # feeds
        agent_feed = {
            policy.obs: o.reshape(-1, self.dimo),
            policy.goals: g.reshape(-1, self.dimg),
            policy.actions: np.zeros((o.size // self.dimo, self.dimu), dtype=np.float32)
        }

        # Evaluating policy weights with agent information
        ret = self.sess.run(policy_weights, feed_dict=agent_feed)

        # print(ret)

        # action postprocessing
        action = ret[0]
        noise = noise_eps * self.action_scale * np.random.randn(*action.shape)  # gaussian noise
        action += noise
        action = np.clip(action, -self.action_scale, self.action_scale)
        action += np.random.binomial(1, random_eps, action.shape[0]).reshape(-1, 1) * (self._random_action(action.shape[0]) - action)  # eps-greedy
        if action.shape[0] == 1:
            action = action[0]
        action = action.copy()
        ret[0] = action

        if len(ret) == 1:
            return ret[0]
        else:
            return ret

    # Not used
    def init_demo_buffer(self, demoDataFile, update_stats=True):  # function that initializes the demo buffer

        demoData = np.load(demoDataFile)  # load the demonstration data from data file
        info_keys = [key.replace('info_', '') for key in self.input_dims.keys() if key.startswith('info_')]
        info_values = [np.empty((self.T - 1, 1, self.input_dims['info_' + key]), np.float32) for key in info_keys]

        demo_data_obs = demoData['obs']
        demo_data_acs = demoData['acs']
        demo_data_info = demoData['info']

        for epsd in range(self.num_demo): # we initialize the whole demo buffer at the start of the training
            obs, acts, goals, achieved_goals = [], [] ,[] ,[]
            i = 0
            for transition in range(self.T - 1):
                obs.append([demo_data_obs[epsd][transition].get('observation')])
                acts.append([demo_data_acs[epsd][transition]])
                goals.append([demo_data_obs[epsd][transition].get('desired_goal')])
                achieved_goals.append([demo_data_obs[epsd][transition].get('achieved_goal')])
                for idx, key in enumerate(info_keys):
                    info_values[idx][transition, i] = demo_data_info[epsd][transition][key]

            obs.append([demo_data_obs[epsd][self.T - 1].get('observation')])
            achieved_goals.append([demo_data_obs[epsd][self.T - 1].get('achieved_goal')])

            episode = dict(observations=obs,
                           u=acts,
                           g=goals,
                           ag=achieved_goals)
            for key, value in zip(info_keys, info_values):
                episode['info_{}'.format(key)] = value

            episode = convert_episode_to_batch_major(episode)
            global DEMO_BUFFER
            DEMO_BUFFER.ddpg_store_episode(episode) # create the observation dict and append them into the demonstration buffer
            logger.debug("Demo buffer size currently ", DEMO_BUFFER.get_current_size()) #print out the demonstration buffer size

            if update_stats:
                # add transitions to normalizer to normalize the demo data as well
                episode['o_2'] = episode['o'][:, 1:, :]
                episode['ag_2'] = episode['ag'][:, 1:, :]
                num_normalizing_transitions = transitions_in_episode_batch(episode)
                transitions = self.sample_transitions(episode, num_normalizing_transitions)

                o, g, ag = transitions['o'], transitions['g'], transitions['ag']
                transitions['o'], transitions['g'] = self._preprocess_og(o, ag, g)
                # No need to preprocess the o_2 and g_2 since this is only used for stats

                self.o_stats.update(transitions['o'])
                self.g_stats.update(transitions['g'])

                self.o_stats.recompute_stats()
                self.g_stats.recompute_stats()
            episode.clear()

        logger.info("Demo buffer size: ", DEMO_BUFFER.get_current_size()) # print out the demonstration buffer size

    def ddpg_store_episode(self, episode_batch, dump_buffer, w_potential, w_linear, w_rotational, rank_method, clip_energy, update_stats=True):
        """
        episode_batch: array of batch_size x (T or T+1) x dim_key
                       'o' is of size T+1, others are of size T
        """

        if self.prioritization == 'tderror':
            self.buffer.store_episode(episode_batch, dump_buffer)
        elif self.prioritization == 'energy':
            self.buffer.store_episode(episode_batch, w_potential, w_linear, w_rotational, rank_method, clip_energy)
        else:
            self.buffer.store_episode(episode_batch)

        if update_stats:
            # add transitions to normalizer
            episode_batch['o_2'] = episode_batch['o'][:, 1:, :]
            episode_batch['ag_2'] = episode_batch['ag'][:, 1:, :]
            num_normalizing_transitions = transitions_in_episode_batch(episode_batch)
            # print("START ddpg sample transition")
            # n_cycles calls HER sampler
            if self.prioritization == 'energy':
                if not self.buffer.current_size == 0 and not len(episode_batch['ag']) == 0:
                    transitions = self.sample_transitions(episode_batch, num_normalizing_transitions, 'none', 1.0, True)
            elif self.prioritization == 'tderror':
                transitions, weights, episode_idxs = \
                    self.sample_transitions(self.buffer, episode_batch, num_normalizing_transitions, beta=0)
            else:
                transitions = self.sample_transitions(episode_batch, num_normalizing_transitions)
            # print("END ddpg sample transition")

            o, g, ag = transitions['o'], transitions['g'], transitions['ag']
            transitions['o'], transitions['g'] = self._preprocess_og(o, ag, g)
            # No need to preprocess the o_2 and g_2 since this is only used for stats

            self.o_stats.update(transitions['o'])
            self.g_stats.update(transitions['g'])

            self.o_stats.recompute_stats()
            self.g_stats.recompute_stats()

    def get_current_buffer_size(self):
        return self.buffer.get_current_size()

    def _sync_optimizers(self):
        self.critic_optimiser.sync()
        self.actor_optimiser.sync()

    def _grads(self):
        # Avoid feed_dict here for performance!
        critic_loss, actor_loss, critic_grad, actor_grad, td_error = self.sess.run([
            self.critic_loss_tf,  # MSE of target_tf - main.critic_tf
            self.main.critic_with_actor_tf,  # actor_loss
            self.critic_grads,
            self.actor_grads,
            self.td_error_tf
        ])
        return critic_loss, actor_loss, critic_grad, actor_grad, td_error

    def _update(self, critic_grads, actor_grads):
        self.critic_optimiser.update(critic_grads, self.Q_lr)
        self.actor_optimiser.update(actor_grads, self.pi_lr)

    def sample_batch(self, t):
        if self.prioritization == 'energy':
            transitions = self.buffer.sample(self.batch_size, self.rank_method, temperature=self.temperature)
            weights = np.ones_like(transitions['r']).copy()
        elif self.prioritization == 'tderror':
            transitions, weights, idxs = self.buffer.sample(self.batch_size, beta=self.beta_schedule.value(t))
        else:
            transitions = self.buffer.sample(self.batch_size)
            weights = np.ones_like(transitions['r']).copy()

        o, o_2, g = transitions['o'], transitions['o_2'], transitions['g']
        ag, ag_2 = transitions['ag'], transitions['ag_2']
        transitions['o'], transitions['g'] = self._preprocess_og(o, ag, g)
        transitions['o_2'], transitions['g_2'] = self._preprocess_og(o_2, ag_2, g)

        transitions['w'] = weights.flatten().copy()  # note: ordered dict
        transitions_batch = [transitions[key] for key in self.stage_shapes.keys()]
        if self.prioritization == 'tderror':
            return (transitions_batch, idxs)
        else:
            return transitions_batch

    def stage_batch(self, t, batch=None): 
        if batch is None:
            if self.prioritization == 'tderror':
                batch, idxs = self.sample_batch(t)
            else:
                batch = self.sample_batch(t)
        assert len(self.buffer_ph_tf) == len(batch)
        self.sess.run(self.stage_op, feed_dict=dict(zip(self.buffer_ph_tf, batch)))

        if self.prioritization == 'tderror':
            return idxs

    def ddpg_train(self, t, dump_buffer, stage=True):
        if stage:
            if self.prioritization == 'tderror':
                idxs = self.stage_batch(t)
            else:
                self.stage_batch(t)

        self.critic_loss, self.actor_loss, Q_grad, pi_grad, td_error = self._grads()

        if self.prioritization == 'tderror':
            new_priorities = np.abs(td_error) + self.eps  # td_error
            if dump_buffer:
                T = self.buffer.buffers['u'].shape[1]
                episode_idxs = idxs // T
                t_samples = idxs % T
                batch_size = td_error.shape[0]
                with self.buffer.lock:
                    for i in range(batch_size):
                        self.buffer.buffers['td'][episode_idxs[i]][t_samples[i]] = td_error[i]

            self.buffer.update_priorities(idxs, new_priorities)
            
        # Update gradients for actor and critic networks
        self._update(Q_grad, pi_grad)

        # My variables
        self.visual_actor_loss = 1 - self.actor_loss
        self.critic_loss_episode.append(self.critic_loss)
        self.actor_loss_episode.append(self.visual_actor_loss)

        # print("Critic loss: ", self.critic_loss, " Actor loss: ", self.actor_loss)
        return self.critic_loss, np.mean(self.actor_loss)

    def _init_target_net(self):
        self.sess.run(self.init_target_net_op)

    def ddpg_update_target_net(self):
        self.critic_loss_avg = np.mean(self.critic_loss_episode)
        self.actor_loss_avg = np.mean(self.actor_loss_episode)

        self.sess.run(self.update_target_net_op)

    def clear_buffer(self):
        self.buffer.clear_buffer()

    def _vars(self, scope):
        res = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.scope + '/' + scope)
        assert len(res) > 0
        return res

    def _global_vars(self, scope):
        res = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.scope + '/' + scope)
        return res

    def _create_network(self, reuse=False):
        logger.info("Creating a DDPG agent with action space %d x %s..." % (self.dimu, self.action_scale))
        self.sess = tf_util.get_session()

        # running averages
        with tf.variable_scope('o_stats') as variable_scope:
            if reuse:
                variable_scope.reuse_variables()
            self.o_stats = Normalizer(self.dimo, self.norm_eps, self.norm_clip, sess=self.sess)
        with tf.variable_scope('g_stats') as variable_scope:
            if reuse:
                variable_scope.reuse_variables()
            self.g_stats = Normalizer(self.dimg, self.norm_eps, self.norm_clip, sess=self.sess)

        # mini-batch sampling.
        batch = self.staging_tf.get()
        batch_tf = OrderedDict([(key, batch[i])
                                for i, key in enumerate(self.stage_shapes.keys())])
        batch_tf['r'] = tf.reshape(batch_tf['r'], [-1, 1])

        # choose only the demo buffer samples
        mask = np.concatenate((np.zeros(self.batch_size - self.demo_batch_size), np.ones(self.demo_batch_size)), axis=0)

        # networks
        with tf.variable_scope('main') as variable_scope:
            if reuse:
                variable_scope.reuse_variables()

            # Create actor critic network
            self.main = self.create_actor_critic(batch_tf, net_type='main', **self.__dict__)
            variable_scope.reuse_variables()

        with tf.variable_scope('target') as variable_scope:
            if reuse:
                variable_scope.reuse_variables()
            target_batch_tf = batch_tf.copy()
            target_batch_tf['o'] = batch_tf['o_2']
            target_batch_tf['g'] = batch_tf['g_2']
            self.target = self.create_actor_critic(
                target_batch_tf, net_type='target', **self.__dict__)
            variable_scope.reuse_variables()
        assert len(self._vars("main")) == len(self._vars("target"))

        # loss functions
        target_critic_actor_tf = self.target.critic_with_actor_tf
        clip_range = (-self.clip_return, 0. if self.clip_pos_returns else np.inf)

        target_tf = tf.clip_by_value(batch_tf['r'] + self.gamma * target_critic_actor_tf, *clip_range)

        # MSE of target_tf - critic_tf. This is the TD Learning step
        self.td_error_tf = tf.stop_gradient(target_tf) - self.main.critic_tf
        self.critic_loss_tf = tf.reduce_mean(tf.square(tf.stop_gradient(target_tf) - self.main.critic_tf))

        #
        self.actor_loss_tf = -tf.reduce_mean(self.main.critic_with_actor_tf)
        self.actor_loss_tf += self.action_l2 * tf.reduce_mean(tf.square(self.main.actor_tf / self.action_scale))

        # Constructs symbolic derivatives of sum of critic_loss_tf vs _vars('main/Q')
        critic_grads_tf = tf.gradients(self.critic_loss_tf, self._vars('main/Q'))
        actor_grads_tf = tf.gradients(self.actor_loss_tf, self._vars('main/pi'))
        assert len(self._vars('main/Q')) == len(critic_grads_tf)
        assert len(self._vars('main/pi')) == len(actor_grads_tf)
        self.critic_grads_vars_tf = zip(critic_grads_tf, self._vars('main/Q'))
        self.actor_grads_vars_tf = zip(actor_grads_tf, self._vars('main/pi'))

        # Flattens variables and their gradients.
        self.critic_grads = flatten_grads(grads=critic_grads_tf, var_list=self._vars('main/Q'))
        self.actor_grads = flatten_grads(grads=actor_grads_tf, var_list=self._vars('main/pi'))

        # optimizers
        self.critic_optimiser = MpiAdam(self._vars('main/Q'), scale_grad_by_procs=False)
        self.actor_optimiser = MpiAdam(self._vars('main/pi'), scale_grad_by_procs=False)

        # polyak averaging used to update target network
        self.main_vars = self._vars('main/Q') + self._vars('main/pi')
        self.target_vars = self._vars('target/Q') + self._vars('target/pi')
        self.stats_vars = self._global_vars('o_stats') + self._global_vars('g_stats')

        # list( map( lambda( assign() ), zip()))
        self.init_target_net_op = list(
            map(    # Apply lambda to each item item in the zipped list
                lambda v: v[0].assign(v[1]),
                zip(self.target_vars, self.main_vars))
            )

        # Polyak-Ruppert averaging where most recent iterations are weighted more than past iterations.
        self.update_target_net_op = list(
            map(    # Apply lambda to each item item in the zipped list
                lambda v: v[0].assign(self.polyak * v[0] + (1. - self.polyak) * v[1]),  # polyak averaging
                zip(self.target_vars, self.main_vars))  # [(target_vars, main_vars), (), ...]
            )

        # initialize all variables
        tf.variables_initializer(self._global_vars('')).run()
        self._sync_optimizers()
        self._init_target_net()

    def logs(self, prefix=''):
        logs = []
        logs += [('actor_critic/critic_loss', self.critic_loss_avg)]
        logs += [('actor_critic/actor_loss', self.actor_loss_avg)]

        logs += [('stats_o/mean', np.mean(self.sess.run([self.o_stats.mean])))]
        logs += [('stats_o/std', np.mean(self.sess.run([self.o_stats.std])))]
        logs += [('stats_g/mean', np.mean(self.sess.run([self.g_stats.mean])))]
        logs += [('stats_g/std', np.mean(self.sess.run([self.g_stats.std])))]

        # logs += [('critic_loss', np.mean(self.sess.run([self.critic_loss])))]
        # logs += [('actor_loss', np.mean(self.sess.run([self.actor_loss])))]

        if prefix != '' and not prefix.endswith('/'):
            return [(prefix + '/' + key, val) for key, val in logs]
        else:
            return logs

    def __getstate__(self):
        """Our policies can be loaded from pkl, but after unpickling you cannot continue training.
        """
        excluded_subnames = ['_tf', '_op', '_vars', '_adam', 'buffer', 'sess', '_stats',
                             'main', 'target', 'lock', 'env', 'sample_transitions',
                             'stage_shapes', 'create_actor_critic']

        state = {k: v for k, v in self.__dict__.items() if all([not subname in k for subname in excluded_subnames])}
        state['buffer_size'] = self.buffer_size
        state['tf'] = self.sess.run([x for x in self._global_vars('') if 'buffer' not in x.name])
        return state

    def __setstate__(self, state):
        if 'sample_transitions' not in state:
            # We don't need this for playing the policy.
            state['sample_transitions'] = None

        self.__init__(**state)
        # set up stats (they are overwritten in __init__)
        for k, v in state.items():
            if k[-6:] == '_stats':
                self.__dict__[k] = v
        # load TF variables
        vars = [x for x in self._global_vars('') if 'buffer' not in x.name]
        assert(len(vars) == len(state["tf"]))
        node = [tf.assign(var, val) for var, val in zip(vars, state["tf"])]
        self.sess.run(node)

    def save(self, save_path):
        tf_util.save_variables(save_path)
