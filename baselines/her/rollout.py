from collections import deque

import numpy as np
import pickle

from baselines.her.util import convert_episode_to_batch_major, store_args


class RolloutWorker:

    @store_args
    def __init__(self, venv, policy, dims, logger, T, rollout_batch_size=1,
                 exploit=False, use_target_net=False, compute_Q=False, noise_eps=0,
                 random_eps=0, history_len=100, render=False, monitor=False, **kwargs):
        """
        Rollout worker generates experience by interacting with one or many environments.
        Args:
            :param venv: vectorized gym environments.
            :param policy (object): the policy that is used to act
            :param dims (dict of ints): the dimensions for observations (o), goals (g), and actions (u)
            :param logger (object): the logger that is used by the rollout worker
            :param rollout_batch_size (int): the number of parallel rollouts that should be used
            :param exploit (boolean): whether or not to exploit, i.e. to act optimally according to the
                current policy without any exploration
            :param use_target_net (boolean): whether or not to use the target net for rollouts
            :param compute_Q (boolean): whether or not to compute the Q values alongside the actions
            :param noise_eps (float): scale of the additive Gaussian noise
            :param random_eps (float): probability of selecting a completely random action
            :param history_len (int): length of history for statistics smoothing
            :param render (boolean): whether or not to render the rollouts
        """

        assert self.T > 0

        self.info_keys = [key.replace('info_', '') for key in dims.keys() if key.startswith('info_')]
        print("dims.items() ", dims.items())  # prints keys and values

        self.success_history = deque(maxlen=history_len)
        self.Q_history = deque(maxlen=history_len)
        self.reward_history = deque(maxlen=history_len)

        self.n_episodes = 0
        self.reset_all_rollouts()
        self.clear_history()
        self.episode_counter = 0
        self.episode_reward = 0
        # print("Episode Reward: ", type(self.episode_reward))
        # print("Episode Counter: ", type(self.episode_counter))

    def reset_all_rollouts(self):
        self.obs_dict = self.venv.reset()
        self.initial_o = self.obs_dict['observation']
        self.initial_ag = self.obs_dict['achieved_goal']
        self.g = self.obs_dict['desired_goal']

    def generate_rollouts(self):

        """Performs `rollout_batch_size` rollouts in parallel for time horizon `T` with the current
        policy acting on it accordingly.
        """
        self.reset_all_rollouts()

        # compute observations. Initialize array of zeros
        observations = np.empty((self.rollout_batch_size, self.dims['o']), np.float32)  # observations
        ag = np.empty((self.rollout_batch_size, self.dims['g']), np.float32)  # achieved goals
        # Whole array assigned
        observations[:] = self.initial_o
        ag[:] = self.initial_ag

        # generate episodes
        obs, achieved_goals, acts, goals, successes, ep_reward_list = [], [], [], [], [], []

        ep_reward = 0
        env_step_counter = 0
        dones = []

        # print(self.info_keys)
        info_values = [np.empty((self.T - 1, self.rollout_batch_size, self.dims['info_' + key]), np.float32) for key in self.info_keys]
        Qs = []

        # Do for rollout time horizon. This is equal to 50 because episode length is 50
        # Not really much use if we go for a longer trajectory. If anything, shorten and test results
        # TODO: Shorten trajectory and check results
        for t in range(self.T):
            policy_output = self.policy.get_actions(
                observations, ag, self.g,
                compute_Q=self.compute_Q,
                noise_eps=self.noise_eps if not self.exploit else 0.,
                random_eps=self.random_eps if not self.exploit else 0.,
                use_target_net=self.use_target_net)

            if self.compute_Q:  # Evaluator  only
                action, Q = policy_output
                Qs.append(Q)
            else:
                action = policy_output

            if action.ndim == 1:
                # The non-batched case should still have a reasonable shape.
                action = action.reshape(1, -1)

            new_observation = np.empty((self.rollout_batch_size, self.dims['o']))
            new_achieved_goal = np.empty((self.rollout_batch_size, self.dims['g']))
            success = np.zeros(self.rollout_batch_size)

            # compute new states and observations
            obs_dict_new, reward, done, info = self.venv.step(action)

            # obs_dict_new {'achieved_goal': array([[1.3519502 , 0.73200333, 0.5274352 ]], dtype=float32),
            # 'desired_goal': array([[1.2729537 , 0.62809974, 0.51270455]], dtype=float32),
            # 'observation': array([[ 1.3519502e+00,  7.3200333e-01,  5.2743518e-01,  0.0000000e+00,
            # 0.0000000e+00,  1.7498910e-03, -3.6469495e-03, -1.8837147e-03,
            # -5.2045716e-06,  1.0831429e-04]], dtype=float32)}
            # reward [-1.]
            # info [{'is_success': 0.0}]

            # print(reward)
            # ep_reward_list.append(ep_reward)
            ep_reward += reward
            env_step_counter += 1
            # print("env_step_counter, ep_reward ", env_step_counter, ep_reward)

            new_observation = obs_dict_new['observation']
            new_achieved_goal = obs_dict_new['achieved_goal']
            success = np.array([i.get('is_success', 0.0) for i in info])

            if any(done):
                # here we assume all environments are done is ~same number of steps, so we terminate rollouts whenever
                # any of the envs returns done
                # trick with using vecenvs is not to add the obs from the environments that are "done", because those
                # are already observations after a reset
                break

            for i, info_dict in enumerate(info):
                for idx, key in enumerate(self.info_keys):
                    info_values[idx][t, i] = info[i][key]

            if np.isnan(new_observation).any():
                self.logger.warn('NaN caught during rollout generation. Trying again...')
                self.reset_all_rollouts()
                return self.generate_rollouts()

            dones.append(done)
            obs.append(observations.copy())
            achieved_goals.append(ag.copy())
            successes.append(success.copy())
            acts.append(action.copy())
            goals.append(self.g.copy())
            observations[...] = new_observation
            ag[...] = new_achieved_goal

        self.episode_counter += 1
        self.episode_reward = ep_reward[-1]  # Appending total ep_reward to episode_reward
        # print("episode_counter, episode_reward", self.episode_counter, self.episode_reward)

        obs.append(observations.copy())
        achieved_goals.append(ag.copy())

        episode = dict(o=obs,
                       u=acts,
                       g=goals,
                       ag=achieved_goals)

        for key, value in zip(self.info_keys, info_values):
            # print(key, value)
            episode['info_{}'.format(key)] = value

        # stats
        successful = np.array(successes)[-1, :]
        assert successful.shape == (self.rollout_batch_size,)
        success_rate = np.mean(successful)

        # print("success_rate: ", success_rate)
        self.success_history.append(success_rate)  # Used for tensorboard
        # print(self.success_history)

        self.reward_history.append(self.episode_reward)  # Used for tensorboard
        # print(self.reward_history)

        if self.compute_Q:  # Evaluator only
            self.Q_history.append(np.mean(Qs))

        self.n_episodes += self.rollout_batch_size

        # print("Rollout Done")

        return convert_episode_to_batch_major(episode)

    def clear_history(self):
        """Clears all histories that are used for statistics
        """
        # print("New worker")
        self.episode_counter = 0
        self.success_history.clear()
        self.reward_history.clear()
        self.Q_history.clear()

    def current_success_rate(self):
        return np.mean(self.success_history)

    def current_mean_Q(self):
        return np.mean(self.Q_history)

    def save_policy(self, path):
        """Pickles the current policy for later inspection.
        """
        with open(path, 'wb') as f:
            pickle.dump(self.policy, f)

    def logs(self, prefix='worker'):
        """Generates a dictionary that contains all collected statistics.
        """

        # print("self.reward_history", self.reward_history)

        logs = []
        logs += [('success_rate', np.mean(self.success_history))]
        # print(np.mean(self.success_history))

        logs += [('avg_episode_reward', np.mean(self.reward_history))]
        # print("Inside Log", np.mean(self.reward_history))

        if self.compute_Q:  # Evaluator only
            logs += [('mean_Q_val', np.mean(self.Q_history))]

        logs += [('episode', self.n_episodes)]

        if prefix != '' and not prefix.endswith('/'):
            return [(prefix + '/' + key, val) for key, val in logs]
        else:
            return logs

    def seed(self, seed):
        """Seeds each environment with a distinct seed derived from the passed in global seed.
        """
        for idx, env in enumerate(self.envs):
            env.seed(seed + 1000 * idx)