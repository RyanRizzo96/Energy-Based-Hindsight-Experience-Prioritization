import numpy as np


def make_sample_her_transitions(replay_strategy, replay_k, reward_fun):
    print("Inside make_sample_her_transitions: ")
    print("replay_strategy: ", replay_strategy)
    print("replay_k: ", replay_k)
    print("reward_fun: ", reward_fun)

    """Creates a sample function that can be used for HER experience replay.
    Args:
        replay_strategy (in ['future', 'none']): the HER replay strategy; if set to 'none',
            regular DDPG experience replay is used
        replay_k (int): the ratio between HER replays and regular replays (e.g. k = 4 -> 4 times
            as many HER replays as regular replays are used)
        reward_fun (function): function to re-compute the reward with substituted goals
    """
    if replay_strategy == 'future':
        future_p = 1 - (1. / (1 + replay_k))  # 0.8
    else:  # 'replay_strategy' == 'none'
        future_p = 0

    def _sample_her_transitions(episode_batch, batch_size_in_transitions):
        """episode_batch is {key: array(buffer_size x T x dim_key)}
        """
        T = episode_batch['u'].shape[1]  # 49

        # print("START HER Samlper ")
        # print("episode_batch['u'] ", episode_batch['u'])
        # print("T: ", T)

        rollout_batch_size = episode_batch['u'].shape[0]
        # print("rollout_batch_size: ", rollout_batch_size)

        batch_size = batch_size_in_transitions
        # print("batch_size: ", batch_size)

        # Select which episodes and time steps to use.
        episode_idxs = np.random.randint(0, rollout_batch_size, batch_size)
        # print("episode_idxs: ", episode_idxs)
        t_samples = np.random.randint(T, size=batch_size)
        # print("t_samples: ", t_samples)
        transitions = {key: episode_batch[key][episode_idxs, t_samples].copy()
                       for key in episode_batch.keys()}

        # Select future time indexes proportional with probability future_p. These
        # will be used for HER replay by substituting in future goals.
        her_indexes = np.where(np.random.uniform(size=batch_size) < future_p)
        # print("her_indexes: ", her_indexes)
        future_offset = np.random.uniform(size=batch_size) * (T - t_samples)
        future_offset = future_offset.astype(int)
        future_t = (t_samples + 1 + future_offset)[her_indexes]

        # Replace goal with achieved goal but only for the previously-selected
        # HER transitions (as defined by her_indexes). For the other transitions,
        # keep the original goal.
        future_ag = episode_batch['ag'][episode_idxs[her_indexes], future_t]
        transitions['g'][her_indexes] = future_ag

        # Reconstruct info dictionary for reward  computation.
        info = {}
        for key, value in transitions.items():
            if key.startswith('info_'):
                info[key.replace('info_', '')] = value

        # Re-compute reward since we may have substituted the goal.
        reward_params = {k: transitions[k] for k in ['ag_2', 'g']}
        reward_params['info'] = info
        transitions['r'] = reward_fun(**reward_params)

        transitions = {k: transitions[k].reshape(batch_size, *transitions[k].shape[1:])
                       for k in transitions.keys()}

        assert (transitions['u'].shape[0] == batch_size_in_transitions)

        # print("END HER Samlper ")
        return transitions

    return _sample_her_transitions


def make_sample_her_transitions_energy(replay_strategy, replay_k, reward_fun):
    if (replay_strategy == 'future') or (replay_strategy == 'final'):
        future_p = 1 - (1. / (1 + replay_k))
    else:
        future_p = 0

        is_print = 0

    def _sample_her_transitions(episode_batch, batch_size_in_transitions, rank_method, temperature, sample_count, 
                                cycle_count, update_stats=False):

        # print("Inside her samlper1")
        logger = False
        if sample_count % 40 == 0:
            logger = True
            # print("Sampler cycle count", cycle_count)
        
        # if cycle_count == 1:
               
        # print("SAMPLER cycle",  cycle_count)
            
        T = episode_batch['u'].shape[1]  # 50
        rollout_batch_size = episode_batch['u'].shape[0]    # 1- 10
        batch_size = batch_size_in_transitions  # 256

        # 256 random numbers between (0, rollout_batch_size)
        episode_idxs = np.random.randint(0, rollout_batch_size, batch_size)
        # 256 random number between 0 and 50
        t_samples = np.random.randint(T, size=batch_size)

        if not update_stats:
            if rank_method == 'none':
                energy_trajectory = episode_batch['e']
                energy_direction = episode_batch['ed']
                # print("Inside her samlper")
                # print(energy_direction, sample_count, cycle_count-1)
                if energy_direction[cycle_count-1] == 1:
                    if logger:
                        print(energy_direction, sample_count, cycle_count - 1)
                        print("Altering temperature to increase probability of sampling at index [", 
                              cycle_count-1, "]", energy_trajectory[cycle_count-1])
                    temperature = 0.7
                else:
                    temperature = 1
                # 
                # normalized_ed = energy_direction / np.sqrt(np.sum(energy_direction ** 2))
            else:
                energy_trajectory = episode_batch['p']

            p_trajectory = np.power(energy_trajectory, 1 / (temperature + 1e-2))  # traj / 0.9900990099009901
            p_trajectory_sum = p_trajectory / p_trajectory.sum()
            
            # p_trajectory_new = np.power(energy_trajectory + normalized_ed, 1 / (temperature + 1e-2))  # traj / 
            # 0.9900990099009901 
            # print("p_trajectory_NEW", p_trajectory_new) p_trajectory_new = p_trajectory_new / 
            # p_trajectory_new.sum() print("p_trajectory_NEW_sum ", p_trajectory_new) print("P traj: ", p_trajectory)
            # print("P traj NEW: ", p_trajectory_new) 
            episode_idxs_energy = np.random.choice(rollout_batch_size, size=batch_size, replace=True,
                                                   p=p_trajectory_sum.flatten())
            episode_idxs = episode_idxs_energy

            if logger:
                print("en traj", episode_batch['e'])
                print("p_trajectory", p_trajectory)
                print("p_trajectory_sum", p_trajectory_sum)
                print("energy_direction", energy_direction)
                print("Cycle count:", cycle_count-1)
                # print("normalized_ed", normalized_ed)

            if sample_count > 0:
                sample_count += 1

        transitions = {}
        for key in episode_batch.keys():
            if not key == 'd' and not key == 's' and not key == 'e' and not key == 'ed':
                transitions[key] = episode_batch[key][episode_idxs, t_samples].copy()

        # Select future time indexes proportional with probability future_p. These
        # will be used for HER replay by substituting in future goals.
        her_indexes = np.where(np.random.uniform(size=batch_size) < future_p)

        future_offset = np.random.uniform(size=batch_size) * (T - t_samples)
        future_offset = future_offset.astype(int)
        future_t = (t_samples + 1 + future_offset)[her_indexes]

        if replay_strategy == 'final':
            future_t[:] = T

        # Replace goal with achieved goal but only for the previously-selected
        # HER transitions (as defined by her_indexes). For the other transitions,
        # keep the original goal.
        future_ag = episode_batch['ag'][episode_idxs[her_indexes], future_t]

        transitions['g'][her_indexes] = future_ag

        # Reconstruct info dictionary for reward computation.
        info = {}
        for key, value in transitions.items():
            if key.startswith('info_'):
                info[key.replace('info_', '')] = value

        # Re-compute reward since we may have substituted the goal.
        reward_params = {k: transitions[k] for k in ['ag_2', 'g']}
        reward_params['info'] = info

        transitions['r'] = reward_fun(**reward_params)

        transitions = {k: transitions[k].reshape(batch_size, *transitions[k].shape[1:])
                       for k in transitions.keys()}

        assert (transitions['u'].shape[0] == batch_size_in_transitions)
    
        return transitions

    return _sample_her_transitions


def make_sample_her_transitions_prioritized_replay(replay_strategy, replay_k, reward_fun):
    if (replay_strategy == 'future') or (replay_strategy == 'final'):
        future_p = 1 - (1. / (1 + replay_k))
    else:
        future_p = 0

    def _sample_proportional(self, rollout_batch_size, batch_size, T):
        episode_idxs = []
        t_samples = []
        for _ in range(batch_size):
            self.n_transitions_stored = min(self.n_transitions_stored, self.size_in_transitions)
            mass = random.random() * self._it_sum.sum(0, self.n_transitions_stored - 1)
            idx = self._it_sum.find_prefixsum_idx(mass)
            assert idx < self.n_transitions_stored
            episode_idx = idx // T
            assert episode_idx < rollout_batch_size
            t_sample = idx % T
            episode_idxs.append(episode_idx)
            t_samples.append(t_sample)

        return (episode_idxs, t_samples)

    def _sample_her_transitions(self, episode_batch, batch_size_in_transitions, beta):
        """episode_batch is {key: array(buffer_size x T x dim_key)}
        """

        T = episode_batch['u'].shape[1]
        rollout_batch_size = episode_batch['u'].shape[0]
        batch_size = batch_size_in_transitions

        if rollout_batch_size < self.current_size:
            episode_idxs = np.random.randint(0, rollout_batch_size, batch_size)
            t_samples = np.random.randint(T, size=batch_size)
        else:
            assert beta >= 0
            episode_idxs, t_samples = _sample_proportional(self, rollout_batch_size, batch_size, T)
            episode_idxs = np.array(episode_idxs)
            t_samples = np.array(t_samples)

        weights = []
        p_min = self._it_min.min() / self._it_sum.sum()
        max_weight = (p_min * self.n_transitions_stored) ** (-beta)

        for episode_idx, t_sample in zip(episode_idxs, t_samples):
            p_sample = self._it_sum[episode_idx * T + t_sample] / self._it_sum.sum()
            weight = (p_sample * self.n_transitions_stored) ** (-beta)
            weights.append(weight / max_weight)

        weights = np.array(weights)

        transitions = {}
        for key in episode_batch.keys():
            if not key == "td" and not key == "e":
                episode_batch_key = episode_batch[key].copy()
                transitions[key] = episode_batch_key[episode_idxs, t_samples].copy()

        # Select future time indexes proportional with probability future_p. These
        # will be used for HER replay by substituting in future goals.
        her_indexes = np.where(np.random.uniform(size=batch_size) < future_p)

        future_offset = np.random.uniform(size=batch_size) * (T - t_samples)
        future_offset = future_offset.astype(int)
        future_t = (t_samples + 1 + future_offset)[her_indexes]

        if replay_strategy == 'final':
            future_t[:] = T

        # Replace goal with achieved goal but only for the previously-selected
        # HER transitions (as defined by her_indexes). For the other transitions,
        # keep the original goal.
        future_ag = episode_batch['ag'][episode_idxs[her_indexes], future_t]

        # Reconstruct info dictionary for reward computation.
        info = {}
        for key, value in transitions.items():
            if key.startswith('info_'):
                info[key.replace('info_', '')] = value

        reward_params = {k: transitions[k] for k in ['ag_2', 'g']}
        reward_params['info'] = info

        transitions['g'][her_indexes] = future_ag

        # Re-compute reward since we may have substituted the goal.
        reward_params = {k: transitions[k] for k in ['ag_2', 'g']}
        reward_params['info'] = info

        transitions['r'] = reward_fun(**reward_params)

        transitions = {k: transitions[k].reshape(batch_size, *transitions[k].shape[1:])
                       for k in transitions.keys()}

        assert (transitions['u'].shape[0] == batch_size_in_transitions)

        idxs = episode_idxs * T + t_samples

        return (transitions, weights, idxs)

    return _sample_her_transitions
