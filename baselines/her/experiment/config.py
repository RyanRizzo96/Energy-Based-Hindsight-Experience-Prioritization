import os
import numpy as np
import gym

from baselines import logger
from baselines.her.ddpg import DDPG
from baselines.her.her_sampler import make_sample_her_transitions, \
                                      make_sample_her_transitions_energy, \
                                      make_sample_her_transitions_prioritized_replay
from baselines.bench.monitor import Monitor

DEFAULT_ENV_PARAMS = {
    'FetchReach-v1': {
        'n_cycles': 10,  # per epoch
    },
}


DEFAULT_PARAMS = {
    # env
    'action_scale': 1.,  # max absolute value of actions on different coordinates

    # ddpg
    'layers': 3,  # number of layers in the critic/actor networks
    'hidden': 256,  # number of neurons in each hidden layers
    'network_class': 'baselines.her.actor_critic:ActorCritic',
    'Q_lr': 0.001,  # critic learning rate
    'pi_lr': 0.001,  # actor learning rate
    'buffer_size': int(1E6),  # (int) the max number of transitions to store, size of the replay buffer
    'polyak': 0.95,  # polyak averaging coefficient
    'action_l2': 1.0,  # quadratic penalty on actions (before rescaling by action_scale')
    'clip_obs': 200.,
    'scope': 'ddpg',  # can be tweaked for testing
    'relative_goals': False,

    # training
    'n_cycles': 50,  # per epoch
    'rollout_batch_size': 4,  # per mpi thread
    'n_batches': 40,  # training batches per cycle
    'batch_size': 256,  # per mpi thread, measured in transitions and reduced to even multiple of chunk_length.
    'n_test_rollouts': 10,  # number of test rollouts per epoch, each consists of rollout_batch_size rollouts
    'test_with_polyak': False,  # run test episodes with the target network

    # exploration
    'random_eps': 0.3,  # (float) Probability of taking a random action (as in an epsilon-greedy strategy)
                        # This is not needed for DDPG normally but can help exploring when using HER + DDPG.
                        # This hack was present in the original OpenAI Baselines repo (DDPG + HER)
    'noise_eps': 0.2,   # std of gaussian noise added to not-completely-random actions as a percentage of action_scale'

    # HER
    'replay_strategy': 'future',  # supported modes: future, none
    'replay_k': 4,  # number of additional goals used for replay, only used if off_policy_data=future

    # normalization
    'norm_eps': 0.01,  # epsilon used for observation normalization
    'norm_clip': 5,  # normalized observations are cropped to this values

    'bc_loss': 0,  # whether or not to use the behavior cloning loss as an auxilliary loss
    'q_filter': 0,  # whether or not a Q value filter should be used on the Actor outputs
    'num_demo': 100,  # number of expert demo episodes
    'demo_batch_size': 128,  # number of samples to be used from the demonstrations buffer, per mpi thread 128/1024 or 32/256
    'prm_loss_weight': 0.001,  # Weight corresponding to the primary loss
    'aux_loss_weight':  0.0078,  # Weight corresponding to the auxilliary loss also called the cloning loss

    # prioritized_replay (tderror)
    'alpha': 0.6,  # 0.6
    'beta0': 0.4,  # 0.4
    'beta_iters': None,  # None
    'eps': 1e-6,

    # energy-based prioritization
    'w_potential': 1.0,
    'w_linear': 1.0,
    'w_rotational': 1.0,
}


CACHED_ENVS = {}


def cached_make_env(make_env):
    """
    Only creates a new environment from the provided function if one has not yet already been
    created. This is useful here because we need to infer certain properties of the env, e.g.
    its observation and action spaces, without any intend of actually using it.
    """
    if make_env not in CACHED_ENVS:
        env = make_env()
        CACHED_ENVS[make_env] = env
    return CACHED_ENVS[make_env]


# Called in beginning of learn() in her.py
def prepare_params(kwargs):
    # DDPG params
    ddpg_params = dict()  # Create empty dictionary
    env_name = kwargs['env_name']

    def make_env(subrank=None):
        env = gym.make(env_name)  # Create gym environment

        # Check MPI rank, warm if single MPI process
        if subrank is not None and logger.get_dir() is not None:
            try:
                from mpi4py import MPI
                mpi_rank = MPI.COMM_WORLD.Get_rank()
            except ImportError:
                MPI = None
                mpi_rank = 0
                logger.warn('Running with a single MPI process. This should work, but the results may differ from the ones publshed in Plappert et al.')

            max_episode_steps = env._max_episode_steps  # Get maximum episode steps (50)

            # Pass to Monitor class which monitors the episode reward, length, time and other data.
            env = Monitor(env,
                          os.path.join(logger.get_dir(), str(mpi_rank) + '.' + str(subrank)),  # name of monitor.csv
                          allow_early_resets=True)

            # hack to re-expose _max_episode_steps (ideally should replace reliance on it downstream)
            env = gym.wrappers.TimeLimit(env, max_episode_steps=max_episode_steps)
        return env

    kwargs['make_env'] = make_env
    tmp_env = cached_make_env(kwargs['make_env'])
    assert hasattr(tmp_env, '_max_episode_steps')
    kwargs['T'] = tmp_env._max_episode_steps   # Setting time horizon

    # kwargs['T'] = 40
    print("TEST: ", kwargs['T'])

    kwargs['action_scale'] = np.array(kwargs['action_scale']) if isinstance(kwargs['action_scale'], list) else kwargs['action_scale']
    kwargs['gamma'] = 1. - 1. / kwargs['T']

    print ("GAMMA: ", kwargs['gamma'])
    if 'lr' in kwargs:
        kwargs['pi_lr'] = kwargs['lr']
        kwargs['Q_lr'] = kwargs['lr']
        del kwargs['lr']
    for name in ['buffer_size', 'hidden', 'layers',
                 'network_class',
                 'polyak',
                 'batch_size', 'Q_lr', 'pi_lr',
                 'norm_eps', 'norm_clip', 'action_scale',
                 'action_l2', 'clip_obs', 'scope', 'relative_goals',
                 'alpha', 'beta0', 'beta_iters', 'eps']:
        ddpg_params[name] = kwargs[name]
        kwargs['_' + name] = kwargs[name]
        del kwargs[name]
    kwargs['ddpg_params'] = ddpg_params

    return kwargs


# Print out params before training
def log_params(params, logger=logger):
    for key in sorted(params.keys()):
        logger.info('{}: {}'.format(key, params[key]))


def configure_her(params):
    print("Inside configure her")
    env = cached_make_env(params['make_env'])
    env.reset()

    def reward_fun(ag_2, g, info):  # vectorized
        return env.compute_reward(achieved_goal=ag_2, desired_goal=g, info=info)

    # Prepare configuration for HER.
    her_params = {
        'reward_fun': reward_fun,
    }
    for name in ['replay_strategy', 'replay_k']:
        her_params[name] = params[name]
        params['_' + name] = her_params[name]
        del params[name]

    if params['prioritization'] == 'energy':
        sample_her_transitions = make_sample_her_transitions_energy(**her_params)
    elif params['prioritization'] == 'tderror':
        sample_her_transitions = make_sample_her_transitions_prioritized_replay(**her_params)
    else:
        sample_her_transitions = make_sample_her_transitions(**her_params)

    return sample_her_transitions


def simple_goal_subtract(a, b):
    assert a.shape == b.shape
    return a - b


def configure_ddpg(dims, params, reuse=False, use_mpi=True, clip_return=True):
    sample_her_transitions = configure_her(params)
    # Extract relevant parameters.
    gamma = params['gamma']
    rollout_batch_size = params['rollout_batch_size']
    ddpg_params = params['ddpg_params']
    temperature = params['temperature']
    prioritization = params['prioritization']
    env_name = params['env_name']
    total_timesteps = params['total_timesteps']
    rank_method = params['rank_method']

    input_dims = dims.copy()

    # DDPG agent
    env = cached_make_env(params['make_env'])
    env.reset()
    ddpg_params.update({'input_dims': input_dims,  # agent takes an input observations
                        'T': params['T'],
                        'clip_pos_returns': True,  # clip positive returns
                        'clip_return': (1. / (1. - gamma)) if clip_return else np.inf,  # max abs of return
                        'rollout_batch_size': rollout_batch_size,
                        'subtract_goals': simple_goal_subtract,
                        'sample_transitions': sample_her_transitions,
                        'gamma': gamma,
                        'bc_loss': params['bc_loss'],
                        'q_filter': params['q_filter'],
                        'num_demo': params['num_demo'],
                        'demo_batch_size': params['demo_batch_size'],
                        'prm_loss_weight': params['prm_loss_weight'],
                        'aux_loss_weight': params['aux_loss_weight'],
                        'temperature': temperature,
                        'prioritization': prioritization,
                        'env_name': env_name,
                        'total_timesteps': total_timesteps,
                        'rank_method': rank_method,
                        })
    ddpg_params['info'] = {
        'env_name': params['env_name'],
    }

    # Call to initialise DDPG
    policy = DDPG(reuse=reuse, **ddpg_params, use_mpi=use_mpi)
    return policy


def configure_dims(params):
    env = cached_make_env(params['make_env'])
    env.reset()

    # take action in environment
    obs, _, _, info = env.step(env.action_space.sample())
    print('obs: {}'.format(obs))

    print('env.action_space: {}'.format(env.action_space))
    print('env.observation_space: {}'.format(env.observation_space))

    print('observation dim: {}'.format(obs['observation'].shape[0]))
    print('action space dim: {}'.format(env.action_space.shape[0]))
    print('desired goal dim: {}'.format(obs['desired_goal'].shape[0]))

    dims = {
        'o': obs['observation'].shape[0],
        'u': env.action_space.shape[0],
        'g': obs['desired_goal'].shape[0],
    }
    for key, value in info.items():
        value = np.array(value)
        if value.ndim == 0:
            value = value.reshape(1)
        dims['info_{}'.format(key)] = value.shape[0]
    return dims
