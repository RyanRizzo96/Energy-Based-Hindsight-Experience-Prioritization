import tensorflow as tf
from baselines.her.util import store_args, create_nerual_net


class ActorCritic:
    @store_args
    def __init__(self, inputs_tf, dimo, dimg, dimu, action_scale, o_stats, g_stats, hidden, layers,
                 **kwargs):
        """The actor-critic network and related training code.
        Args:
            inputs_tf (dict of tensors): all necessary inputs for the network: the
                observation (o), the goal (g), and the action (u)
            dimo (int): the dimension of the observations
            dimg (int): the dimension of the goals
            dimu (int): the dimension of the actions
            action_scale (float): the maximum magnitude of actions; action outputs will be scaled
                accordingly
            o_stats (baselines.her.Normalizer): normalizer for observations
            g_stats (baselines.her.Normalizer): normalizer for goals
            hidden (int): number of hidden units that should be used in hidden layers
            layers (int): number of hidden layers
        """

        # Calls util.py stpre_args(method)
        self.obs = inputs_tf['o']
        self.goals = inputs_tf['g']
        self.actions = inputs_tf['u']
        # print("INIT Actor-Critic with", hidden, "hidden units and ", layers, "hidden layers")
        # print("inputs_tf['u']", self.actions)
        # print("inputs_tf['o']", self.o_tf)
        # print("inputs_tf['g']", self.goals)

        # Prepare inputs for actor and critic.
        observations = self.o_stats.normalize(self.obs)
        goals = self.g_stats.normalize(self.goals)

        # Actor receives observation and goal to improve policy
        input_actor = tf.concat(axis=1, values=[observations, goals])  # for actor

        # Creates actor Actor network (updated by Policy gradient)
        with tf.variable_scope('pi'):
            self.actor_tf = self.action_scale * tf.tanh(create_nerual_net(
                input_actor, [self.hidden] * self.layers + [self.dimu]))

        # Creates actor Critic network
        with tf.variable_scope('Q'):
            # Critic receives obs, goals from env and output of actor as inputs
            input_critic_actor = tf.concat(axis=1, values=[observations, goals, self.actor_tf / self.action_scale])

            # For policy training - used to compute gradient for the actor.
            self.critic_with_actor_tf = create_nerual_net(input_critic_actor, [self.hidden] * self.layers + [1])

            # Critic - Updated by TD Learning and minimizing the MBSE
            # Used
            input_critic = tf.concat(axis=1, values=[observations, goals, self.actions / self.action_scale])
            self._input_critic = input_critic  # exposed for tests
            self.critic_tf = create_nerual_net(input_critic, [self.hidden] * self.layers + [1], reuse=True)

            # critic_with_actor_tf - critic_tf for TD Learning
