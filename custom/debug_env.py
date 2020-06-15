import gym
import numpy as np

env = gym.make('HandReach-v0')
for i_episode in range(20):

    observation = env.reset()
    achieved_goals = []
    # achieved_goals = np.array(achieved_goals)
    desired_goals = []
    # desired_goals = np.array(desired_goals)

    for t in range(100):
        env.render()
        # print(observation)
        action = env.action_space.sample()
        observation, reward, done, info = env.step(action)

        # achieved_goals = np.append(achieved_goals, observation['achieved_goal'])
        achieved_goals.append(observation['achieved_goal'])
        desired_goals = np.array(observation['desired_goal'])

        print("achieved_goal", achieved_goals[t])
        # print("achieved_goal_y", achieved_goals[1])
        print("desired_goal", desired_goals)

        # print("reward", reward)
        # print("info", info)

        if t == 9:
            np_ag = np.array(achieved_goals)
            print("AG", np_ag)
            # print("AG y axis", np_ag[:, 1])    # AG (10, 3)
            # print("AG x axis", np_ag[:, 0])  # AG (10, 3)

        if done:
            # print(achieved_goals)
            print("Episode finished after {} timesteps".format(t+1))

            break
env.close()

