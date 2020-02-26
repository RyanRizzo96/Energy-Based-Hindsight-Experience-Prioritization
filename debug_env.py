import gym

env = gym.make('FetchPickAndPlace-v1')
for i_episode in range(20):
    observation = env.reset()
    for t in range(100):
        env.render()
        print(achieved_goals)
        action = env.action_space.sample()
        observation, reward, done, info = env.step(action)
        print("reward", reward)
        print("info", info)
        if done:
            print("Episode finished after {} timesteps".format(t+1))
            break
env.close()

