import gym
env = gym.make('FetchPickAndPlace-v1')
print("Action Space: {}".format(env.action_space))
print("Observation Space: {}".format(env.observation_space))

for i_episode in range(20):
    observation = env.reset()
    for t in range(100):
        env.render()
        print(observation)
        action = env.action_space.sample()
        observation, reward, done, info = env.step(action)
        if done:
            print("Episode finished after {} timesteps".format(t+1))
            break
env.close()