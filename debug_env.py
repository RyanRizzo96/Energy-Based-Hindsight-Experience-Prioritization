# import gym
#
# env = gym.make('FetchReach-v1')
# for i_episode in range(20):
#     observation = env.reset()
#     for t in range(100):
#         env.render()
#         print(observation)
#         action = env.action_space.sample()
#         observation, reward, done, info = env.step(action)
#         print("reward", reward)
#         print("info", info)
#         if done:
#             print("Episode finished after {} timesteps".format(t+1))
#             break
# env.close()

from hunter import trace, Q, Debugger
from pdb import Pdb
import subprocess

trace(
    # drop into a Pdb session on``myscript.mainMethod()`` call
    subprocess.call(['./run_bash.sh'])
)

