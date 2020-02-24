from baselines.common import plot_util as plot
from baselines.common import plot_util as pu
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import csv

# osprint(os.environ['OPENAI_LOGDIR'])

# OPENAI_LOGDIR=".log/FetchReach/HER5k/trial1"

# results = plot.load_results(".log/HER/test_5k/2019-10-27-13-52-48")  # pass variable from bash script
# print(results)
# r = results[0]
# plt.plot(np.cumsum(r.monitor.l), plot.smooth(r.monitor.r, radius=1))  # radius to smoother
# print(r.monitor.l)
# plt.xlabel('Episodes')
# plt.ylabel('Reward')
# plt.show()

# Plotting with random seeds
# results = pu.load_results('/Users/ryanr/logs/her_seed/run_1')
# print(len(results))
# pu.plot_results(results)
# # pu.plot_results(results, average_group=True)    # average over all seeds
# pu.plot_results(results, average_group=True, split_fn=lambda _: '')     # plot both groups on the same graph


# results = pu.load_results('/Users/ryanr/B.Eng/MCAST_Degree_4/Thesis/code/gym/RL_baselines/.log/run_1/6_layer')


results = pu.load_results('/Users/ryanr/B.Eng/MCAST_Degree_4/Thesis/code/gym/RL_baselines/.log/two_seeds/run_2/aggregates/testsuccess_rate-tb-run_2.csv')

print(len(results))
pu.plot_results(results)
plt.show()
# r = results[0]
# plt.plot(np.cumsum(r.monitor.l), pu.smooth(r.monitor.r, radius=10))

pu.plot_results(results, average_group=True)    # average over all seeds
plt.show()

pu.plot_results(results, average_group=True, split_fn=lambda _: '')     # plot both groups on the same graph
plt.xlabel('Iteration Number')
plt.ylabel('Reward')
plt.title('Reward vs Iteration')
plt.show()
# We can disable either light shaded region (corresponding to standard deviation of the curves in the group)
# or darker shaded region (corresponding to the error in mean estimate) by using shaded_std=False or shaded_
# err=False options respectively.

pu.plot_results(results, average_group=True, split_fn=lambda _: '', shaded_std=False)
plt.show()
