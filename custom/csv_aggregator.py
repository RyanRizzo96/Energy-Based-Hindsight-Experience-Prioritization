import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import make_interp_spline, BSpline
import csv

headers = ['Mean', 'aMin', 'aMax', 'Median', 'std', 'var']
df = pd.read_csv(
    '/Users/ryanr/B.Eng/MCAST_Degree_4/Thesis/code/gym/RL_baselines/.log/two_seeds/run_2/aggregates/testsuccess_rate-tb-run_2.csv',
    sep=',',  dtype=float)

# Preview the first 5 lines of the loaded data
print(df.head(10))

count_row = df.shape[0]  # gives number of row count
count_col = df.shape[1]  # gives number of col count

print("rows", count_row)
print("cols", count_col)
# print(df.iloc[:,0])

# Obtaining std
std = df.iloc[:, 5]
x = df.iloc[:, 0]
y = df['mean']

# First three columns to obtain mean, min and max
for i in range(1, 4):
    plt.plot(df.iloc[:, 0], df.iloc[:,i], label='id %s' %i)

plt.legend()
plt.show()

# Plotting standard deviation
plt.plot(x, y, 'k-')
plt.fill_between(x, y-std, y+std, color='C0', alpha=0.3,  interpolate=True)
# plt.show()

# Plotting error in estimate of the mean [std/root(no.seeds)]
error = std/np.sqrt(6)
plt.plot(x, y, 'k-')
plt.fill_between(x, y-error, y+error, color='C1', alpha=0.3, interpolate=True)
plt.show()

# Interpolation
xnew = np.linspace(x.min(), x.max(), 40)
spl = make_interp_spline(x, y, k=3)  # type: BSpline
# Smoothing mean
mean_smooth = spl(xnew)

# Smoothing std
spl_std = make_interp_spline(x, std, k=3)  # type: BSpline
std_smooth = spl_std(xnew)

# Smoothing error of the mean
spl_err = make_interp_spline(x, error, k=3)  # type: BSpline
err_smooth = spl_err(xnew)

plt.plot(xnew, mean_smooth)
plt.fill_between(xnew, mean_smooth-std_smooth, mean_smooth+std_smooth, color='C0', alpha=0.3)
plt.fill_between(xnew, mean_smooth-err_smooth, mean_smooth+err_smooth, color='C1', alpha=0.3)
plt.ylabel('Reward ')
plt.xlabel('Episode number')
plt.title('Episode Reward')
plt.show()
