import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style="darkgrid")

# Load an example dataset with long-form data
df = pd.read_csv('/Users/ryanr/B.Eng/MCAST_Degree_4/Thesis/code/gym/RL_EBP/EBP_new_results/FPAP200k/trial_logdir/2020-02-25-18-52-13/progress.csv')
df.head()

# Plot the responses for different events and regions
# sns.lineplot(x="timepoint", y="signal",
#              hue="region", style="event",
#              data=df)