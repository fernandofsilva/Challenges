#%%
import numpy as np
from sklearn.utils import resample
import scipy.stats as stats

# Define the vector
x = np.array([4, 13, 9, 9, 8, 8, 11, 10, 12, 7, 20, 11, 7, 16, 13])

def bootstrap_replicate_1d(data, func):
    return func(np.random.choice(data, size=len(data)))

def draw_bs_reps(data, func, size=1):
    """Draw bootstrap replicates."""

    # Initialize array of replicates: bs_replicates
    bs_replicates = np.empty(size)

    # Generate replicates
    for i in range(size):
        bs_replicates[i] = bootstrap_replicate_1d(data, func)

    return bs_replicates

# Draw bootstrap replicates of the mean no-hitter time (equal to tau): bs_replicates
bs_replicates = draw_bs_reps(x, np.mean, size=1000)

# Compute the 95% confidence interval: conf_int
conf_int = np.percentile(bs_replicates, (2.5, 97.5))

# Print the confidence interval
print('95% confidence interval =', conf_int)