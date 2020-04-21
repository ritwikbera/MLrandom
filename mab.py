import numpy as np 
import matplotlib
import matplotlib.pyplot as plt 
import seaborn as sns 
import scipy
from scipy import stats 

p_bandits = [0.45, 0.55, 0.60]

fig, axs = plt.subplots(5, 2, figsize=(8, 10))
axs = axs.flat

trials = [0, 0, 0]  
wins = [0, 0, 0] 

plots = [1, 2, 5, 10, 25, 50, 100, 200, 500, 1000]
x = np.linspace(0.001, .999, 100) # plotting discretization


n = plots[-1]
for step in range(1, n+1):
    bandit_priors = [stats.beta(a=1+w, b=1+t-w) for t, w in zip(trials, wins)]

    if step in plots:
        ax = next(axs)
        [ax.plot(x, prior.pdf(x)) for prior in bandit_priors]

    theta_samples = [d.rvs(1) for d in bandit_priors]

    chosen_bandit = np.argmax(theta_samples)
    trials[chosen_bandit] += 1
    wins[chosen_bandit] += int(np.random.rand() < p_bandits[chosen_bandit])

plt.show()