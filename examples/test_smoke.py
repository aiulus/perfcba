from ..SCM import LinearGaussianSCM, Intervention
from ..Bandit import SCMBandit
from ..Experiment import Experiment, RunConfig
from ..classical_bandits import ExploreThenCommit
import numpy as np

nodes = ["X", "Y"]
W = np.array([[0.0, 0.0],
              [1.5, 0.0]])  # edge X -> Y with weight 1.5 (remember W[i,j] is j->i)
c = np.array([0.0, 0.0])
mu_u = np.array([0.0, 0.0])
Sigma_u = np.diag([0.2**2, 0.2**2])

scm = LinearGaussianSCM(nodes, W, c, mu_u, Sigma_u)
arms = [Intervention("do(X=0)", hard={"X": 0.0}),
        Intervention("do(X=1)", hard={"X": 1.0})]

bandit = SCMBandit(scm, arms, reward_node="Y", observe="parents", feedback="causal")
policy = ExploreThenCommit(tau=0.3)
hist = Experiment(bandit, policy, RunConfig(T=500, seed=123)).run()

# Now Profiler can compute regret etc. using exact μ(Y|do(X)).
