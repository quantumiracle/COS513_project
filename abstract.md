The project investigates the latent dynamics identification with observed trajectories based on variational inference methods. Previous works like Deep Variational Bayes Filters (DVBF) [1], Switching Linear Dynamical Systems (SLDS) [2] and Switching Nonlinear Dynamical Systems (SNLDS) [3] have demonstrated the effectiveness on learning latent state transition for model-based reinforcement learning. We will investigate the methods based on vanilla auto-encoder or variational auto-encoder together with above methods for latent dynamics inference with simulated trajectories of a robot arm. Probabilistic representation of the latent variables will be applied for handling the uncertainty in dynamics inference with limited observations. Finally, the inference of latent dynamics will be used in the downstream control tasks on the robot with reinforcement learning under varying dynamics situations.



[1]. Deep variational Bayes filters: Unsupervised learning of state space models from raw data

[2]. Switching Linear Dynamics for Variational Bayes Filtering

[3]. Collapsed amortized variational inference for switching nonlinear dynamical systems
