2.03:

Write up the abstract of the project.

2.15:

Creat the environment wrapper for OpenAI gym env, add basic reinforcement learning components.

2.22:

Add the environment randomization part for supporting dynamics randomization.

3.03:

Finishing set up the environment and reinforcement learning algorithm like TD3. 

3.07:

Add support for the Isaac Gym environments.

3.13:

Fix problems in policy training with randomized environment, get the RL policy trained in randomized dynamics environments.

3.16:

Collect training and testing datasets on inverted-doublependulum environment;

Add switch linear dynamics (SLD) model with linear approximation (ipynb)

3.22:

Finish the SLD tests on the collected dataset, get primary results for latent dynamics prediction, it's okayish but not very accurate, also without uncertainty prediction.

4.1:

Explore the Pyro package for BNN/SVI as a probabilistic approach for latent dynamics inference.

4.9:

Tried a two-stage inference procedure for latent dynamics prediction: 1. training forward dynamics model and dynamic encoding model; 2. fix the two and use BNN/SVI for latent dynamics inference.

4.16:

Extensive tests with the two-stage procedure.

Several things found when using Pyro:

1. If using y=BNN (x, $\alpha$), and output y is sampled from a Gaussian with mean as deterministic forward dynamics prediction, the means $\alpha$pha is fitted well, and the variances of y can come from two parts: (i) the varian$\alpha$ alpha and (ii) the output Gaussian variance. And in this case, (i) is extremely small (about <10^-2) and (ii) is large (about 1.). (i) cannot cover the prediction error.

2. If using y=f(x, $\alpha$) as a deterministic function, the variance only comes fr$\alpha$pha, but in this cas$\alpha$ alpha is not fitted well (not accurate means) although it fits a large variance (aobut 1.).
   
   Neither of these is good to use.

4.20:

Finish the primary tests of Bayesian neural networks (BNN) for fitting the probabilistic distribution of the latent variables.

4.26:

Tune the performance of the method, try the one without dynamics encoding, i.e. using BNN for identifying $\theta$ directly with forward dynamics model trained. 

The results show that it doesn't perform well if directly infering $\theta$ instead of $\alpha$.
