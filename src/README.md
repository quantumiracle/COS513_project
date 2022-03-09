# UPESI

Code for paper [Not Only Domain Randomization: Universal Policy with Embedding System Identification](https://arxiv.org/abs/2109.13438).

## Installation

For isaac-gym installation:

Go to [https://github.com/major-second/isaac-gym](https://github.com/major-second/isaac-gym), run `./create_conda_env_rlgpu.sh` to create a new conda env `rlgpu`  with isaac-gym installed. 

For the error

```
ImportError: libpython3.7m.so.1.0: cannot open shared object file: No such file or directory
```

Go to the folder of `rlgpu` env under conda directory, find `libpython3.7m.so.1.0` and run `sudo cp libpython3.7m.so.1.0 /usr/lib`.

`source activate rlgpu` after installation.



For [IsaacGymEnvs]([GitHub - NVIDIA-Omniverse/IsaacGymEnvs: Isaac Gym Reinforcement Learning Environments](https://github.com/NVIDIA-Omniverse/IsaacGymEnvs)) installation:

Just clone and `pip install -e .`

The relative paths of `isaacgymenvs/tasks/` may need to be fixed before installation.



This repo uses the same environment as [another repo](https://github.com/quantumiracle/Robotic_Door_Opening_with_Tactile_Simulation). Please follow the steps there to install our modified version of robosuite.



Then for the rest dependency, run:

`pip install -r requirements.txt`

## Citation:

Please cite the our paper if you make use of this repo:

```
@article{ding2021not,
  title={Not Only Domain Randomization: Universal Policy with Embedding System Identification},
  author={Ding, Zihan},
  journal={arXiv preprint arXiv:2109.13438},
  year={2021}
}
```

## Training Procedure

 For the universal policy (UP) with embedding system identification (ESI), we use the following commands.

First pretrained models are needed for each environment to rollout samples for further usage (learn the dynamics prediction in our method):

0. Get pretrained model
   
   ```
   python train.py --train --env inverteddoublependulum --process 1 --alg td3
   ```
   
   as an example for the InvertedDoublePendulum environment, using TD3 algorithm for training. After training, there will be weights in the data folder. You just need to replace the model path in later scripts with the one you got to make it run.

Go to the directory:

```bash
 cd dynamics_predict
```

1. Collect training and testing dataset
   
   ```bash
   python train_dynamics.py --collect_train_data --env Env_NAME
   python train_dynamics.py --collect_test_data --env Env_NAME
   ```

2. Normailize data
   Run
   
   ```bash
   cd ../data/dynamics_data
   jupyter notebook
   ```
   
   and open ```data_process_*ENV_NAME*.ipynb``` and go through each cell.

3. Train dynamics embedding (encoder, decoder and dynamics prediction model)
   
   Back to the terminal in ```dynamics_predict/```.
   
   Run the following to lauch training,
   
   ```bash
   python train_dynamics.py --train_embedding --env Env_NAME
   ```
   
   and use launch ```tensorboard --logdir runs``` to monitor the training process. 

4. Test learned encoder and dynamics predictor
   Test the preformance of learned encoder and dynamics predictor by applying them in ESI on collected test data:
   
   ```bash
   jupyter notebook
   ```
   
   and open ```test_dynamics_*ENV_NAME*.ipynb``` and go through each cell, including a Bayesian optimization (BO) process.

5. Train UP
   
   ```bash
   cd ..
   python train.py --train --env *ENV_NAME*dynamics --process NUM 
   ```
   
   Select the encoder-decoder type in `./environment/*ENV_NAME*dynamics.py` to match with the one used in `./dynamics_predict/train_dynamics.py`.

6. Test ESI with UP against other methods
   
   ```bash
   cd dynamics_predict
   python compare_methods_*ENV_NAME*.py
   ```
