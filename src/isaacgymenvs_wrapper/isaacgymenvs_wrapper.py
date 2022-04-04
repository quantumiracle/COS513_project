from numpy import ndarray
from omegaconf import DictConfig, OmegaConf
from torch import Tensor
# Resolvers used in hydra configs (see https://omegaconf.readthedocs.io/en/2.1_branch/usage.html#resolvers)
OmegaConf.register_new_resolver('eq', lambda x, y: x.lower()==y.lower(), replace=True)
OmegaConf.register_new_resolver('contains', lambda x, y: x.lower() in y.lower(), replace=True)
OmegaConf.register_new_resolver('if', lambda pred, a, b: a if pred else b, replace=True)
# allows us to resolve default arguments which are copied in multiple places in the config. used primarily for
# num_ensv
OmegaConf.register_new_resolver('resolve_default', lambda default, arg: default if arg=='' else arg, replace=True)

from rl_games.common import env_configurations, vecenv

import sys
path_old = sys.path.copy()
# if '' in sys.path:
#     sys.path.remove('')
# if './IsaacGymEnvs/isaacgymenvs' not in sys.path:
#     sys.path.insert(0, './IsaacGymEnvs/isaacgymenvs')
from isaacgymenvs.utils.reformat import omegaconf_to_dict
from isaacgymenvs.utils.rlgames_utils import RLGPUEnv, get_rlgames_env_creator
sys.path = path_old

class IsaacGymEnv:
    def __init__(self, env_name, hydra_cfg) -> None:
        # `create_rlgpu_env` is environment construction function which is passed to RL Games and called internally.
        # We use the helper function here to specify the environment config.
        create_rlgpu_env = get_rlgames_env_creator(
            omegaconf_to_dict(hydra_cfg.task),
            hydra_cfg.task_name,
            hydra_cfg.sim_device,
            hydra_cfg.rl_device,
            hydra_cfg.graphics_device_id,
            hydra_cfg.headless,
            multi_gpu=hydra_cfg.multi_gpu,
        )

        vecenv.register('RLGPU',
                        lambda config_name, num_actors, **kwargs: RLGPUEnv(config_name, num_actors, **kwargs))
        env_configurations.register(env_name, {
            'vecenv_type': 'RLGPU',
            'env_creator': lambda **kwargs: create_rlgpu_env(**kwargs),
        })

        config = hydra_cfg.train.params.config
        num_actors = config['num_actors']
        env_config = config.get('env_config', {})
        env = vecenv.create_vec_env(env_name, num_actors, **env_config)
        env_info = env.get_env_info()
        self.env = env
        self.observation_space = env_info['observation_space']
        self.action_space = env_info['action_space']

    def reset(self):
        state_dict = self.env.reset()
        state_gpu_tensor: Tensor = state_dict['obs']
        return state_gpu_tensor.detach().cpu().numpy()

    def step(self, action: ndarray):
        action = Tensor(action).cuda()
        next_state, reward, done, info = self.env.step(action)

        next_state = next_state['obs'].detach().cpu().numpy()
        reward = reward.detach().cpu().numpy()
        done = done.detach().cpu().numpy()

        return next_state, reward, done, info