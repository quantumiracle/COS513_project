import isaacgym # must be imported before torch
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train or test neural net motor controller.')
    parser.add_argument('--train', dest='train', action='store_true', default=False)
    parser.add_argument('--test', dest='test', action='store_true', default=False)
    parser.add_argument('--env', type=str, help='Environment', required=True)
    parser.add_argument('--render', dest='render', action='store_true',
                    help='Enable openai gym real-time rendering')
    parser.add_argument('--process', type=int, default=1,
                    help='Process count for parallel exploration')
    parser.add_argument('--model', dest='path', type=str, default=None,
                help='Moddel weights location')
    parser.add_argument('--model_id', dest='model_id', type=int, default=0,
            help='Moddel weights id (step for saving the model)')
    parser.add_argument('--finetune', dest='finetune', action='store_true', default=False,
            help='Load a pretrained model and finetune it')
    parser.add_argument('--seed', dest='seed', type=int, default=1234,
            help='Random seed')
    parser.add_argument('--alg', dest='alg', type=str, default='td3',
                help='Choose algorithm type')
    
    env_name: str
    try:
        args = parser.parse_args()
    except:
        from omegaconf import DictConfig, OmegaConf
        # Resolvers used in hydra configs (see https://omegaconf.readthedocs.io/en/2.1_branch/usage.html#resolvers)
        OmegaConf.register_new_resolver('eq', lambda x, y: x.lower()==y.lower())
        OmegaConf.register_new_resolver('contains', lambda x, y: x.lower() in y.lower())
        OmegaConf.register_new_resolver('if', lambda pred, a, b: a if pred else b)
        # allows us to resolve default arguments which are copied in multiple places in the config. used primarily for
        # num_ensv
        OmegaConf.register_new_resolver('resolve_default', lambda default, arg: default if arg=='' else arg)
        import hydra

        import sys
        path_old = sys.path.copy()
        # if '' in sys.path:
        #     sys.path.remove('')
        # if './IsaacGymEnvs/isaacgymenvs' not in sys.path:
        #     sys.path.insert(0, './IsaacGymEnvs/isaacgymenvs')
        from isaacgymenvs.utils.utils import set_seed
        sys.path = path_old

        hydra_cfg: DictConfig
        @hydra.main(config_name='placeholder', config_path='placeholder')
        def get_hydra_cfg(cfg: DictConfig):
            # sets seed. if seed is -1 will pick a random one
            cfg.seed = set_seed(cfg.seed, torch_deterministic=cfg.torch_deterministic)
            global hydra_cfg
            hydra_cfg = cfg
        get_hydra_cfg()

        args = argparse.Namespace()
        args.env = hydra_cfg.task.name

        # todo: magic
        args.train = True
        args.test = False
        args.finetune = False
        args.path = None
        args.model_id = 0
        args.render = False
        args.process = 1
        args.seed = 42
        args.alg = 'td3'
    
    env_name = args.env
    print('Environment Name:', env_name)

    MUJOCO_ENVS = ['acrobot', 'cartpole', 'mountaincar', 'pendulum', 'inverteddoublependulumdisc', 'inverteddoublependulum', \
        'inverteddoublependulumdynamics', 'inverteddoublependulumdynamicsembedding', 'halfcheetah', 'halfcheetahdynamics', 'halfcheetahdynamicsembedding']
    from gym import envs as gym_envs
    gym_envs = gym_envs.registry.all()
    GYM_MOJUCO_ENVS = [env_spec.id for env_spec in gym_envs]
    ISAAC_ENVS = ['Cartpole', 'FrankaCabinet'] # todo: implement other envs

    env_type: str
    if env_name in MUJOCO_ENVS:
        env_type = 'our_mujoco'
    elif env_name in GYM_MOJUCO_ENVS:
        env_type = 'gym_mujoco'
    elif env_name in ISAAC_ENVS:
        env_type = 'isaac'
    else:
        raise ValueError('Environment {} not exists!'.format(args.env))
    print('Environment Type:', env_type)

    if env_type == 'our_mujoco':
        from environment import envs as our_envs_dic
        env = our_envs_dic[env_name]()

        hydra_cfg = {} # not using hydra config
    elif env_type == 'gym_mujoco':
        from gym import make
        env = make(env_name).unwrapped

        hydra_cfg = {} # not using hydra config
        our_envs_dic = {} # not using our envs
    elif env_type == 'isaac':
        from isaacgymenvs_wrapper.isaacgymenvs_wrapper import IsaacGymEnv
        env = IsaacGymEnv(env_name, hydra_cfg)

        our_envs_dic = {} # not using our envs
    
    print('Observation space: {}  Action space: {}'.format(env.observation_space, env.action_space))

    if args.alg=='td3':
        from rl.td3.train_td3 import train_td3
        train_td3(env, our_envs_dic, env_name, env_type, hydra_cfg, args.train, args.test, args.finetune, args.path, args.model_id, args.render, args.process, args.seed)
    elif args.alg=='ppo':
        from rl.ppo.train_ppo import train_ppo
        # todo: mimic td3 modifications
        raise NotImplemented
        train_ppo(env, envs, args.train, args.test, args.finetune, args.path, args.model_id, args.render, args.process, args.seed)
    else:
        print("Algorithm type is not implemented!")