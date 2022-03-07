import isaacgym # before torch!
import gym
import argparse
from gym import envs
import os
from rl.td3.train_td3 import train_td3
from rl.ppo.train_ppo import train_ppo

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
    args = parser.parse_args()

    MUJOCO_ENVS = ['acrobot', 'cartpole', 'mountaincar', 'pendulum', 'inverteddoublependulumdisc', 'inverteddoublependulum', \
        'inverteddoublependulumdynamics', 'inverteddoublependulumdynamicsembedding', 'halfcheetah', 'halfcheetahdynamics', 'halfcheetahdynamicsembedding']
    envs = envs.registry.all()
    GYM_MOJUCO_ENVS = [env_spec.id for env_spec in envs]
    ISAAC_ENVS = ['frankacabinet']

    if args.env in GYM_MOJUCO_ENVS:
        env = gym.make(args.env).unwrapped  # original envs from gym mujoco
    elif args.env in (MUJOCO_ENVS):
        from environment import envs
        env = envs[args.env]()
    elif args.env in ISAAC_ENVS:
        import yaml
        with open('environment/isaacgymenvs/cfg/config.yaml') as f:
            yaml_cfg = yaml.safe_load(f)
        task_params, sim_device, graphics_device_id, headless = yaml_cfg['task'], yaml_cfg['sim_device'], yaml_cfg['graphics_device_id'], yaml_cfg['headless']
        from environment.isaacgymenvs.tasks.franka_cabinet import FrankaCabinet
        env = FrankaCabinet(cfg=task_params, sim_device=sim_device, graphics_device_id=graphics_device_id, headless=headless)
        assert args.alg=='td3' and args.process==1
    else:
        print('Environment {} not exists!'.format(args.env))
    print('Environment Name:', args.env)
    print('Observation space: {}  Action space: {}'.format(env.observation_space, env.action_space))

    if args.alg=='td3':
        train_td3(env, envs, args.train, args.test, args.finetune, args.path, args.model_id, args.render, args.process, args.seed)
    elif args.alg=='ppo':
        train_ppo(env, envs, args.train, args.test, args.finetune, args.path, args.model_id, args.render, args.process, args.seed)
    else:
        print("Algorithm type is not implemented!")