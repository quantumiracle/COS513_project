from functools import total_ordering
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F
import torch.nn as nn
import os

import pyro
import torch.nn as nn
from pyro.infer.autoguide import AutoNormal, AutoDiagonalNormal
from pyro.infer import SVI, Trace_ELBO, Predictive

from networks import DynamicsEncoder, EmbeddingDynamicsNetwork, DynamicsParamsOptimizer, EmbeddingFit
import argparse
parser = argparse.ArgumentParser(description='Arguments.')


# device = 'cpu'
# device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
device = 'cuda:0'

def load_data(env_name):
    data_path = '../data/dynamics_data/'+env_name+'/dynamics.npy'
    train_data = np.load(data_path, allow_pickle=True)
    print('number of samples in data: ', len(train_data))
    # split data
    data_s, data_a, data_param, data_s_ = [], [], [], []
    for d in train_data:
        [s,a,param], s_ = d
        data_s.append(s)
        data_a.append(a)
        data_param.append(param)
        data_s_.append(s_)

    data_s = np.array(data_s)
    data_a = np.array(data_a)
    data_param = np.array(data_param)
    data_s_ = np.array(data_s_)

    print(f'Data shape for training: state: {data_s.shape}, action: {data_a.shape}, parameter: {data_param.shape}, next_state:{data_s_.shape}')

    return data_s, data_a, data_param, data_s_

def load_test_data(env_name, test_idx):
    # test_idx: index of sample to test: 0-10
    # load test data
    test_data_path = '../data/dynamics_data/'+env_name+'/test_dynamics.npy'
    test_data = np.load(test_data_path, allow_pickle=True)
    print('number of samples in dest data: ', len(test_data))
    test_s = np.array(test_data[test_idx]['sa'])[:, :-1]
    test_a = np.array(test_data[test_idx]['sa'])[:, -1:]
    test_param = np.array(test_data[test_idx]['params'])
    test_s_ = np.array(test_data[test_idx]['s_'])

    print(f'Data shape for testing: state: {test_s.shape}, action: {test_a.shape}, parameter: {test_param.shape}, next_state:{test_s_.shape}')

    return test_s, test_a, test_param, test_s_

def compare_plot(pre_mean, pre_var, true, idx=2):
    pre_mean = np.array(pre_mean)
    pre_var = np.array(pre_var)
    values = np.concatenate([pre_mean, true])
    max_x = np.max(values[:,0])+0.2
    min_x = np.min(values[:,0])-0.2
    max_y = np.max(values[:,1])+0.2
    min_y = np.min(values[:,1])-0.2

    for i, (m, v, t) in enumerate(zip(pre_mean, pre_var, true)):
        plt.scatter(*t[:idx], s=80, label=i, alpha=0.7)
        plt.errorbar(*m[:idx], xerr=v[0], yerr=v[1], fmt="o", capsize=6)
    plt.xlim(min_x, max_x)
    plt.ylim(min_y, max_y)
    plt.savefig('compare.png')
    plt.legend()
    plt.show()

if __name__ == "__main__":
    parser.add_argument('--train', dest='train', action='store_true', default=False)
    parser.add_argument('--eval', dest='eval', action='store_true', default=False)
    args = parser.parse_args()

    env_name = 'inverteddoublependulum'
    data_s, data_a, data_param, data_s_ = load_data(env_name)

    x = np.concatenate((data_s,data_a), axis=-1)
    theta = data_param
    y = data_s_
    print(x.shape, y.shape)

    s_dim = data_s.shape[-1]
    a_dim = data_a.shape[-1]
    param_dim = data_param.shape[-1]
    latent_dim = 3
    hidden_layers = 3
    lr = 0.001
    svi_lr = 0.01   
    hidden_dim = 32
    batch = 10000
    total_epochs = 500000
    lr_schedule_step = int(total_epochs/10)  # step the scheduler every n epochs
    print('parameter dimension: ', param_dim)

    if args.train:
        #stage 1, learning forward dynamics and dynamics encoder
        opt = DynamicsParamsOptimizer(s_dim, a_dim, param_dim, latent_dim, hidden_dim=hidden_dim, num_hidden_layers=hidden_layers, batch=batch, lr=lr, device=device)
        data = (x, theta, y)
        model_save_path = './model/test/'
        os.makedirs(model_save_path, exist_ok=True)
        opt.update(data, epoch=total_epochs, lr_schedule_step=lr_schedule_step, model_save_path=model_save_path)

    if args.eval:
        # load trained dynamics model
        dynamics_model = EmbeddingDynamicsNetwork(s_dim, a_dim, latent_dim, hidden_dim=hidden_dim, hidden_activation=F.relu, output_activation=None, num_hidden_layers=hidden_layers, lr=lr, gamma=0.99).to(device)
        model_save_path = './model/test/'
        dynamics_model.load_state_dict(torch.load(model_save_path+'dynamics_model', map_location=device))
        dynamics_model.eval()
        for name, param in dynamics_model.named_parameters():
            param.requires_grad = False  # this is critical! set not gradient for the trained model, otherwise will be updated in Pyro
        dynamics_encoder = DynamicsEncoder(param_dim, latent_dim, hidden_dim=hidden_dim, hidden_activation=F.relu, output_activation=None, num_hidden_layers=hidden_layers, lr=lr, gamma=0.99).to(device)
        model_save_path = './model/test/'
        dynamics_encoder.load_state_dict(torch.load(model_save_path+'dynamics_encoder', map_location=device))
        # for name, param in dynamics_encoder.named_parameters():
        #     if param.requires_grad:
        #         print(name, param.data)

        #stage 2, using BNN and SVI to fit alpha
        pre_means = []
        pre_vars = []
        true_vals = []
        for idx in range(1,5):
            test_s, test_a, test_param, test_s_ = load_test_data(env_name, idx)
            test_num_samples = 1000
            test_x = torch.from_numpy(np.concatenate((test_s,test_a), axis=-1)).float()[:test_num_samples]
            test_y = torch.from_numpy(test_s_).float()[:test_num_samples]
            test_param = torch.from_numpy(test_param).float()
            x_dim = test_x.shape[1]
            y_dim = test_y.shape[1]

            # fit unknown parameters
            pyro.clear_param_store() # this is important in notebook; elease memory!
            # pyro.set_rng_seed(1)
            model = EmbeddingFit(latent_dim, dynamics_model)
            guide = AutoDiagonalNormal(model)  # posterior dist. before learning AutoDiagonalNormal
            svi = SVI(model, guide, pyro.optim.Adam({"lr": svi_lr}), Trace_ELBO())  # parameters to optimize are determined by guide()

            for step in range(10):
                loss = svi.step(test_x, test_y) / test_y.numel()  # data in step() are passed to both model() and guide()
                
                if step % 1000 == 0:
                    print("step {} loss = {:0.4g}".format(step, loss))
            for name, value in pyro.get_param_store().items():
                print(name, pyro.param(name))
                if 'loc' in name:
                    pred_encoding_mean = pyro.param(name).detach().cpu().numpy()
                    pre_means.append(pred_encoding_mean)
                elif 'scale' in name:
                    pred_encoding_var = pyro.param(name).detach().cpu().numpy()
                    pre_vars.append(pred_encoding_var)
            print(model.sigma)

            # get true value
            true_encoding = dynamics_encoder(test_param).detach().cpu().numpy()
            true_vals.append(true_encoding)
            print(f'true params: {test_param}, true encoding: {true_encoding}')

        compare_plot(pre_means, pre_vars, true_vals)
