#!/local_scratch/adupless/test_env/bin/python
from control_step import control_step
from environment import SimulationEnv
from model import Autoencoder, Autoencoder_Dense
from utils import Last_cumulated, lr_function
from parser import Parser
from vs_env import VsEnv
import random
import numpy
import gym
import math
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

if torch.cuda.is_available():
  dev = "cuda:0"
else:
  dev = "cpu"

device = torch.device(dev)


def control(env, initial_position, goal_position, nb_steps, nb_epochs_list, horizon, AB_horizon, obs_dim, embed_dim, control_dim, rho, secondary_horizon, tensorboard, alpha, beta, learning_rate):
    save_value = []
    d = Last_cumulated(m=horizon, h=secondary_horizon, T=AB_horizon, length=obs_dim, width = obs_dim)
    # d.append(initial_position)
    d.append(torch.from_numpy(env.current_state_matrix()))
    u = Last_cumulated(m=horizon-1, h=0, T=AB_horizon, length=control_dim)
    # env.env_initialize(initial_position)
    for t in range(1, horizon + secondary_horizon):
        # state, control = env.env_random_control(device)
        state, reward, done, info = env.step()
        env.render()
        d.append(torch.from_numpy(env.observation_to_matrix(state)))
        # print("Random state {} : {}".format(t, state))
        u.append(torch.from_numpy(info['action']))
        # print("Random control {} : {}".format(t, control))

    # print("First state :", state)
    model = Autoencoder(args.embed_dim)
    model.to(device)
    print("====================")
    print("Model:")
    print("Number of learnable parameters:", model.number_parameters())
    print(model)
    print("====================")
    optimizer = torch.optim.Adam(
        model.parameters(), lr=learning_rate, weight_decay=1e-4)
    # scheduler = 0.
    # scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_function)
    # scheduler = torch.optim.lr_scheduler.StepLR(
    #     optimizer, step_size=100, gamma=0.5)
    # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min')
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.995)


    current_state = state

    for i in range(1, nb_steps+1):
        control, model = control_step(current_state, goal_position, nb_epochs_list[i-1], d, u, embed_dim, control_dim,
                                      AB_horizon, horizon, rho, model, optimizer, tb, i, device, alpha, beta, scheduler=scheduler, secondary_horizon=secondary_horizon)
        with torch.no_grad():
            # new_state = env.env_control(control)
            control_env = control.detach().cpu().numpy()
            print(control)
            new_state, reward, done, info = env.step({'abscisse': control_env[0], 'ordonnee': control_env[1], 'depth': control_env[2]})
            env.render()
            print("Round {} : {}".format(i, reward))
            d.append(new_state)
            tb.add_scalar("Reward", reward, i)
            save_value.append(new_state.mean())
            # print("with control ", control)
            u.append(control)
            current_state = new_state
    return d, save_value


if __name__ == "__main__":
    print("Starting...")
    
    args = Parser().parse()
    # tb = SummaryWriter("Experiment_lr_{}_rho_{}_odim_{}_edim_{}_epochs_{}_hor_{}".format(args.lr, args.regularization, args.obs_dim, args.embed_dim, args.epochs, args.learning_horizon))
    tb = SummaryWriter()
    
    assert args.learning_horizon >= args.AB_horizon
    
    # hand_init_pos = torch.tensor([1. for _ in range(args.obs_dim)]).to(device)
    # print("init pos :", hand_init_pos)
    # hand_goal_pos = torch.tensor([i*1.+17. for i in range(args.obs_dim)]).to(device)
    # hand_goal_pos = torch.tensor([2. for i in range(args.obs_dim)]).to(device)
    # print("goal pos :", hand_goal_pos)
    
    env = VsEnv(length=args.obs_dim, width=args.obs_dim, max_steps=args.learning_horizon + args.secondary_horizon + args.steps)
    env.reset()
    
    # env = SimulationEnv(args.obs_dim)

    res, save_value = control(
        env=env,
        initial_position=torch.from_numpy(env.initial_matrix()).to(device),
        goal_position=torch.from_numpy(env.goal_matrix()).to(device),
        nb_steps=args.steps,
        nb_epochs_list=[args.epochs]*(args.steps),
        horizon=args.learning_horizon,
        AB_horizon=args.AB_horizon,
        obs_dim=args.obs_dim,
        embed_dim=args.embed_dim,
        control_dim=args.control_dim,
        rho=args.regularization,
        secondary_horizon=args.secondary_horizon,
        tensorboard=tb,
        alpha=args.alpha,
        beta=args.beta,
        learning_rate=args.lr)
    
    # plt.plot(save_value)
    # plt.show()
    # print("res :{}".format(res))
    tb.close()