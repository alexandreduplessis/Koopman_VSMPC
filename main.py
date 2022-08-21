#!/local_scratch/adupless/test_env/bin/python
from src.control_step import control_step
from src.env.environment import SimulationEnv
from src.model import Autoencoder, Autoencoder_Dense
from src.utils import Last_cumulated, lr_function
from src.parser import Parser
from src.env.vs_env import VsEnv
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


def control(env, initial_position, goal_position, nb_steps, nb_epochs_list, horizon, AB_horizon, obs_dim, embed_dim, control_dim, rho, secondary_horizon, tensorboard, alpha, beta, learning_rate, random_control, show_model):
    d = Last_cumulated(m=horizon, h=secondary_horizon, T=AB_horizon, length=obs_dim, width = obs_dim)
    u = Last_cumulated(m=horizon-1, h=0, T=AB_horizon, length=control_dim)
    # d.append(initial_position)
    if random_control:
        env.reset()
        state_dict = {}
        control_dict = {}
        d.append(torch.from_numpy(env.current_state_matrix()))
        state_dict[0] = torch.from_numpy(env.current_state_matrix())
    # env.env_initialize(initial_position)
        for t in range(1, horizon + secondary_horizon):
            # state, control = env.env_random_control(device)
            state, reward, done, info = env.step()
            env.render()
            d.append(torch.from_numpy(env.observation_to_matrix(state))) # poorly efficient, modify append method to take a list of states
            state_dict[t] = torch.from_numpy(env.observation_to_matrix(state))
            # print("Random state {} : {}".format(t, state))
            u.append(torch.from_numpy(info['action']))
            control_dict[t] = torch.from_numpy(info['action'])
            print("Random control {} : reward = {}".format(t, reward), end="\r")
        torch.save(state_dict, './output/tensors_backup/state_dict.pt')
        torch.save(control_dict, './output/tensors_backup/control_dict.pt')
        print("Random control finished")
    else:
        state_dict = torch.load('./output/tensors_backup/state_dict.pt')
        control_dict = torch.load('./output/tensors_backup/control_dict.pt')
        d.append(state_dict[0])
        first_state_environment = env.matrix_to_observation(state_dict[0].numpy())
        env.reset(x=first_state_environment['abscisse'], y=first_state_environment['ordonnee'], depth=first_state_environment['depth'])
        env.render()
        state = env.current_state_matrix()
        for t in range(1, horizon + secondary_horizon):
            d.append(state_dict[t])
            u.append(control_dict[t])
        print("Random control historic loaded")
            
    # print("First state :", state)
    model = Autoencoder(args.embed_dim)
    model.to(device)
    if show_model:
        print("====================")
        print("Model:")
        print("Number of learnable parameters:", model.number_parameters())
        print(model)
        print("====================")
    optimizer = torch.optim.Adam(
        model.parameters(), lr=learning_rate)
    # scheduler = 0.
    # scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_function)
    # scheduler = torch.optim.lr_scheduler.StepLR(env = VsEnv()
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
            d.append(torch.from_numpy(env.current_state_matrix()))
            tb.add_scalar("Reward", reward, i)
            # print("with control ", control)
            u.append(control.unsqueeze(-1))
            current_state = new_state
    return d


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
    # print("goal pos :", hand_goal([ 18494.5254, -13808.1035,   7870.1992],im)
    
    env = VsEnv(length=28, width=28, max_steps=args.learning_horizon + args.secondary_horizon + args.steps)
    print("Simulation environment created")

    res = control(
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
        learning_rate=args.lr,
        random_control=args.random_control,
        show_model=args.show_model)
    print("Done")
    tb.close()
