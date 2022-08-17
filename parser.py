import numpy as np
import random
import argparse
import os
import torch


class Parser(argparse.ArgumentParser):
    def __init__(self):
        super(Parser, self).__init__(description='Read')
        
        # hyperparameters
        self.add_argument('--epochs', '-e', type=int, default=100, help="number of epochs, constant over steps")
        self.add_argument('--steps', '-s', type=int, default=5, help="number of training data")
        self.add_argument('--alpha', '-a', type=float, default=1., help='weight of auto_loss in loss')
        self.add_argument('--beta', '-b', type=float, default=1., help='weight of pred_loss in loss')
        self.add_argument('--regularization', '-rho', type=float, default=1e4, help='ADAM weight decay')

        # learning
        self.add_argument('--learning_horizon', '-T', type=int, default=500, help='horizon of learning')
        self.add_argument('--AB_horizon', '-m', type=int, default=400, help='number of values used to compute A and B')
        self.add_argument('--lr', '-lr', type=float, default=1e-3, help='ADAM learning rate')
        self.add_argument('--weight_decay', '-wd', type=float, default=0., help='ADAM weight decay')
        self.add_argument('--embed_dim', '-n', type=int, default=8, help='dimension of latent space')
        self.add_argument('--secondary_horizon', '-sh', type=int, default=0, help='time horizon for dependancy')
        
        # environment
        self.add_argument('--init_pos', '-d0', type=int, default=random.uniform(-20., 20.), help='initial position')
        self.add_argument('--goal_pos', '-dstar', type=int, default=random.uniform(-20., 20.), help='desired position')
        self.add_argument('--obs_dim', '-o', type=int, default=28, help='dimension of observation space')
        self.add_argument('--control_dim', '-c', type=int, default=3, help='dimension of control space')
        
        #random
        self.add_argument('--seed', '-seed', type=int, default=None, help='set seed or not')

    def parse(self, dirs=True):
        args = self.parse_args()
        
        if args.seed is None:
            args.seed = random.randint(1, 10000)
        
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        np.random.seed(seed=args.seed)

        return args